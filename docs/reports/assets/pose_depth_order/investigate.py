import collections
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from depth_anything_3.api import DepthAnything3

ROOT = Path.cwd()
OUT = ROOT / 'tmp/depth_order'
CACHE = Path('/storage/huggingface/hub/models--depth-anything--DA3-BASE/snapshots/f4a6c9b3c95e41c82048423d3493a81ec3fa810e')

def sha(path):
    return hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sample(depth, xy, width, height):
    h, w = depth.shape
    x = np.clip((xy[:, 0] + .5) * w / width - .5, 0, w - 1)
    y = np.clip((xy[:, 1] + .5) * h / height - .5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = x - x0, y - y0
    return (depth[y0, x0] * (1-wx) * (1-wy) + depth[y0, x1] * wx * (1-wy)
            + depth[y1, x0] * (1-wx) * wy + depth[y1, x1] * wx * wy)

def visible_values(depth, ann, width, height, flipped=False):
    points = np.array(ann['keypoints'], dtype=float).reshape(14, 3)
    mask = ((points[:, 2] == 2) & np.isfinite(points[:, :2]).all(1)
            & (points[:, 0] >= 0) & (points[:, 0] < width)
            & (points[:, 1] >= 0) & (points[:, 1] < height))
    xy = points[mask, :2].copy()
    if flipped:
        xy[:, 0] = width - 1 - xy[:, 0]
    return sample(depth, xy, width, height)

def checks():
    depth = np.arange(20, dtype=np.float32).reshape(4, 5)
    xy = np.array([[0, 0], [1.2, 2.1], [4, 3]], dtype=float)
    ours = sample(depth, xy, 5, 4)
    grid = torch.tensor((xy + .5) / [5, 4] * 2 - 1, dtype=torch.float32)[None, None]
    ref = torch.nn.functional.grid_sample(torch.tensor(depth)[None, None], grid, align_corners=False, padding_mode='border').flatten().numpy()
    assert np.allclose(ours, ref, atol=1e-5)
    depth[3, 4] = 1e12
    points = np.zeros((14, 3)); points[:3] = [[1, 1, 2], [4, 3, 1], [4, 3, 0]]
    ann = {'keypoints':points.flatten().tolist()}
    assert visible_values(depth, ann, 5, 4).tolist() == [6.]
    points[0, 2] = 1; ann['keypoints'] = points.flatten().tolist()
    assert visible_values(depth, ann, 5, 4).size == 0
    assert np.argsort([3, 1, 2]).tolist() == [1, 2, 0]
    return {'visible_only_exclusion':True, 'all_invisible_undefined':True, 'independent_grid_sample':True, 'near_to_far_sort':True}

coverage = {}
by_image = None
for split in ['train', 'val']:
    path = ROOT / f'data/crowdpose/annotations/crowdpose_{split}.json'
    data = json.loads(path.read_text())
    anns = [a for a in data['annotations'] if not a.get('iscrowd', 0) and a.get('num_keypoints', 0)>0]
    grouped = collections.defaultdict(list)
    for a in anns:
        grouped[a['image_id']].append(a)
    sizes = {x['id']:(x['width'],x['height']) for x in data['images']}
    counts = collections.Counter(); invalid = 0; complete = 0; none = 0
    for image in data['images']:
        aa = grouped[image['id']]; n_defined = 0
        for a in aa:
            p = np.array(a['keypoints']).reshape(14, 3); w,h=sizes[image['id']]
            raw = p[:,2] == 2
            valid = raw & np.isfinite(p[:,:2]).all(1) & (p[:,0]>=0) & (p[:,0]<w) & (p[:,1]>=0) & (p[:,1]<h)
            invalid += int((raw & ~valid).sum()); counts[int(valid.sum())] += 1; n_defined += bool(valid.any())
        complete += bool(aa) and n_defined == len(aa)
        none += n_defined == 0
    coverage[split] = dict(sha256=sha(path), images=len(data['images']),valid_people=len(anns), visible_count_hist=dict(sorted(counts.items())), invalid_visible_coordinates=invalid, complete_rank_images=complete, no_rankable_images=none)
    if split == 'val':
        by_image=grouped; images={x['id']:x for x in data['images']}

rng=np.random.default_rng(42)
ids=sorted(rng.choice(sorted(images),size=32,replace=False).tolist())
manifest=dict(teacher='DA3-BASE', teacher_revision=CACHE.name, teacher_weight_sha256=sha(CACHE/'model.safetensors'), source_commit='3d835ec1a5802d64a8b8b15f817a1ab54809bfe4', seed=42, image_ids=ids, single_image_independent_inference=True, primary_resolution=504, secondary_resolution=420, sampling='half-pixel bilinear, border at depth-map edge, original in-bounds visible==2 only', checks=checks(), coverage=coverage)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('COVERAGE',json.dumps(coverage),flush=True)
torch.set_num_threads(4)
model=DepthAnything3.from_pretrained(str(CACHE)).to('cuda').eval()
rows=[]; pairs=[]; sensitivity=[]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
for idx,image_id in enumerate(ids):
    im=images[image_id]; path=ROOT/'data/crowdpose/images'/im['file_name']; rgb=Image.open(path).convert('RGB'); w,h=rgb.size
    assert (w,h)==(im['width'],im['height'])
    baseline=model.inference([rgb],process_res=504).depth[0]
    flipped=model.inference([rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)],process_res=504).depth[0]
    resized=model.inference([rgb],process_res=420).depth[0]
    assert all(np.isfinite(x).all() for x in [baseline,flipped,resized])
    np.savez_compressed(OUT/f'{image_id}_maps.npz',baseline=baseline,flipped=flipped,resized=resized)
    people=[]
    for a in by_image[image_id]:
        vals=visible_values(baseline,a,w,h)
        fv=visible_values(flipped,a,w,h,True); rv=visible_values(resized,a,w,h)
        mean=lambda x:float(x.mean()) if x.size else None
        row=dict(image_id=image_id,ann_id=a['id'],visible_count=len(vals),depth_mean=mean(vals),flip_mean=mean(fv),resize_mean=mean(rv),joint_depths=vals.tolist(),bbox=a['bbox'])
        people.append(row)
    defined=sorted([p for p in people if p['depth_mean'] is not None],key=lambda p:(p['depth_mean'],p['ann_id']))
    for rank,p in enumerate(defined): p['rank']=rank+1
    for a,b in itertools.combinations(defined,2):
        gap=b['depth_mean']-a['depth_mean']; scale=max(abs(a['depth_mean']),abs(b['depth_mean']),1e-8)
        one=dict(image_id=image_id,a=a['ann_id'],b=b['ann_id'],relative_gap=gap/scale,flip_changed=bool(a['flip_mean']>b['flip_mean']),resize_changed=bool(a['resize_mean']>b['resize_mean']))
        pairs.append(one)
        for p,other in [(a,b),(b,a)]:
            v=np.array(p['joint_depths'])
            if len(v)>1:
                leave=(v.sum()-v)/(len(v)-1)
                changed=((leave < other['depth_mean']) != (p['depth_mean'] < other['depth_mean']))
                sensitivity.extend(bool(x) for x in changed)
    rows.extend(people)
    if idx < 8:
        thumb=rgb.copy(); thumb.thumbnail((600,600)); draw=ImageDraw.Draw(thumb); sx,sy=thumb.width/w,thumb.height/h
        for p,a in zip(people,by_image[image_id]):
            color=(255,210,0) if 'rank' in p else (255,90,90)
            pts=np.array(a['keypoints']).reshape(14,3)
            for x,y,v in pts:
                if v==2 and 0<=x<w and 0<=y<h: draw.ellipse((x*sx-3,y*sy-3,x*sx+3,y*sy+3),fill=color)
            x,y,bw,bh=p['bbox'];draw.rectangle((x*sx,y*sy,(x+bw)*sx,(y+bh)*sy),outline=color,width=2)
            text=f"r{p.get('rank','?')} v{p['visible_count']} id{p['ann_id']}"
            draw.text((x*sx,max(0,y*sy-18)),text,font=font,fill=color,stroke_width=1,stroke_fill='black')
        canvas=Image.new('RGB',(620,650),'#16191f');canvas.paste(thumb,(10,40));ImageDraw.Draw(canvas).text((10,10),f'VAL {image_id} | near -> far; ? = undefined',font=font,fill='white');canvas.save(OUT/f'{image_id}_overlay.jpg')
    print(f'PROGRESS {idx+1}/32 image={image_id} people={len(people)} defined={len(defined)}',flush=True)

summary=dict(images=32,people=len(rows),defined_people=sum(p['depth_mean'] is not None for p in rows),undefined_people=sum(p['depth_mean'] is None for p in rows),comparable_pairs=len(pairs),flip_changed_pairs=sum(p['flip_changed'] for p in pairs),resize_changed_pairs=sum(p['resize_changed'] for p in pairs),leave_one_out_comparisons=len(sensitivity),leave_one_out_changed=sum(sensitivity),near_pair_threshold=.05,near_pairs=sum(p['relative_gap']<.05 for p in pairs),nonnear_pairs=sum(p['relative_gap']>=.05 for p in pairs),nonnear_flip_changed=sum(p['flip_changed'] for p in pairs if p['relative_gap']>=.05),nonnear_resize_changed=sum(p['resize_changed'] for p in pairs if p['relative_gap']>=.05))
(OUT/'results.json').write_text(json.dumps(dict(summary=summary,people=rows,pairs=pairs),indent=2))
print('SUMMARY',json.dumps(summary),flush=True)
