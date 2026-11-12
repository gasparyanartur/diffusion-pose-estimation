import builtins
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

from ..diffusionregpose.evaluation import box_cxcywh_to_xyxy
from .quick_eval_v13 import evaluation_context, select_test_images
from .quick_eval_v21 import v21_test_batches


def postprocess_boxes(output, target_sizes):
    height, width = target_sizes.unbind(-1)
    scale = torch.stack((width, height, width, height), -1)
    boxes = box_cxcywh_to_xyxy(output['pred_boxes'])*scale[:, None]
    return [dict(boxes=b, scores=s.sigmoid(), labels=torch.ones_like(s, dtype=torch.long))
            for b, s in zip(boxes, output['existence_logits'])]


def box_records(image_id, result):
    records = []
    for box, score in zip(result['boxes'], result['scores']):
        x1, y1, x2, y2 = box.detach().cpu().tolist()
        records.append(dict(image_id=int(image_id), category_id=1,
            bbox=[x1, y1, x2-x1, y2-y1], score=float(score)))
    return records


def evaluate_box_records(records, annotations, image_ids):
    if not records:
        return dict(AP=0., AP50=0., AP75=0., AR=0.)
    from crowdposetools.coco import COCO
    from crowdposetools.cocoeval import COCOeval

    had_unicode, had_float = hasattr(builtins, 'unicode'), hasattr(np, 'float')
    if not had_unicode:
        builtins.unicode = str
    if not had_float:
        np.float = float
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(annotations))
            selected = set(image_ids)
            gt.dataset['images'] = [r for r in gt.dataset['images'] if r['id'] in selected]
            gt.dataset['annotations'] = [r for r in gt.dataset['annotations'] if r['image_id'] in selected]
            gt.createIndex()
            detected = COCO.loadRes(gt, records)
            evaluator = COCOeval(gt, detected, 'bbox')
            evaluator.params.imgIds = list(image_ids)
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        return dict(AP=float(evaluator.stats[0]), AP50=float(evaluator.stats[1]),
                    AP75=float(evaluator.stats[2]), AR=float(evaluator.stats[8]))
    finally:
        if not had_unicode:
            del builtins.unicode
        if not had_float:
            del np.float


def box_coverage(output, batch):
    targets = batch['gt_boxes_canon'][0]
    valid = batch['person_mask'][0].bool() & (targets[:, 2:] > 0).all(-1)
    targets = targets[valid]
    scores = output['existence_logits'][0].sigmoid()
    iou = box_iou(box_cxcywh_to_xyxy(output['pred_boxes'][0]), box_cxcywh_to_xyxy(targets))
    raw = (iou >= .5).any(0).sum().item()
    confident = ((iou >= .5) & (scores[:, None] >= .5)).any(0).sum().item()
    return dict(gt_people=len(targets), covered_iou50=raw,
                covered_iou50_score50=confident, accepted_predictions=int((scores >= .5).sum()))


@torch.no_grad()
def run_v35_evaluation(model, data, root, *, epoch, global_step, count=32, seed=42):
    from src.data.datasets import PoseDataset

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = PoseDataset.from_path(data.test_ann_path, img_root=data.img_dir, num_keypoints=data.num_keypoints)
    candidates = sorted(int(sample['image_id']) for sample in dataset.samples)
    image_ids = candidates if count is None else select_test_images(candidates, root/'subset.json',
        count=count, seed=seed, annotations=data.test_ann_path)
    directory = root/f'epoch{epoch:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    conditions = {f'noise_var_100/steps_{k}': dict(steps=k, records=[], diagnostics=dict(
        gt_people=0, covered_iou50=0, covered_iou50_score50=0, accepted_predictions=0)) for k in (1, 4)}
    old_steps = model.flow_sampling_steps
    seen = []
    try:
        with evaluation_context(model):
            for batch in v21_test_batches(data, dataset, image_ids, next(model.parameters()).device):
                image_id = int(batch['image_id'][0]); seen.append(image_id)
                for row in conditions.values():
                    model.flow_sampling_steps = row['steps']
                    generator = torch.Generator(device=batch['image'].device).manual_seed(seed+image_id)
                    output = model.sample_boxes(batch['image'], padding_mask=batch.get('padding_mask'), generator=generator)
                    if not all(torch.isfinite(output[k]).all() for k in ('pred_boxes', 'existence_logits')):
                        raise ValueError('Nonfinite V35 box predictions')
                    result = postprocess_boxes(output, batch.get('original_img_hw', batch['img_hw']).to(batch['image']))[0]
                    row['records'].extend(box_records(image_id, result))
                    for name, value in box_coverage(output, batch).items():
                        row['diagnostics'][name] += value
                if len(seen) % 100 == 0 or len(seen) == len(image_ids):
                    progress = dict(completed=len(seen), total=len(image_ids))
                    (directory/'progress.json').write_text(json.dumps(progress)+'\n')
                    print('V35 BOX EVALUATION', json.dumps(progress), flush=True)
    finally:
        model.flow_sampling_steps = old_steps
    if seen != image_ids:
        raise ValueError('V35 evaluation image coverage mismatch')
    for row in conditions.values():
        records = row.pop('records')
        path = directory/f"k{row['steps']}_boxes.json"
        path.write_text(json.dumps(records, allow_nan=False))
        row['metrics'] = evaluate_box_records(records, data.test_ann_path, image_ids)
        row['predictions'] = str(path)
        row['detections'] = len(records)
    report = dict(epoch=epoch, global_step=global_step, image_ids=image_ids, images=len(seen),
        annotations=str(data.test_ann_path), seed=seed, per_image_seed='seed + image_id',
        resize=[data.eval_resize_scale, data.eval_max_size], sampler=model.sampler_kind,
        precision='FP32', matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32, postprocessing='all queries; no NMS; person label1',
        scope='full held-out test' if count is None else f'fixed{len(seen)} held-out diagnostic',
        conditions=conditions)
    (directory/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
