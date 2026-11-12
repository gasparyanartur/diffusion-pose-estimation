from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


def interval(values):
    values = np.asarray(values,dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(20261023)
    means = values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'families':len(values)}


def cue_check(manifest):
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(repo/'tmp/h_control_information_suite/deps'))
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    families = manifest['families']
    images, labels, groups = [], [], []
    for index,family in enumerate(families):
        for member in (0,1):
            path = family['conditions'][f'main_{member}']['path']
            image = Image.open(path).convert('RGB').crop((72,36,302,144)).resize((96,48))
            images.append(np.asarray(image,dtype=np.float32).reshape(-1)/255)
            labels.append(member);groups.append(index)
    images,labels,groups = np.array(images),np.array(labels),np.array(groups)
    predicted = np.zeros(len(labels),dtype=int)
    for fold in range(8):
        test = groups%8==fold
        scaler = StandardScaler().fit(images[~test])
        train = scaler.transform(images[~test])
        pca = PCA(n_components=16,random_state=42,svd_solver='randomized').fit(train)
        model = LogisticRegression(C=1,solver='lbfgs',max_iter=2000,random_state=42)
        model.fit(pca.transform(train),labels[~test])
        predicted[test] = model.predict(pca.transform(scaler.transform(images[test])))
    return {'accuracy':float((predicted==labels).mean()),'correct':int((predicted==labels).sum()),
            'images':len(labels),'folds':8,'split':'whole physical family',
            'passed':bool((predicted==labels).mean()>=.85),
            'local_patch_and_shared_coordinate_accuracy_bound':.5,
            'configuration':'RGB body crop; train-fold standardization/PCA16; C=1 logistic'}


def summarize(root):
    manifest = json.loads((root/'pilot_manifest.json').read_text())
    truth = {family['family_id']:family for family in manifest['families']}
    output = {'phase':manifest.get('phase','association_pilot'),'models':{},'admitted':False}
    all_rows = {}
    for arm in ('control','h'):
        rows = [json.loads(line) for line in (root/'native'/f'{arm}.jsonl').read_text().splitlines()]
        groups = {}
        for row in rows:
            groups.setdefault(row['family_id'],{})[row['condition']] = row
        assert len(groups)==32
        all_rows[arm] = groups
        metrics = {}
        for style in ('clear','visible','main','nuisance'):
            family_rates,paired,counts,errors = [],[],[0,0,0],[]
            by_target = {0:[],1:[]}
            for family,conditions in groups.items():
                if f'{style}_0' not in conditions:
                    continue
                successes=[]
                for member in (0,1):
                    row = conditions[f'{style}_{member}']
                    person=row['people'][row['target_person']]
                    successes.append(person['wrist_correct'])
                    counts[0]+=1;counts[1]+=person['detected'];counts[2]+=person['wrist_correct']
                    by_target[row['target_person']].append(person['wrist_correct'])
                    if person['detected']:
                        errors.append(person['wrist_error_pixels'])
                family_rates.append(np.mean(successes));paired.append(all(successes))
            if counts[0]:
                metrics[style]={'images':counts[0],'detected':counts[1],'correct':counts[2],
                                'per_image_correct':interval(family_rates),'both_members_correct':interval(paired),
                                'median_wrist_error_pixels':float(np.median(errors)),
                                'by_target_person':{str(k):{'correct':int(sum(v)),'images':len(v)} for k,v in by_target.items()}}
        output['models'][arm]=metrics
    if output['phase']=='visible_controls_only':
        output['raw_renderer_competence_passed']=all(output['models'][arm]['clear']['per_image_correct']['mean']>=.85 for arm in output['models'])
        output['reason']='Only raw visible controls evaluated; association composites failed construction validity.'
    else:
        output['RGB_positive_control']=cue_check(manifest)
        output['native_same_occluder_competence_passed']=all(output['models'][arm]['clear']['per_image_correct']['mean']>=.85 for arm in output['models'])
        output['native_visible_competence_passed']=all(output['models'][arm]['visible']['per_image_correct']['mean']>=.85 for arm in output['models'])
        output['admitted']=output['RGB_positive_control']['passed'] and output['native_same_occluder_competence_passed'] and output['native_visible_competence_passed']
        output['reason']='All numerical/cue/native gates passed; visual audit must also be signed off.' if output['admitted'] else 'Admission failed; no main association inference or representation claim is licensed.'
    (root/'pilot_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    return output,all_rows,truth


def gallery(root,all_rows,truth):
    families=list(truth)[:2]
    canvas=Image.new('RGB',(384*3,290*len(families)),'white')
    for number,family in enumerate(families):
        condition='clear_0'
        image=Image.open(truth[family]['conditions'][condition]['path']).convert('RGB')
        target=truth[family]['target_person']
        for column,arm in enumerate(('truth','control','h')):
            picture=image.copy();draw=ImageDraw.Draw(picture)
            point=truth[family]['conditions'][condition]['keypoints'][target][4]
            x,y=point;draw.ellipse((x-3,y-3,x+3,y+3),outline='lime',width=2)
            if arm!='truth':
                row=all_rows[arm][family][condition]['people'][target]
                if row['detected']:
                    x,y=row['predicted_wrist'];draw.ellipse((x-3,y-3,x+3,y+3),outline='red',width=2)
            left,top=384*column,290*number
            canvas.paste(picture,(left,top+25))
            ImageDraw.Draw(canvas).text((left+5,top+5),f'{arm}; family{family}; person{target}',fill='black')
    canvas.save(root/'control_gallery.png')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    summary,rows,truth=summarize(args.root)
    gallery(args.root,rows,truth)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
