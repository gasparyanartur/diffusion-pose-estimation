from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scripts.investigation.information_suite.geometry import camera_project
from scripts.investigation.information_suite.scoring import score_people


def score_matching_only(result, keypoints, matching_boxes, original_boxes):
    records = score_people(result, keypoints, matching_boxes)
    for record, box in zip(records, original_boxes):
        record['matching_box_tau'] = record['tau']
        record['tau'] = .04 * float(np.sqrt(box[2] * box[3]))
        record['wrist_correct'] = bool(
            record['detected'] and record['wrist_error_pixels'] <= record['tau'])
    return records


def shared_mesh_boxes(directory,style):
    scenes={r['condition']:r for r in json.loads((directory/'scenes.json').read_text())}
    values=[]
    for member in (0,1):
        record=scenes[f'{style}_{member}']
        vertices=np.load(directory/record['mesh'])['vertices']
        values.append([camera_project(v,record['scale'],[192,128]) for v in np.split(vertices,2)])
    boxes=[]
    for person in (0,1):
        points=np.concatenate([v[person] for v in values])
        low=np.maximum(points.min(0)-1,[0,0])
        high=np.minimum(points.max(0)+1,[384,256])
        boxes.append(np.r_[low,high-low].tolist())
    return boxes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    manifest=json.loads((root/'pilot_manifest.json').read_text())
    originals={family['family_id']:family for family in manifest['families']}
    output=root/'matching_diagnostic';output.mkdir(exist_ok=True)
    revised=copy.deepcopy(manifest)
    revised['phase']='corrected_box_diagnostic; not retroactive admission'
    revisions={}
    for family in revised['families']:
        directory=Path(family['conditions']['main_0']['path']).parent
        boxes={style:shared_mesh_boxes(directory,style) for style in ('main','nuisance','clear','visible')}
        for condition,entry in family['conditions'].items():
            style='main' if condition=='neutral' else condition.split('_')[0]
            entry['boxes']=boxes[style]
        revisions[family['family_id']]=family
    (output/'pilot_manifest.json').write_text(json.dumps(revised,indent=2)+'\n')
    native=output/'native';native.mkdir(exist_ok=True)
    summary={}
    for arm in ('control','h'):
        rows=[json.loads(line) for line in (root/'native'/f'{arm}.jsonl').read_text().splitlines()]
        for row in rows:
            entry=revisions[row['family_id']]['conditions'][row['condition']]
            original=originals[row['family_id']]['conditions'][row['condition']]
            row['people']=score_matching_only(row['predictions'],entry['keypoints'],entry['boxes'],original['boxes'])
            if row['condition']=='neutral':
                row['alternative_people']=score_matching_only(row['predictions'],entry['alternative_keypoints'],entry['boxes'],original['boxes'])
        (native/f'{arm}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        summary[arm]={}
        for style in ('clear','visible'):
            selected=[r['people'][r['target_person']] for r in rows if r['condition'].startswith(style+'_')]
            summary[arm][style]={'images':len(selected),'detected':sum(r['detected'] for r in selected),'correct':sum(r['wrist_correct'] for r in selected)}
    (output/'diagnostic_summary.json').write_text(json.dumps({'note':'Cached-prediction matching-only audit using deterministic within-pair mesh-union bounds for IoU. Confidence/IoU thresholds and original absolute pixel PCK tolerances preserved. Original admission failure remains recorded.','models':summary},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
