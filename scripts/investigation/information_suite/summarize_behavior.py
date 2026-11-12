from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.investigation.information_suite.summarize_pilot import interval
from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import evaluate_crowdpose_results


def load_rows(root, arm):
    path = root/'native'/f'{arm}.jsonl'
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    with gzip.open(root/'features'/f'{arm}_native.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def synthetic_ground_truth(manifest):
    images, annotations, index = [], [], {}
    for family in manifest['families']:
        for condition, entry in family['conditions'].items():
            if condition == 'neutral':
                continue
            image_id = len(images)+1
            index[(family['family_id'], condition)] = image_id
            images.append(dict(id=image_id, width=384, height=256, file_name=entry['path'], crowdIndex=0.))
            for points, box in zip(entry['keypoints'], entry['boxes']):
                joints = np.c_[points, np.r_[np.full(12, 2), 0, 0]]
                annotations.append(dict(id=len(annotations)+1, image_id=image_id, category_id=1,
                    keypoints=joints.reshape(-1).tolist(), num_keypoints=12, bbox=box,
                    area=float(box[2]*box[3]), iscrowd=0))
    return dict(images=images, annotations=annotations, categories=[dict(id=1, name='person', keypoints=[str(i) for i in range(14)], skeleton=[])]), index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.root/'pilot_manifest.json').read_text())
    truth, image_ids = synthetic_ground_truth(manifest)
    annotation_path = args.output.parent/'synthetic_annotations.json'
    annotation_path.write_text(json.dumps(truth)+'\n')
    summary, paired = {}, {}
    for arm in ('control', 'h'):
        rows = load_rows(args.root, arm)
        grouped, detections = {}, []
        for row in rows:
            if row['condition'] == 'neutral':
                continue
            grouped.setdefault(row['family_id'], {})[row['condition']] = row['people'][0]
            result = row['predictions']
            for label, score, box, pose in zip(result['labels'], result['scores'], result['boxes'], result['keypoints']):
                detections.append(dict(image_id=image_ids[(row['family_id'], row['condition'])], category_id=label,
                    score=score, bbox=[box[0], box[1], box[2]-box[0], box[3]-box[1]], keypoints=pose))
        metrics = {}
        for style in ('main', 'nuisance', 'clear'):
            eligible = [family for family in grouped.values() if style+'_0' in family]
            if not eligible:
                continue
            values = [all(family[f'{style}_{member}']['wrist_correct'] for member in (0, 1)) for family in eligible]
            strict = [all(family[f'{style}_{member}']['wrist_correct'] and family[f'{style}_{member}']['candidate_count'] == 1 for member in (0, 1)) for family in eligible]
            people = [family[f'{style}_{member}'] for family in eligible for member in (0, 1)]
            metrics[style] = dict(images=len(people), detected=sum(person['detected'] for person in people),
                correct=sum(person['wrist_correct'] for person in people), paired_correct=interval(values),
                strict_single_candidate_pair_correct=interval(strict),
                duplicate_images=sum(person['candidate_count'] > 1 for person in people))
            if style == 'main':
                paired[arm] = np.asarray(values, dtype=float)
        detection_path = args.output.parent/f'{arm}_synthetic_detections.json'
        detection_path.write_text(json.dumps(detections)+'\n')
        metrics['synthetic_AP'] = evaluate_crowdpose_results(detection_path, annotation_path)
        summary[arm] = metrics
    summary['h_minus_control_native_pair_correct'] = interval(paired['h']-paired['control'])
    summary['AP_scope'] = 'Synthetic common12-joint rig only; head/neck unlabeled. Not CrowdPose validation/test AP.'
    args.output.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
