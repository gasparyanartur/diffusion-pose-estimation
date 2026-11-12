from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from scripts.evaluation.evaluate_poseflow import load_poseflow_model
from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import postprocess_crowdpose, resize_crowdpose_eval_image
from evaluate_pilot import forward
from feature_capture import capture_maps, select_features
from scoring import score_people


INITIALIZATION = Path('/storage/outputs/pose-graph-diff/results/poseflow_v24_matched/20261022_192822/initialization/seed42_shared.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--arm', choices=['initialization', 'control', 'h'], required=True)
    parser.add_argument('--limit-families', type=int)
    args = parser.parse_args()
    manifest = json.loads((args.root/'pilot_manifest.json').read_text())
    assert manifest['complete']
    families = manifest['families'][:args.limit_families]
    checkpoint_arm = 'control' if args.arm == 'initialization' else args.arm
    model, _ = load_poseflow_model(args.suite/'checkpoints'/checkpoint_arm/'epoch065.ckpt', args.suite/'checkpoints'/checkpoint_arm/'config.yaml')
    if args.arm == 'initialization':
        state = torch.load(INITIALIZATION, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model.cuda().eval()
    destination = args.root/'features'; destination.mkdir(exist_ok=True)
    prefix = args.arm + ('_smoke' if args.limit_families else '')
    arrays, metadata = {}, []
    native_calls, parity_calls = 0, 0
    start = time.monotonic()
    with torch.inference_mode(), gzip.open(destination/f'{prefix}_native.jsonl.gz', 'wt') as stream:
        for index, family in enumerate(families):
            target_joint = 2 if manifest['study'] == 'B' else 4
            shared_point = family['conditions']['main_0']['keypoints'][0][target_joint]
            for condition in ('main_0', 'main_1', 'nuisance_0', 'nuisance_1'):
                entry = family['conditions'][condition]
                rgb = np.asarray(Image.open(entry['path']).convert('RGB')).copy()
                image = resize_crowdpose_eval_image(torch.from_numpy(rgb).permute(2, 0, 1).float()/255, shortest_side=256, max_size=427)[None].cuda()
                with capture_maps(model, shared_point) as (maps, decoder):
                    output = forward(model, image, family['family_id'])
                native_calls += 1
                if index == 0 and condition == 'main_0':
                    reference = forward(model, image, family['family_id'])
                    parity_calls += 1
                    parity = {key:float((output[key]-reference[key]).abs().max()) for key in ('pred_logits', 'pred_boxes', 'pred_keypoints')}
                    assert max(parity.values()) <= 1e-6
                    (destination/f'{prefix}_parity.json').write_text(json.dumps(parity, indent=2)+'\n')
                selected, correspondence = select_features(maps, decoder, output, entry['boxes'][0], target_joint)
                assert all(value.flags['OWNDATA'] for value in selected.values())
                for key, value in selected.items():
                    arrays.setdefault(key, []).append(value)
                metadata.append(dict(family_id=family['family_id'], condition=condition,
                                     label=int(condition[-1]), stratum=family['stratum'], **correspondence))
                sizes = image.new_tensor([[256, 384]])
                predictions = {key:value.detach().cpu().numpy().tolist() for key,value in postprocess_crowdpose(output, sizes, nms_target_sizes=sizes)[0].items()}
                people = score_people(predictions, entry['keypoints'], entry['boxes'])
                stream.write(json.dumps(dict(family_id=family['family_id'], condition=condition, predictions=predictions, people=people))+'\n')
            if (index+1) % 32 == 0:
                progress = dict(arm=args.arm, families=index+1, total=len(families), seconds=time.monotonic()-start)
                (destination/f'{prefix}_progress.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress), flush=True)
    np.savez_compressed(destination/f'{prefix}.npz', **{key:np.stack(values) for key,values in arrays.items()})
    (destination/f'{prefix}_rows.json').write_text(json.dumps(metadata)+'\n')
    (destination/f'{prefix}_DONE.json').write_text(json.dumps(dict(families=len(families), native_calls=native_calls, parity_calls=parity_calls, seconds=time.monotonic()-start), indent=2)+'\n')
    print('FEATURES_COMPLETE', args.arm, len(metadata), flush=True)


if __name__ == '__main__':
    main()
