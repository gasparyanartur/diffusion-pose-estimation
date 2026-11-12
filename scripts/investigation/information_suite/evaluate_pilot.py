from __future__ import annotations

import argparse
import gzip
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from scripts.evaluation.evaluate_poseflow import load_poseflow_model
from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import postprocess_crowdpose, resize_crowdpose_eval_image
from scoring import score_people


@contextmanager
def capture_features(model):
    capture, pending, handles = {}, {}, []
    def decoder_hook(module, args, output):
        for index, features in enumerate(output[0]):
            capture[f'geometry_{index}'] = features.detach().cpu().numpy().copy()
    handles.append(model.transformer.decoder.register_forward_hook(decoder_hook))
    if model.h_enabled:
        def before_branch(module, args, kwargs):
            if kwargs['decoder_context'].hypotheses is None:
                features = args[0].transpose(0, 1)
                pending['boxes'] = features.reshape(features.shape[0], -1, 15, features.shape[-1])[:, :, 0]
        def image_hook(module, args, output):
            capture['h_initial'] = (pending['boxes']+output[0]).detach().cpu().numpy().copy()
        def branch_hook(module, args, kwargs, output):
            layer = kwargs['layer_index']
            capture[f'h_written_{layer}'] = kwargs['decoder_context'].hypotheses.detach().cpu().numpy().copy()
        handles.append(model.h_branch.register_forward_pre_hook(before_branch, with_kwargs=True))
        handles.append(model.h_branch.image_attention.register_forward_hook(image_hook))
        handles.append(model.h_branch.register_forward_hook(branch_hook, with_kwargs=True))
    try:
        yield capture
    finally:
        for handle in handles:
            handle.remove()


def forward(model, image, family_id):
    torch.manual_seed(42+family_id)
    return model.sample_keypoints(image, generator=torch.Generator(device='cuda').manual_seed(42+family_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--reuse-feature-native', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.root/'pilot_manifest.json').read_text())
    assert manifest['complete'] and len(manifest['families']) > 0
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.monotonic()
    outputs = args.root/'native'; outputs.mkdir(exist_ok=True)
    native_forwards = 0
    parity_forwards = 0
    cached_forwards = 0
    for arm in ('control', 'h'):
        model, _ = load_poseflow_model(args.suite/'checkpoints'/arm/'epoch065.ckpt', args.suite/'checkpoints'/arm/'config.yaml')
        model.cuda().eval()
        assert model.h_enabled == (arm == 'h') and model.flow_sampling_steps == 1
        cached = {}
        if args.reuse_feature_native:
            with gzip.open(args.root/'features'/f'{arm}_native.jsonl.gz', 'rt') as stream:
                cached = {(row['family_id'], row['condition']):row for row in map(json.loads, stream)}
        with torch.inference_mode(), (outputs/f'{arm}.jsonl').open('w') as stream:
            for number, family in enumerate(manifest['families']):
                for condition, entry in family['conditions'].items():
                    existing = cached.get((family['family_id'], condition))
                    if existing is not None:
                        stream.write(json.dumps(dict(existing, arm=arm, target_person=family['target_person']))+'\n')
                        cached_forwards += 1
                        continue
                    rgb = np.asarray(Image.open(entry['path']).convert('RGB')).copy()
                    image = resize_crowdpose_eval_image(torch.from_numpy(rgb).permute(2, 0, 1).float()/255, shortest_side=256, max_size=427)[None].cuda()
                    output = forward(model, image, family['family_id'])
                    native_forwards += 1
                    if number == 0 and condition == next(iter(family['conditions'])):
                        with capture_features(model) as capture:
                            captured_output = forward(model, image, family['family_id'])
                        repeated = forward(model, image, family['family_id'])
                        parity_forwards += 2
                        parity = {key: float((output[key]-captured_output[key]).abs().max()) for key in ('pred_logits', 'pred_boxes', 'pred_keypoints')}
                        repeat_error = max(float((output[key]-repeated[key]).abs().max()) for key in parity)
                        assert max(parity.values()) <= 1e-6 and repeat_error <= 1e-6
                        np.savez_compressed(outputs/f'{arm}_capture.npz', **capture)
                        (outputs/f'{arm}_capture_audit.json').write_text(json.dumps({'hook_errors': parity, 'repeat_error': repeat_error,
                            'shapes': {key:list(value.shape) for key,value in capture.items()},
                            'source_module': model.__class__.__module__}, indent=2)+'\n')
                    assert all(torch.isfinite(output[key]).all() for key in ('pred_logits','pred_boxes','pred_keypoints'))
                    sizes = torch.tensor([[256,384]], device='cuda', dtype=torch.float32)
                    result = {key: value.detach().cpu().numpy().tolist() for key, value in postprocess_crowdpose(output,sizes,nms_target_sizes=sizes)[0].items()}
                    people = score_people(result, entry['keypoints'], entry['boxes'])
                    row = {'arm': arm, 'family_id': family['family_id'], 'condition': condition,
                           'target_person': family['target_person'], 'people': people, 'predictions': result}
                    if condition == 'neutral':
                        row['alternative_people'] = score_people(result,entry['alternative_keypoints'],entry['boxes'])
                        target = family['target_person']
                        assert not (people[target]['wrist_correct'] and row['alternative_people'][target]['wrist_correct'])
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                progress={'stage':'native_evaluation','arm':arm,'completed_families':number+1,'total_families':len(manifest['families']),'seconds':time.monotonic()-started}
                (args.root/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                print(json.dumps(progress),flush=True)
        del model
        torch.cuda.empty_cache()
    (outputs/'DONE.json').write_text(json.dumps({'models':2,'families':len(manifest['families']),
        'phase':manifest.get('phase','native_pilot'),'native_forwards':native_forwards,
        'reused_feature_forwards':cached_forwards,
        'parity_forwards':parity_forwards,'seconds':time.monotonic()-started},indent=2)+'\n')


if __name__ == '__main__':
    main()
