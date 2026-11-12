from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from scripts.investigation.diffusionregpose_inspection.analysis import (
    evaluate_records, postprocess_with_queries,
)
from scripts.investigation.diffusionregpose_inspection.run import (
    Settings, assert_same_predictions, experiment_manifest, load_model, prepare_image,
)
from scripts.investigation.diffusionregpose_inspection.trace import cpu_copy
from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import crowdpose_result_records


def replace_joint_rows(value, *, stride, axis, replacement=0.):
    rows = torch.arange(value.shape[axis], device=value.device) % stride != 0
    indices = [slice(None)] * value.ndim
    indices[axis] = rows
    changed = value.clone()
    changed[tuple(indices)] = replacement
    return changed


@contextmanager
def mask_decoder_routes(model, *, first_image=False, final_image=False, first_updates=False):
    """Mask joint residual updates while preserving box rows and projection bias."""
    if model.training:
        raise ValueError('route masks require model.eval()')
    decoder = model.transformer.decoder
    first = decoder.num_box_decoder_layers
    stride = decoder.num_body_points + 1
    handles = []

    def image_bias_only(module, args, output):
        bias = module.output_proj.bias
        return replace_joint_rows(output, stride=stride, axis=1,
                                  replacement=0. if bias is None else bias)

    def zero_attention(module, args, output):
        return (replace_joint_rows(output[0], stride=stride, axis=0), *output[1:])

    def zero_feedforward(module, args, output):
        return replace_joint_rows(output, stride=stride, axis=0)

    try:
        if first_image:
            handles.append(decoder.layers[first].cross_attn.register_forward_hook(image_bias_only))
        if final_image:
            handles.append(decoder.layers[-1].cross_attn.register_forward_hook(image_bias_only))
        if first_updates:
            handles.append(decoder.layers[first].self_attn.register_forward_hook(zero_attention))
            handles.append(decoder.layers[first].linear2.register_forward_hook(zero_feedforward))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def observe_calls(model):
    """Keep small state/reference observations without copying encoder memory."""
    calls, handles = [], []
    decoder = model.transformer.decoder
    first = decoder.num_box_decoder_layers
    stride = decoder.num_body_points + 1

    def begin(module, args, kwargs):
        if args[0].shape[0] != 1:
            raise ValueError('observations require one image per call')
        calls.append({'time': int(args[2][0]), 'state': cpu_copy(args[1]), 'references': {}})

    def finish(module, args, kwargs, output):
        calls[-1]['prediction'] = cpu_copy({key: output[key] for key in
                                           ('pred_logits', 'pred_boxes', 'pred_keypoints')})
        calls[-1]['first_pose_prediction'] = cpu_copy(output['aux_outputs'][first]['pred_keypoints'])

    def references(index):
        def before(module, args, kwargs):
            calls[-1]['references'][index] = cpu_copy(kwargs['raw_reference_points'][:, 0])
        return before

    def selected(module, args, kwargs, output):
        scores = decoder.class_embed[first - 1](output).max(-1).values
        calls[-1]['selected_boxes'] = cpu_copy(scores.topk(decoder.num_group, dim=0).indices[:, 0])

    def first_image_values(module, args, output):
        joints = torch.arange(output.shape[1], device=output.device) % stride != 0
        bias = module.output_proj.bias
        contribution = output[:, joints] - (0. if bias is None else bias)
        calls[-1]['first_image_value_abs_max'] = float(contribution.abs().max())

    try:
        handles.append(model.register_forward_pre_hook(begin, with_kwargs=True))
        handles.append(model.register_forward_hook(finish, with_kwargs=True))
        handles.append(decoder.layers[first - 1].register_forward_hook(selected, with_kwargs=True))
        for index in (first, len(decoder.layers) - 1):
            handles.append(decoder.layers[index].register_forward_pre_hook(references(index), with_kwargs=True))
        handles.append(decoder.layers[first].cross_attn.register_forward_hook(first_image_values))
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def mean_shift(left, right, pixel_scale):
    return float(((left - right) * pixel_scale).norm(dim=-1).mean())


def compare_calls(baseline, changed, original_hw, scale, first, stride):
    torch.testing.assert_close(baseline['selected_boxes'], changed['selected_boxes'], rtol=0, atol=0)
    pixel_scale = torch.tensor(original_hw[::-1], dtype=torch.float32)
    left = baseline['prediction']['pred_keypoints'][0, :, :28].reshape(-1, 14, 2)
    right = changed['prediction']['pred_keypoints'][0, :, :28].reshape(-1, 14, 2)
    top = baseline['prediction']['pred_logits'][0].sigmoid().amax(-1).topk(10).indices
    joint_rows = torch.arange(baseline['references'][first].shape[0]) % stride != 0
    incoming = baseline['references'][first][joint_rows, :2]
    incoming_changed = changed['references'][first][joint_rows, :2]
    return {
        'state_mean_shift_px': mean_shift(baseline['state'].reshape(-1, 2),
                                         changed['state'].reshape(-1, 2), pixel_scale / (2 * scale)),
        'incoming_reference_mean_shift_px': mean_shift(incoming, incoming_changed, pixel_scale),
        'first_pose_mean_shift_px': mean_shift(baseline['first_pose_prediction'][..., :28].reshape(-1, 2),
                                              changed['first_pose_prediction'][..., :28].reshape(-1, 2), pixel_scale),
        'final_mean_joint_shift_px': mean_shift(left, right, pixel_scale),
        'top10_final_mean_joint_shift_px': mean_shift(left[top], right[top], pixel_scale),
        'final_max_coordinate_delta': float((left - right).abs().max()),
        'logit_abs_max_delta': float((baseline['prediction']['pred_logits'] -
                                      changed['prediction']['pred_logits']).abs().max()),
    }


def append_records(records, key, sample, call, resized_hw, device):
    prediction = {name: tensor.to(device) for name, tensor in call['prediction'].items()}
    result = postprocess_with_queries(prediction, sample['img_hw'], resized_hw)
    records.setdefault(key, []).extend(crowdpose_result_records(sample['image_id'], [result]))
    return result


def aggregate_rows(rows):
    keys = [key for key in rows[0] if key.endswith('_px') or key.endswith('_delta')]
    return {key: {'mean': float(np.mean([row[key] for row in rows])),
                  'median': float(np.median([row[key] for row in rows])),
                  'max': float(np.max([row[key] for row in rows]))} for key in keys}


def summarize_reference_geometry(observations, times, first, stride):
    summary = {}
    for time in times:
        references, clamped_changes = [], []
        for key, calls in observations.items():
            if len(key) != 2:
                continue
            call = next(item for item in calls if item['time'] == time)
            points = call['references'][first].reshape(-1, stride, 4)[:, 1:, :2]
            references.append(points)
            if key[1] > 0:
                baseline = next(item for item in observations[(key[0], 0)] if item['time'] == time)
                base_points = baseline['references'][first].reshape(-1, stride, 4)[:, 1:, :2]
                clamped_changes.append(float((points.clamp(0, 1) - base_points.clamp(0, 1)).abs().max()))
        points = torch.stack(references)
        summary[str(time)] = {
            'xy_min': points.amin((0, 1, 2)).tolist(),
            'xy_max': points.amax((0, 1, 2)).tolist(),
            'joint_reference_count': points.numel() // 2,
            'in_image_center_count': int(((points >= 0) & (points <= 1)).all(-1).sum()),
            'max_clamped_reference_change_across_seeds': max(clamped_changes),
        }
    return summary


def run_state_sensitivity(cache_path, output, seed_count=5):
    if seed_count < 2:
        raise ValueError('at least two seeds are needed')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    settings = Settings(**cache['manifest']['settings'])
    model, epoch = load_model(settings)
    decoder = model.transformer.decoder
    if len(decoder.layers) != 3 or decoder.num_box_decoder_layers != 1:
        raise ValueError('this experiment is scoped to the reduced three-layer checkpoint')
    first, stride = decoder.num_box_decoder_layers, decoder.num_body_points + 1
    model.sampling_timesteps = 3
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    current_manifest = experiment_manifest(settings)
    for key in ('checkpoint_sha256', 'config_sha256', 'annotations_sha256'):
        if current_manifest[key] != cache['manifest'][key]:
            raise ValueError(f'cached experiment mismatch: {key}')
    manifest = {'settings': asdict(settings), 'epoch': epoch, 'cache': str(cache_path),
                'image_ids': cache['image_ids'], 'seed_count': seed_count,
                'seed_rule': '1729 + image_id + 100000 * seed_index',
                'schedule': list(model.ddim_timestep_pairs()), **current_manifest,
                'intervention_contract': 'single-call replay at baseline image, timestep and state; no feedback'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    masks = {'first_joint_image_bias_only': {'first_image': True},
             'final_joint_image_bias_only': {'final_image': True},
             'first_joint_feature_updates_off': {'first_updates': True},
             'first_updates_off_and_final_image_bias_only': {'first_updates': True, 'final_image': True}}
    records, comparisons, observations, ablations = {}, [], {}, []
    with torch.inference_mode():
        for index, sample in enumerate(cache['samples']):
            image_id = sample['image_id']
            image = prepare_image(sample, settings)
            baseline = None
            for seed in range(seed_count):
                generator = torch.Generator(device=settings.device).manual_seed(
                    settings.proposal_seed + image_id + 100000 * seed)
                with observe_calls(model) as calls:
                    prediction = model.sample_keypoints(image, generator=generator)
                if seed == 0:
                    assert_same_predictions(cache['sweep'][(image_id, 3)]['prediction'], prediction)
                    baseline = calls
                else:
                    for base, changed in zip(baseline, calls):
                        comparisons.append({'image_id': image_id, 'time': base['time'], 'seed': seed,
                                            **compare_calls(base, changed, sample['img_hw'], model.scale, first, stride)})
                observations[(image_id, seed)] = calls
                for call in calls:
                    processed = append_records(records, f'seed{seed}_t{call["time"]}', sample, call,
                                               image.shape[-2:], settings.device)
                    if seed == 0 and call['time'] == calls[-1]['time']:
                        cached_result = cache['sweep'][(image_id, 3)]['result']
                        for name in ('scores', 'boxes', 'keypoints', 'query_ids'):
                            torch.testing.assert_close(processed[name].cpu(), cached_result[name], rtol=0, atol=0)
            for condition, options in masks.items():
                for base in baseline:
                    with mask_decoder_routes(model, **options), observe_calls(model) as changed_calls:
                        model(image, base['state'].to(settings.device),
                              torch.tensor([base['time']], device=settings.device))
                    changed = changed_calls[0]
                    row = {'image_id': image_id, 'time': base['time'], 'condition': condition,
                           **compare_calls(base, changed, sample['img_hw'], model.scale, first, stride)}
                    ablations.append(row)
                    observations[(image_id, condition, base['time'])] = changed
                    append_records(records, f'{condition}_t{base["time"]}', sample, changed,
                                   image.shape[-2:], settings.device)
            print(f'{index + 1}/{len(cache["samples"])} image={image_id}: seed parity and fixed-time replays passed', flush=True)
            (output / 'progress.json').write_text(json.dumps({'completed_images': index + 1,
                                                            'total_images': len(cache['samples'])}))
    metrics = {}
    for key, predictions in records.items():
        aggregate, _ = evaluate_records(settings.annotations, predictions, cache['image_ids'], per_image=False)
        metrics[key] = aggregate
        (output / f'predictions_{key}.json').write_text(json.dumps(predictions))
        print(f'{key}: AP={aggregate["AP"]:.6f}', flush=True)
    times = [pair[0] for pair in model.ddim_timestep_pairs()]
    sensitivity = {str(time): aggregate_rows([row for row in comparisons if row['time'] == time]) for time in times}
    intervention = {condition: {str(time): aggregate_rows([row for row in ablations if
                                                         row['condition'] == condition and row['time'] == time])
                                for time in times} for condition in masks}
    contributions = [call['first_image_value_abs_max'] for key, calls in observations.items()
                     if len(key) == 2 for call in calls]
    result = {'manifest': manifest, 'metrics': metrics, 'sensitivity': sensitivity,
              'interventions': intervention, 'state_comparisons': comparisons,
              'intervention_comparisons': ablations,
              'reference_geometry': summarize_reference_geometry(observations, times, first, stride),
              'verification': {'seed_zero_final_cache_exact': True,
                               'seed_zero_final_postprocessing_cache_exact': True,
                               'selected_box_slots_identical_at_fixed_time': True,
                               'first_joint_image_contribution_abs_max': max(contributions),
                               'forward_calls': len(cache['samples']) * 3 * (seed_count + len(masks)),
                               'hooks_removed': all(not module._forward_hooks and not module._forward_pre_hooks
                                                    for module in model.modules())}}
    torch.save(observations, output / 'observations.pt')
    (output / 'results.json').write_text(json.dumps(result, indent=2))
    print(f'Completed: {output / "results.json"}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default='tmp/diffusionregpose_diffusion_inspection/inspection.pt')
    parser.add_argument('--output', default='tmp/diffusionregpose_state_sensitivity_verified')
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()
    run_state_sensitivity(args.cache, args.output, args.seeds)
