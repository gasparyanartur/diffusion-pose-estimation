import json
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from src.models.pipelines.pose_autoencoder.evaluation import export_reconstructions, paired_metrics
from ..diffusionregpose.evaluation import crowdpose_result_records, postprocess_crowdpose
from ..diffusionregpose.training import DiffusionRegPoseMatcher
from .latent_v43 import assign_latent_sources, geometry_targets
from .quick_eval_v13 import evaluate_records, evaluation_context, select_test_images
from .quick_eval_v21 import v21_test_batches


def _collect_small_noise(model, batch, context, sources, geometry, collections, *, seed, levels):
    positive = sources['positive']
    if not positive.any():
        return
    rows = sources['target_indices'][positive]
    target = {name: value[rows].cpu() for name, value in geometry.items()}
    sizes = batch['img_hw'].flip(-1).to(batch['image'])
    source = sources['latents']
    generator = torch.Generator(device=source.device).manual_seed(seed + int(batch['image_id'][0]))
    noise = torch.randn(source.shape, device=source.device, generator=generator)
    schedule_sigma = model.sqrt_recipm1_alphas_cumprod.to(source)
    for fraction in levels:
        time = int((schedule_sigma - fraction).abs().argmin())
        times = torch.full((len(source),), time, dtype=torch.long, device=source.device)
        noisy = model.q_sample(source, times, noise)
        output = model.denoise(noisy, times, context, sizes)
        input_geometry = model.codec.decode_latents(noisy, sizes[:, None].expand(-1, model.num_proposals, -1))
        row = collections.setdefault(f'noise_{fraction*100:g}', dict(target=[], input=[], output=[], input_error=[], output_error=[], time=time,
            actual_sigma=float(schedule_sigma[time])))
        row['target'].append(target)
        for label, prediction in (('input', input_geometry), ('output', output)):
            row[label].append({name: prediction[name][positive].cpu() for name in ('keypoints_px', 'boxes_cxcywh_px')})
        row['input_error'].append((noisy - source)[positive].square().mean(-1).cpu())
        row['output_error'].append((output['pred_latents'] - source)[positive].square().mean(-1).cpu())


def _summarize_small_noise(collections):
    metrics = {}
    for label, row in collections.items():
        target = {name: torch.cat([item[name] for item in row['target']]) for name in row['target'][0]}
        metrics[f'{label}/timestep'] = row['time']
        metrics[f'{label}/actual_sigma'] = row['actual_sigma']
        for kind in ('input', 'output'):
            prediction = {name: torch.cat([item[name] for item in row[kind]]) for name in row[kind][0]}
            metrics[f'{label}/{kind}_latent_mse'] = float(torch.cat(row[f'{kind}_error']).mean())
            metrics.update({f'{label}/{kind}_{name}': value for name, value in paired_metrics(prediction, target).items()})
    return metrics


def evaluate_v44_batches(model, batches, directory, annotations, image_ids, *, steps=(1, 4, 8), seed=42,
                         levels=(.01, .025, .05, .1), metric_evaluator=evaluate_records):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    conditions = {f'noise_var_100/steps_{count}': dict(steps=count, records=[]) for count in steps}
    collections, seen, confidence_error = {}, [], 0.
    matcher = DiffusionRegPoseMatcher(num_body_points=model.num_body_points)
    with evaluation_context(model):
        for batch in batches:
            images = batch['image']
            if len(images) != 1:
                raise ValueError('V44 evaluator expects one image per batch')
            image_id = int(batch['image_id'][0])
            seen.append(image_id)
            sizes = batch['img_hw'].flip(-1).to(images)
            context = model.prepare_context(images, batch.get('padding_mask'))
            geometry, targets = geometry_targets(batch)
            if any(len(target['labels']) > model.num_proposals for target in targets):
                raise ValueError('Evaluation exceeds proposal capacity')
            matches = matcher(dict(pred_boxes=context['boxes'], pred_logits=context['pred_logits']), targets, include_keypoints=False)
            sources = assign_latent_sources(model.codec, geometry, targets, matches,
                slots=model.num_proposals, generator=torch.Generator(), posterior_mode='mean')
            previous = None
            for row in conditions.values():
                output = model.sample(context, sizes, steps=row['steps'],
                    generator=torch.Generator(device=images.device).manual_seed(seed + image_id))
                if not all(torch.isfinite(output[name]).all() for name in ('pred_latents', 'pred_boxes', 'pred_keypoints', 'pred_logits')):
                    raise FloatingPointError('Nonfinite V44 evaluation output')
                if previous is not None:
                    confidence_error = max(confidence_error, float((output['pred_logits'] - previous).abs().max()))
                previous = output['pred_logits']
                processed = postprocess_crowdpose(output, batch.get('original_img_hw', batch['img_hw']).to(images),
                    num_body_points=model.num_body_points, nms_target_sizes=batch['img_hw'].to(images))
                row['records'].extend(crowdpose_result_records(image_id, processed))
                if len(seen) <= 4 and sources['positive'].any():
                    positive = sources['positive']
                    indices = sources['target_indices'][positive]
                    target = {name: value[indices].cpu() for name, value in geometry.items()}
                    prediction = {name: output[name][positive].cpu() for name in ('keypoints_px', 'boxes_cxcywh_px')}
                    target['annotation_id'] = torch.arange(len(indices))
                    export_reconstructions(directory / f'image{image_id}_k{row["steps"]}.png', prediction, target,
                        title='V44 latent diffusion: prediction (blue), fixed-assignment target (gray)')
            _collect_small_noise(model, batch, context, sources, geometry, collections, seed=seed, levels=levels)
            if len(seen) % 100 == 0:
                (directory / 'progress.json').write_text(json.dumps(dict(images=len(seen), total=len(image_ids))))
    if seen != image_ids:
        raise ValueError('V44 VAL coverage/order differs from manifest')
    if confidence_error:
        raise RuntimeError('Fixed proposal confidence changed across reverse schedules')
    for row in conditions.values():
        records = row.pop('records')
        path = directory / f'k{row["steps"]}_predictions.json'
        path.write_text(json.dumps(records, allow_nan=False))
        row.update(predictions=str(path), metrics=metric_evaluator(records, annotations, image_ids), detections=len(records))
    report = dict(images=len(seen), image_ids=seen, conditions=conditions, small_noise=_summarize_small_noise(collections),
        prediction_type=model.prediction_type,
        split='val', annotations=str(annotations), seed=seed, raw_confidence_max_difference=confidence_error,
        sampler=model.sampler_kind, postprocessing='native CrowdPose pose NMS; final output; image-only confidence',
        small_noise_scope='GT-derived forward-corruption reconstruction, not detector AP; nearest discrete sigma',
        detection_scope='Gaussian-start image-conditioned detector AP; no GT input to sampler')
    (directory / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def run_v44_evaluation(model, data, root, *, epoch, count=32, steps=(1, 4, 8), seed=42,
                       levels=(.01, .025, .05, .1)):
    from src.data.datasets import PoseDataset
    root = Path(root)
    dataset = PoseDataset.from_path(data.val_ann_path, img_root=data.img_dir, num_keypoints=14)
    candidates = sorted(int(sample['image_id']) for sample in dataset.samples)
    ids = candidates if count is None else select_test_images(candidates, root / 'subset.json',
        count=count, seed=seed, annotations=data.val_ann_path)
    evaluation_data = copy(data)
    evaluation_data.trainer = SimpleNamespace(training=False)
    report = evaluate_v44_batches(model, v21_test_batches(evaluation_data, dataset, ids, next(model.parameters()).device),
        root / f'epoch{epoch:03d}', data.val_ann_path, ids, steps=steps, seed=seed, levels=levels)
    report.update(epoch=epoch, full_validation=count is None, resize=[data.eval_resize_scale, data.eval_max_size])
    (root / f'epoch{epoch:03d}' / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
