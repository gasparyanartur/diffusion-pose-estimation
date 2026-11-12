import json
from pathlib import Path

import torch

from ..diffusionregpose.evaluation import crowdpose_result_records, postprocess_crowdpose
from .quick_eval_v13 import evaluate_records, evaluation_context, select_test_images
from .quick_eval_v21 import v21_test_batches


def evaluate_v43_grid(model, batches, directory, annotations, image_ids, *, steps=(1, 4, 8), seed=42,
                      metric_evaluator=evaluate_records, full_split=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    conditions = {f'noise_var_100/steps_{count}': dict(steps=count, records=[]) for count in steps}
    original_steps, seen, confidence_error = model.flow_sampling_steps, [], 0.
    try:
        with evaluation_context(model):
            for batch in batches:
                images = batch['image']
                if len(images) != 1:
                    raise ValueError('V43 evaluator requires one image per batch')
                image_id, previous_logits = int(batch['image_id'][0]), None
                seen.append(image_id)
                for row in conditions.values():
                    model.flow_sampling_steps = row['steps']
                    output = model.sample_keypoints(images, image_wh=batch['img_hw'].flip(-1).to(images),
                        padding_mask=batch.get('padding_mask'),
                        generator=torch.Generator(device=images.device).manual_seed(seed+image_id),
                        collect_step_diagnostics=True)
                    if not all(torch.isfinite(output[name]).all() for name in ('pred_logits', 'pred_boxes', 'pred_keypoints')):
                        raise FloatingPointError('Nonfinite V43 inference output')
                    if previous_logits is not None:
                        confidence_error = max(confidence_error, float((output['pred_logits']-previous_logits).abs().max()))
                    previous_logits = output['pred_logits']
                    prediction = postprocess_crowdpose(output,
                        batch.get('original_img_hw', batch['img_hw']).to(images),
                        num_body_points=model.num_body_points, nms_target_sizes=batch['img_hw'].to(images))
                    row['records'].extend(crowdpose_result_records(image_id, prediction))
                    if len(seen) <= 16:
                        torch.save({name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                            for name, value in output.items()}, directory/f'image{image_id}_k{row["steps"]}.pt')
                if len(seen) % 100 == 0:
                    (directory/'progress.json').write_text(json.dumps(dict(completed=len(seen), total=len(image_ids))))
    finally:
        model.flow_sampling_steps = original_steps
    if seen != image_ids:
        raise ValueError('V43 evaluator image coverage differs from manifest')
    if confidence_error:
        raise RuntimeError(f'Image-only confidence changed with diffusion steps: {confidence_error}')
    for row in conditions.values():
        records = row.pop('records')
        path = directory/f'k{row["steps"]}_predictions.json'
        path.write_text(json.dumps(records, allow_nan=False))
        row.update(predictions=str(path), metrics=metric_evaluator(records, annotations, image_ids), detections=len(records))
    report = dict(images=len(seen), image_ids=image_ids, conditions=conditions, seed=seed,
        per_image_seed='seed + image_id', sampler=model.sampler_kind,
        conditioning='standardized latent Gaussian; no GT geometry', annotations=str(annotations),
        raw_confidence_max_difference=confidence_error, precision='FP32',
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        postprocessing='native CrowdPose pose NMS; cached image-only confidence; final step only',
        scope='full test split' if full_split else f'fixed{len(seen)} test-monitored diagnostic; not full-test AP')
    (directory/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def run_v43_evaluation(model, data, root, *, epoch, global_step, count=32, steps=(1, 4, 8), seed=42):
    from src.data.datasets import PoseDataset

    root = Path(root)
    dataset = PoseDataset.from_path(data.test_ann_path, img_root=data.img_dir, num_keypoints=14)
    candidates = sorted(int(sample['image_id']) for sample in dataset.samples)
    image_ids = candidates if count is None else select_test_images(candidates, root/'subset.json',
        count=count, seed=seed, annotations=data.test_ann_path)
    directory = root/f'epoch{epoch:03d}'
    report = evaluate_v43_grid(model,
        v21_test_batches(data, dataset, image_ids, next(model.parameters()).device),
        directory, data.test_ann_path, image_ids, steps=steps, seed=seed, full_split=count is None)
    report.update(epoch=epoch, global_step=global_step, resize=[data.eval_resize_scale, data.eval_max_size],
                  model=type(model).__name__)
    (directory/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
