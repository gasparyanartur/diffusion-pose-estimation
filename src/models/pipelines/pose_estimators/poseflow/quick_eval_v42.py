import json
from pathlib import Path

import torch

from ..diffusionregpose.evaluation import crowdpose_result_records, postprocess_crowdpose
from .quick_eval_v13 import evaluate_records, evaluation_context, select_test_images
from .quick_eval_v21 import v21_test_batches


def evaluate_v42_grid(model, batches, config, directory, annotations, image_ids,
                      *, metric_evaluator=evaluate_records, sampler=None, sampler_description=None):
    if sampler is not None and not sampler_description:
        raise ValueError('A custom V42 sampler requires an explicit policy description')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    conditions = {f'noise_var_100/steps_{steps}': {'steps': steps, 'records': []}
                  for steps in config.quick_eval_steps}
    seen = []
    original_steps = model.sampling_timesteps
    try:
        with evaluation_context(model):
            for batch in batches:
                image = batch['image']
                if image.shape[0] != 1:
                    raise ValueError('V42 quick evaluation requires one image per batch')
                image_id = int(batch['image_id'][0])
                seen.append(image_id)
                for row in conditions.values():
                    model.sampling_timesteps = row['steps']
                    arguments = dict(padding_mask=batch.get('padding_mask'),
                        generator=torch.Generator(device=image.device).manual_seed(config.quick_eval_seed + image_id))
                    output = (model.sample_keypoints(image, **arguments) if sampler is None
                              else sampler(model, image, **arguments))
                    if not all(torch.isfinite(output[name]).all() for name in ('pred_logits', 'pred_boxes', 'pred_keypoints')):
                        raise ValueError('Nonfinite V42 quick-test output')
                    results = postprocess_crowdpose(output,
                        batch.get('original_img_hw', batch['img_hw']).to(image),
                        num_body_points=model.transformer.decoder.num_body_points,
                        nms_target_sizes=batch['img_hw'].to(image))
                    row['records'].extend(crowdpose_result_records(image_id, results))
    finally:
        model.sampling_timesteps = original_steps
    if seen != image_ids:
        raise ValueError('V42 quick-test coverage differs from the fixed manifest')
    for name, row in conditions.items():
        records = row.pop('records')
        path = directory / (name.replace('/', '_') + '_predictions.json')
        path.write_text(json.dumps(records, allow_nan=False))
        row.update(predictions=str(path), metrics=metric_evaluator(records, annotations, image_ids))
    return dict(image_ids=image_ids, images=len(image_ids), seed=config.quick_eval_seed,
        per_image_seed='seed + image_id', annotations=str(annotations),
        sampler=sampler_description or 'native coordinate cosine DDIM eta=0; whole-pose feature denoiser',
        conditioning='Gaussian pose coordinates, no GT sampling input',
        scope='fixed held-out subset diagnostic, not full-test AP', conditions=conditions)


def run_v42_evaluation(model, data, config, root, *, epoch, global_step, count=32,
                       sampler=None, sampler_description=None):
    from src.data.datasets import PoseDataset

    root = Path(root)
    dataset = PoseDataset.from_path(data.test_ann_path, img_root=data.img_dir, num_keypoints=data.num_keypoints)
    candidates = sorted(int(sample['image_id']) for sample in dataset.samples)
    image_ids = candidates if count is None else select_test_images(candidates, root / 'subset.json',
        count=config.quick_eval_images, seed=config.quick_eval_seed, annotations=data.test_ann_path)
    directory = root / f'epoch{epoch:03d}'
    device = next(model.parameters()).device
    report = evaluate_v42_grid(model, v21_test_batches(data, dataset, image_ids, device),
        config, directory, data.test_ann_path, image_ids,
        sampler=sampler, sampler_description=sampler_description)
    report.update(epoch=epoch, global_step=global_step, model=type(model).__name__,
        resize=[data.eval_resize_scale, data.eval_max_size],
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32)
    if getattr(model, 'confidence_mode', None) == 'image_only':
        report['confidence_mode'] = 'image_only'
        report['confidence'] = getattr(model, 'confidence_description',
            'selected image/time features before noisy-pose injection')
    if count is None:
        report['scope'] = 'full held-out test split'
    (directory / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
