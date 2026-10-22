import json
from pathlib import Path

import torch

from ..diffusionregpose.evaluation import crowdpose_result_records, postprocess_crowdpose
from .quick_eval_v13 import evaluate_records, evaluation_context, select_test_images


def evaluate_v21_quick_grid(model, batches, config, directory, annotations, image_ids,
                           *, metric_evaluator=evaluate_records):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    conditions = {
        f'noise_var_100/steps_{steps}/memory_{mode}': {
            'steps': steps, 'hypothesis_mode': mode, 'records': [],
        }
        for steps in config.quick_eval_steps for mode in ('persistent', 'reset')
    }
    seen = []
    previous_steps = model.flow_sampling_steps
    try:
        with evaluation_context(model):
            for batch in batches:
                image = batch['image']
                if image.shape[0] != 1:
                    raise ValueError('V21 quick evaluation requires one image per batch')
                image_id = int(batch['image_id'][0])
                seen.append(image_id)
                original_sizes = batch.get('original_img_hw', batch['img_hw']).to(image)
                for condition in conditions.values():
                    model.flow_sampling_steps = condition['steps']
                    generator = torch.Generator(device=image.device).manual_seed(config.quick_eval_seed+image_id)
                    output = model.sample_keypoints(image, padding_mask=batch.get('padding_mask'),
                        generator=generator, persist_hypotheses=condition['hypothesis_mode'] == 'persistent')
                    if not all(torch.isfinite(output[key]).all()
                               for key in ('pred_logits', 'pred_boxes', 'pred_keypoints')):
                        raise ValueError('Nonfinite V21 quick-test prediction')
                    results = postprocess_crowdpose(output, original_sizes,
                                                   num_body_points=model.transformer.decoder.num_body_points,
                                                   nms_target_sizes=batch['img_hw'].to(image))
                    condition['records'].extend(crowdpose_result_records(image_id, results))
    finally:
        model.flow_sampling_steps = previous_steps
    if seen != image_ids:
        raise ValueError('V21 quick-test coverage differs from its fixed image manifest')
    for name, condition in conditions.items():
        records = condition.pop('records')
        path = directory/(name.replace('/', '_')+'_predictions.json')
        path.write_text(json.dumps(records, allow_nan=False))
        condition['predictions'] = str(path)
        condition['metrics'] = metric_evaluator(records, annotations, image_ids)
    return {
        'image_ids': image_ids, 'images': len(image_ids), 'seed': config.quick_eval_seed,
        'per_image_seed': 'seed + image_id', 'annotations': str(annotations),
        'sampler': 'native V21 fixed-source endpoint Euler',
        'conditioning': 'Gaussian source; no GT-derived sampling state',
        'scope': 'fixed held-out subset diagnostic, not full-test AP',
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'conditions': conditions,
    }


def v21_test_batches(data, dataset, image_ids, device):
    from src.data.datasets import SetPoseDataset, set_pose_collate

    indices = {int(sample['image_id']): index for index, sample in enumerate(dataset.samples)}
    wrapped = SetPoseDataset(dataset)
    for image_id in image_ids:
        batch = set_pose_collate([wrapped[indices[image_id]]])
        batch = {
            key: ([image.to(device) for image in value] if key == 'image'
                  else value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        yield data.on_after_batch_transfer(batch, 0)


def run_v21_quick_test(model, data, config, root, *, epoch, global_step, source=None):
    from src.data.datasets import PoseDataset

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = PoseDataset.from_path(data.test_ann_path, img_root=data.img_dir,
                                    num_keypoints=data.num_keypoints)
    candidates = sorted(int(sample['image_id']) for sample in dataset.samples)
    image_ids = select_test_images(candidates, root/'subset.json',
        count=config.quick_eval_images, seed=config.quick_eval_seed, annotations=data.test_ann_path)
    directory = root/f'epoch{epoch:03d}'
    device = next(model.parameters()).device
    report = evaluate_v21_quick_grid(model, v21_test_batches(data, dataset, image_ids, device),
                                     config, directory, data.test_ann_path, image_ids)
    report.update(epoch=epoch, global_step=global_step, model=type(model).__name__,
                  hypothesis_update_order=model.hypothesis_update_order,
                  time_embedding_scale=getattr(model, 'time_embedding_scale', None),
                  coordinate_batch_layout=getattr(model, 'coordinate_batch_layout', None),
                  training_protocol=getattr(config, 'training_protocol', 'single_call'),
                  resize=[data.eval_resize_scale, data.eval_max_size], source=source or {'kind': 'live epoch end'})
    (directory/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
