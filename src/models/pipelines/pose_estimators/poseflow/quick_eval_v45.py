import json
import time
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from src.models.pipelines.pose_autoencoder.model import geometry_loss_per_person
from ..diffusionregpose.evaluation import crowdpose_result_records, postprocess_crowdpose
from ..diffusionregpose.training import DiffusionRegPoseMatcher
from .latent_v43 import assign_latent_sources, geometry_targets
from .quick_eval_v13 import evaluate_records, evaluation_context, select_test_images
from .quick_eval_v21 import v21_test_batches


def tensorboard_metrics(report):
    prefix = f'quick-test/progressive/stages_{report["stages"]}'
    metrics = {f'{prefix}/{name}': value for name, value in report['metrics'].items()}
    metrics.update({f'quick-test/progressive/{name}': value for name, value in report['refinement'].items()})
    metrics['quick-test/progressive/images_per_second'] = report['images_per_second']
    return metrics


def refinement_errors(output, sources, geometry, image_wh, codec):
    positive = sources['positive']
    errors = {}
    birth = torch.stack([row['pred_latents'][:, row['stage']-1] for row in output['trace'] if row['layer'] == 2], 1)
    for label, latents in (('birth', birth), ('final', output['pred_latents'])):
        mse = (latents-sources['latents']).square().mean(-1)
        if not positive.any():
            continue
        rows = sources['target_indices'][positive]
        decoded = codec.decode_latents(latents[positive], image_wh[:, None].expand(-1, latents.shape[1], -1)[positive])
        geometry_error = geometry_loss_per_person(decoded, {name: value[rows] for name, value in geometry.items()})['loss']
        ranks = positive.nonzero()[:, 1]
        for rank, latent_value, geometry_value in zip(ranks.tolist(), mse[positive].tolist(), geometry_error.tolist()):
            errors[f'rank_{rank+1:02d}/{label}_latent_mse'] = latent_value
            errors[f'rank_{rank+1:02d}/{label}_geometry'] = geometry_value
    return errors


def evaluate_v45_batches(model, batches, directory, annotations, image_ids, *, seed=42,
                         metric_evaluator=evaluate_records):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records, seen, refinements = [], [], {}
    matcher = DiffusionRegPoseMatcher(num_body_points=model.num_body_points)
    started = time.perf_counter()
    with evaluation_context(model):
        for batch in batches:
            if len(batch['image']) != 1:
                raise ValueError('V45 evaluator expects one image per batch')
            image_id = int(batch['image_id'][0])
            seen.append(image_id)
            sizes = batch['img_hw'].flip(-1).to(batch['image'])
            context = model.prepare_context(batch['image'], batch.get('padding_mask'))
            output = model.predict(context, sizes, collect_trace=True)
            if not all(torch.isfinite(output[name]).all() for name in ('pred_latents', 'pred_boxes', 'pred_keypoints', 'pred_logits')):
                raise FloatingPointError('Nonfinite V45 evaluation output')
            processed = postprocess_crowdpose(output, batch.get('original_img_hw', batch['img_hw']).to(sizes),
                num_body_points=model.num_body_points, nms_target_sizes=batch['img_hw'].to(sizes))
            records.extend(crowdpose_result_records(image_id, processed))
            # GT is consumed only after all image-only predictions have been computed.
            geometry, targets = geometry_targets(batch)
            matches = matcher(dict(pred_boxes=context['boxes'], pred_logits=context['pred_logits']), targets, include_keypoints=False)
            sources = assign_latent_sources(model.codec, geometry, targets, matches,
                slots=model.num_proposals, generator=torch.Generator(), posterior_mode='mean')
            for name, value in refinement_errors(output, sources, geometry, sizes, model.codec).items():
                refinements.setdefault(name, []).append(value)
            if len(seen) % 100 == 0:
                (directory/'progress.json').write_text(json.dumps(dict(images=len(seen), total=len(image_ids))))
    elapsed = time.perf_counter()-started
    if seen != image_ids:
        raise ValueError('V45 VAL coverage/order differs from manifest')
    path = directory/'predictions.json'
    path.write_text(json.dumps(records, allow_nan=False))
    report = dict(images=len(seen), image_ids=seen, stages=model.num_proposals,
        metrics=metric_evaluator(records, annotations, image_ids), detections=len(records),
        refinement={name: sum(values)/len(values) for name, values in refinements.items()},
        refinement_counts={name: len(values) for name, values in refinements.items()},
        seconds=elapsed, images_per_second=len(seen)/max(elapsed, 1e-9),
        split='val', annotations=str(annotations), seed=seed, predictions=str(path),
        contract=model.sampler_kind, detection_scope='Image-only final-prefix native AP; GT only for metrics and fixed-assignment diagnostics',
        postprocessing='native CrowdPose pose NMS; image-only confidence')
    (directory/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def run_v45_evaluation(model, data, root, *, epoch, count=32, seed=42):
    from src.data.datasets import PoseDataset
    root = Path(root)
    dataset = PoseDataset.from_path(data.val_ann_path, img_root=data.img_dir, num_keypoints=14)
    candidates = sorted(int(sample['image_id']) for sample in dataset.samples)
    ids = candidates if count is None else select_test_images(candidates, root/'subset.json',
        count=count, seed=seed, annotations=data.val_ann_path)
    evaluation_data = copy(data)
    evaluation_data.trainer = SimpleNamespace(training=False)
    report = evaluate_v45_batches(model, v21_test_batches(evaluation_data, dataset, ids, next(model.parameters()).device),
        root/f'epoch{epoch:03d}', data.val_ann_path, ids, seed=seed)
    report.update(epoch=epoch, full_validation=count is None, resize=[data.eval_resize_scale, data.eval_max_size])
    (root/f'epoch{epoch:03d}'/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
