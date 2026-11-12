import json
from pathlib import Path

import torch

from src.models.pipelines.pose_autoencoder.evaluation import export_reconstructions, paired_metrics
from ..diffusionregpose.evaluation import crowdpose_result_records, postprocess_crowdpose
from .latent_v43 import assign_latent_sources, geometry_targets
from .quick_eval_v13 import evaluate_records, evaluation_context, select_test_images
from .quick_eval_v21 import v21_test_batches


def evaluate_alignment_batches(model, batches, directory, annotations, image_ids, *, matcher,
                               metric_evaluator=evaluate_records):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    conditions = {name: dict(records=[], geometry=[], errors=[])
                  for name in ('correct_image', 'wrong_image')}
    targets, codec_predictions, seen, pairs = [], [], [], []

    def prepare(batch):
        if len(batch['image']) != 1:
            raise ValueError('Alignment evaluation requires one image per batch')
        context = model.encode_context(batch['image'], batch.get('padding_mask'))
        references = model.select_references(model.propose(context))
        return batch, context['image'], references

    def collect(primary, donor):
        batch, context, references = primary
        donor_batch, donor_context, donor_references = donor
        image_id = int(batch['image_id'][0])
        donor_id = int(donor_batch['image_id'][0])
        if image_id == donor_id:
            raise ValueError('Conditioning donor must be a different image')
        seen.append(image_id)
        pairs.append(dict(image_id=image_id, donor_image_id=donor_id))
        geometry, match_targets = geometry_targets(batch)
        matches = matcher(dict(pred_boxes=references['boxes'], pred_logits=references['pred_logits']),
            match_targets, include_keypoints=False)
        sources = assign_latent_sources(model.codec, geometry, match_targets, matches,
            slots=model.num_proposals, generator=torch.Generator(), posterior_mode='mean')
        positive = sources['positive']
        rows = sources['target_indices'][positive]
        sizes = batch['img_hw'].flip(-1).to(batch['image'])
        target = {name: value[rows].cpu() for name, value in geometry.items()}
        targets.append(target)
        clean = model.codec.decode_latents(sources['latents'], sizes[:, None].expand(-1, model.num_proposals, -1))
        codec_predictions.append({name: clean[name][positive].cpu()
            for name in ('keypoints_px', 'boxes_cxcywh_px')})
        for label, row in conditions.items():
            selected = dict(references)
            if label == 'wrong_image':
                selected['features'] = donor_references['features']
            image = donor_context if label == 'wrong_image' else context
            output = model.predict_image_context(image, references=selected, image_wh=sizes)
            if not all(torch.isfinite(output[name]).all() for name in ('pred_latents', 'pred_boxes', 'pred_keypoints')):
                raise FloatingPointError('Nonfinite image-alignment prediction')
            assert selected['boxes'] is references['boxes']
            assert selected['existence_logits'] is references['existence_logits']
            row['geometry'].append({name: output[name][positive].cpu()
                for name in ('keypoints_px', 'boxes_cxcywh_px')})
            row['errors'].append((output['pred_latents']-sources['latents'])[positive].square().mean(-1).cpu())
            processed = postprocess_crowdpose(output, batch.get('original_img_hw', batch['img_hw']).to(sizes),
                num_body_points=model.num_body_points, nms_target_sizes=batch['img_hw'].to(sizes))
            row['records'].extend(crowdpose_result_records(image_id, processed))
        if len(seen) % 100 == 0:
            (directory/'progress.json').write_text(json.dumps(dict(completed=len(seen), total=len(image_ids))))

    with evaluation_context(model):
        first, pending = None, None
        for batch in batches:
            current = prepare(batch)
            if first is None:
                first = current
            if pending is None:
                pending = current
            else:
                collect(pending, current)
                collect(current, pending)
                pending = None
        if pending is not None:
            collect(pending, first)
    if seen != image_ids:
        raise ValueError('Alignment image coverage/order differs from manifest')
    target = {name: torch.cat([row[name] for row in targets]) for name in targets[0]}
    if not len(target['joint_mask']):
        raise ValueError('Alignment evaluation requires labeled people')
    clean = {name: torch.cat([row[name] for row in codec_predictions]) for name in codec_predictions[0]}
    for label, row in conditions.items():
        geometry_rows = row.pop('geometry')
        prediction = {name: torch.cat([value[name] for value in geometry_rows]) for name in clean}
        errors = torch.cat(row.pop('errors'))
        records = row.pop('records')
        path = directory/f'{label}_predictions.json'
        path.write_text(json.dumps(records, allow_nan=False)+'\n')
        row.update(predictions=str(path), metrics=metric_evaluator(records, annotations, image_ids),
            paired=dict(paired_metrics(prediction, target), foreground_latent_mse=float(errors.mean())))
        if label == 'correct_image':
            example_target = dict(target, annotation_id=torch.arange(len(target['joint_mask'])))
            export_reconstructions(directory/'geometry_examples.png', prediction, example_target,
                title='Image-only prediction (blue), target (gray); matched instance indices, labeled joints only')
    report = dict(image_ids=seen, images=len(seen), paired_people=len(target['joint_mask']),
        conditioning_pairs=pairs, conditions=conditions, clean_codec=paired_metrics(clean, target),
        annotations=str(annotations), reference_boxes_scores_unchanged=True,
        conditioning='Image/proposal features only; constant internal latent/time; no geometry or noise input',
        postprocessing='Native CrowdPose pose NMS; original image-only reference confidence for both conditions',
        scope='Validation alignment diagnostic; wrong-image condition is counterfactual, not detector performance',
        precision='FP32', matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32)
    (directory/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def run_alignment_evaluation(model, data, root, *, epoch, count=64, matcher):
    from src.data.datasets import PoseDataset

    root = Path(root)
    dataset = PoseDataset.from_path(data.val_ann_path, img_root=data.img_dir, num_keypoints=14)
    candidates = sorted(int(sample['image_id']) for sample in dataset.samples)
    image_ids = candidates if count is None else select_test_images(candidates, root/'subset.json',
        count=count, seed=42, annotations=data.val_ann_path)
    report = evaluate_alignment_batches(model,
        v21_test_batches(data, dataset, image_ids, next(model.parameters()).device),
        root/f'epoch{epoch:03d}', data.val_ann_path, image_ids, matcher=matcher)
    report.update(epoch=epoch, full_validation=count is None,
        resize=[data.eval_resize_scale, data.eval_max_size])
    (root/f'epoch{epoch:03d}'/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
