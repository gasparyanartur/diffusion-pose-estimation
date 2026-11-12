import json
import time
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from src.models.pipelines.pose_autoencoder.model import geometry_loss_per_person
from ..diffusionregpose.evaluation import crowdpose_result_records, postprocess_crowdpose
from .codec_v43 import geometry_from_batch
from .coordinates_v47 import coordinate_geometry, ordered_coordinates, training_rows
from .quick_eval_v13 import evaluate_records, evaluation_context, select_test_images
from .quick_eval_v21 import v21_test_batches


def tensorboard_metrics(report):
    prefix = f'quick-test/ordered_coordinate/rows_{report["rows"]}'
    result = {f'{prefix}/{name}': value for name, value in report['metrics'].items()}
    result.update({f'{prefix}/{name}': report[name] for name in
        ('count_mae', 'mean_eos_row', 'capacity_stop_fraction', 'inference_images_per_second', 'peak_cuda_allocated_bytes')})
    result.update({f'ordered_coordinate/{name}': value for name, value in report['refinement'].items()})
    return result


def _rank_errors(collection, namespace, coordinates, ranks, sources, geometry, image_wh):
    if not len(ranks):
        return
    mask = sources['coordinate_mask'][0, ranks]
    difference = torch.where(mask, coordinates-sources['coordinates'][0, ranks], 0.)
    coordinate_errors = difference.square().sum(-1)/mask.sum(-1).clamp_min(1)
    indices = sources['target_indices'][0, ranks]
    decoded = coordinate_geometry(coordinates, image_wh.expand(len(ranks), -1))
    geometry_errors = geometry_loss_per_person(decoded, {name: value[indices] for name, value in geometry.items()})['loss']
    for rank, mse, error in zip(ranks.tolist(), coordinate_errors.tolist(), geometry_errors.tolist()):
        for name, value in (('coordinate_mse', mse), ('geometry', error)):
            collection.setdefault(f'{namespace[0]}/rank_{rank+1:02d}/{namespace[1]}_{name}', []).append(value)


def _diagnostics(model, context, output, batch, collection, *, seed):
    geometry, locations = geometry_from_batch(batch)
    sources = ordered_coordinates(geometry, locations, batch_size=1, capacity=model.capacity)
    people = int(sources['counts'][0])
    sizes = batch['img_hw'].flip(-1).to(context['memory'])
    rows = training_rows(sources['coordinates'], sources['counts'], rows=model.rows, scale=model.scale,
        generator=torch.Generator().manual_seed(seed+100000))
    teacher = model.forward_rows(context, **{name: rows[name] for name in model.row_input_fields})
    ranks = torch.arange(people, device=sizes.device)
    _rank_errors(collection, ('teacher_forced', 'birth'), teacher['pred_coordinates'][-1, 0, ranks, ranks],
        ranks, sources, geometry, sizes)
    _rank_errors(collection, ('teacher_forced', 'final'), teacher['pred_coordinates'][-1, 0, -1, :people],
        ranks, sources, geometry, sizes)
    births = {int(row['prefix_counts'][0]): row['pred_coordinates'][0, int(row['prefix_counts'][0])]
        for row in output['trace'] if row['layer'] == 2 and row['birth'][0] and int(row['prefix_counts'][0]) < people}
    if births:
        birth_ranks = torch.tensor(sorted(births), device=sizes.device)
        _rank_errors(collection, ('generated', 'birth'), torch.stack([births[rank] for rank in sorted(births)]),
            birth_ranks, sources, geometry, sizes)
    common = min(people, int(output['person_counts'][0]))
    _rank_errors(collection, ('generated', 'final'), output['pred_coordinates'][0, :common],
        ranks[:common], sources, geometry, sizes)
    return people


def evaluate_v47_batches(model, batches, directory, annotations, image_ids, *, seed=42,
                         metric_evaluator=evaluate_records, split='val'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records, seen, collection, counts, eos_rows, capacity_stops, per_image = [], [], {}, [], [], [], []
    inference_seconds, peak_memory = 0., 0
    started = time.perf_counter()
    with evaluation_context(model):
        for batch in batches:
            if len(batch['image']) != 1:
                raise ValueError('V47 evaluation expects one image per batch')
            image_id = int(batch['image_id'][0])
            seen.append(image_id)
            sizes = batch['img_hw'].flip(-1).to(batch['image'])
            if sizes.is_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            inference_start = time.perf_counter()
            context = model.prepare_context(batch['image'], batch.get('padding_mask'))
            output = model.predict(context, sizes, collect_trace=True,
                generator=torch.Generator().manual_seed(seed+image_id))
            if sizes.is_cuda:
                torch.cuda.synchronize()
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
            inference_seconds += time.perf_counter()-inference_start
            if not all(torch.isfinite(output[name]).all() for name in ('pred_coordinates', 'pred_boxes', 'pred_keypoints', 'pred_logits')):
                raise FloatingPointError('Nonfinite V47 evaluation output')
            valid = output['valid_mask'][0]
            if valid.any():
                selected = {name: output[name][:, valid] for name in ('pred_logits', 'pred_boxes', 'pred_keypoints')}
                processed = postprocess_crowdpose(selected, batch.get('original_img_hw', batch['img_hw']).to(sizes),
                    num_body_points=model.num_body_points, nms_target_sizes=batch['img_hw'].to(sizes))
                records.extend(crowdpose_result_records(image_id, processed))
            # Image-only generation precedes all GT-conditioned diagnostics.
            people = _diagnostics(model, context, output, batch, collection, seed=seed+image_id)
            predicted_count = int(output['person_counts'][0])
            per_image.append(dict(image_id=image_id, people=people, predicted_people=predicted_count))
            counts.append(abs(predicted_count-people))
            eos_rows.append(int(output['eos_row'][0]))
            capacity_stops.append(int(output['stop_reason'][0]) == 2)
            if len(seen) % 100 == 0:
                (directory/'progress.json').write_text(json.dumps(dict(images=len(seen), total=len(image_ids))))
    if seen != image_ids:
        raise ValueError('V47 evaluation coverage/order differs from manifest')
    (directory/'predictions.json').write_text(json.dumps(records, allow_nan=False))
    actual_eos = [value for value in eos_rows if value]
    report = dict(images=len(seen), image_ids=seen, rows=model.rows, capacity=model.capacity,
        metrics=metric_evaluator(records, annotations, image_ids), detections=len(records),
        count_mae=sum(counts)/max(len(counts), 1), exact_count_images=sum(value == 0 for value in counts),
        per_image_counts=per_image, mean_eos_row=sum(actual_eos)/max(len(actual_eos), 1),
        capacity_stop_fraction=sum(capacity_stops)/max(len(capacity_stops), 1),
        inference_seconds=inference_seconds, inference_images_per_second=len(seen)/max(inference_seconds, 1e-9),
        peak_cuda_allocated_bytes=peak_memory, seconds_including_diagnostics=time.perf_counter()-started,
        refinement={name: sum(values)/len(values) for name, values in collection.items()},
        refinement_counts={name: len(values) for name, values in collection.items()},
        split=split, annotations=str(annotations), seed=seed, sampler='coordinate DDIM eta0; forward-noised newborns',
        postprocessing='native CrowdPose pose NMS; birth person-probability confidence; invalid slots excluded',
        diagnostic_scope='Direct canonical-rank errors; teacher_forced final uses clean GT input; generated covers emitted ranks only')
    (directory/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def run_v47_evaluation(model, data, root, *, epoch, count=32, seed=42):
    from src.data.datasets import PoseDataset
    root = Path(root)
    dataset = PoseDataset.from_path(data.val_ann_path, img_root=data.img_dir, num_keypoints=14)
    candidates = sorted(int(sample['image_id']) for sample in dataset.samples)
    ids = candidates if count is None else select_test_images(candidates, root/'subset.json',
        count=count, seed=seed, annotations=data.val_ann_path)
    evaluation_data = copy(data)
    evaluation_data.trainer = SimpleNamespace(training=False)
    report = evaluate_v47_batches(model, v21_test_batches(evaluation_data, dataset, ids, next(model.parameters()).device),
        root/f'epoch{epoch:03d}', data.val_ann_path, ids, seed=seed)
    report.update(epoch=epoch, full_validation=count is None, resize=[data.eval_resize_scale, data.eval_max_size])
    (root/f'epoch{epoch:03d}'/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
