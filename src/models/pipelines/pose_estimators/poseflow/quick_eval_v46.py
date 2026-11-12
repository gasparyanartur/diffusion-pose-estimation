import json
import time
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from src.models.pipelines.pose_autoencoder.model import geometry_loss_per_person
from ..diffusionregpose.evaluation import crowdpose_result_records, postprocess_crowdpose
from .codec_v43 import geometry_from_batch
from .quick_eval_v13 import evaluate_records, evaluation_context, select_test_images
from .quick_eval_v21 import v21_test_batches
from .rows_v46 import build_training_rows, ordered_sources


def tensorboard_metrics(report):
    mode = 'ddim_eta0/' if report.get('evaluation_sampler') == 'ddim' else ''
    prefix = f'quick-test/ordered_sequence/{mode}rows_{report["rows"]}'
    result = {f'{prefix}/{name}': value for name, value in report['metrics'].items()}
    result.update({f'{prefix}/{name}': report[name] for name in
        ('count_mae', 'mean_eos_row', 'capacity_stop_fraction', 'inference_images_per_second', 'peak_cuda_allocated_bytes')})
    result.update({f'ordered_sequence/{name}': value for name, value in report['refinement'].items()})
    return result


def _rank_errors(collection, namespace, latents, ranks, sources, geometry, codec, image_wh):
    if not len(ranks):
        return
    target = sources['latents'][0, ranks]
    latent_errors = (latents-target).square().mean(-1)
    indices = sources['target_indices'][0, ranks]
    decoded = codec.decode_latents(latents, image_wh.expand(len(ranks), -1))
    geometry_errors = geometry_loss_per_person(decoded, {name: value[indices] for name, value in geometry.items()})['loss']
    for rank, latent, error in zip(ranks.tolist(), latent_errors.tolist(), geometry_errors.tolist()):
        for name, value in (('latent_mse', latent), ('geometry', error)):
            collection.setdefault(f'{namespace[0]}/rank_{rank+1:02d}/{namespace[1]}_{name}', []).append(value)


def _diagnostics(model, context, output, batch, collection, *, seed, collect_trajectory=False):
    geometry, locations = geometry_from_batch(batch)
    sources = ordered_sources(model.codec, geometry, locations, batch_size=1, capacity=model.capacity)
    people = int(sources['counts'][0])
    sizes = batch['img_hw'].flip(-1).to(context['memory'])
    rows = build_training_rows(sources['latents'], sources['counts'], rows=model.rows,
        generator=torch.Generator().manual_seed(seed+100000))
    teacher = model.forward_rows(context, **{name: rows[name] for name in
        ('noisy_latents', 'row_times', 'token_types', 'padding_mask')})
    ranks = torch.arange(people, device=sizes.device)
    _rank_errors(collection, ('teacher_forced', 'birth'), teacher['pred_latents'][-1, 0, ranks, ranks],
        ranks, sources, geometry, model.codec, sizes)
    _rank_errors(collection, ('teacher_forced', 'final'), teacher['pred_latents'][-1, 0, -1, :people],
        ranks, sources, geometry, model.codec, sizes)
    births = {int(row['prefix_counts'][0]): row['pred_latents'][0, int(row['prefix_counts'][0])]
        for row in output['trace'] if row['layer'] == 2 and row['birth'][0] and int(row['prefix_counts'][0]) < people}
    if births:
        birth_ranks = torch.tensor(sorted(births), device=sizes.device)
        _rank_errors(collection, ('generated', 'birth'), torch.stack([births[rank] for rank in sorted(births)]),
            birth_ranks, sources, geometry, model.codec, sizes)
    common = min(people, int(output['person_counts'][0]))
    _rank_errors(collection, ('generated', 'final'), output['pred_latents'][0, :common],
        ranks[:common], sources, geometry, model.codec, sizes)
    if collect_trajectory and people:
        trace = [row for row in output['trace'] if row['layer'] == 2 and row['person_counts'][0] > 0]
        if trace:
            latents = torch.stack([row['pred_latents'][0, 0] for row in trace])
            errors = (latents-sources['latents'][0, 0]).square().mean(-1)
            indices = sources['target_indices'][0, :1].expand(len(trace))
            decoded = model.codec.decode_latents(latents, sizes.expand(len(trace), -1))
            geometry_errors = geometry_loss_per_person(decoded,
                {name: value[indices] for name, value in geometry.items()})['loss']
            for row, error, geometry_error in zip(trace, errors.tolist(), geometry_errors.tolist()):
                for name, value in (('latent_mse', error), ('geometry', geometry_error)):
                    collection.setdefault(f'trajectory/row_{row["row"]:02d}/rank_01/{name}', []).append(value)
    return people


def evaluate_v46_batches(model, batches, directory, annotations, image_ids, *, seed=42,
                         metric_evaluator=evaluate_records, sampler='renoise', collect_trajectory=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records, seen, collection, counts, eos_rows, capacity_stops = [], [], {}, [], [], []
    started, inference_seconds = time.perf_counter(), 0.
    with evaluation_context(model):
        for batch in batches:
            if len(batch['image']) != 1:
                raise ValueError('V46 evaluation expects one image per batch')
            image_id = int(batch['image_id'][0])
            seen.append(image_id)
            sizes = batch['img_hw'].flip(-1).to(batch['image'])
            if sizes.is_cuda:
                torch.cuda.synchronize()
            inference_start = time.perf_counter()
            context = model.prepare_context(batch['image'], batch.get('padding_mask'))
            output = model.predict(context, sizes, collect_trace=True,
                generator=torch.Generator().manual_seed(seed+image_id), sampler=sampler)
            if sizes.is_cuda:
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter()-inference_start
            if not all(torch.isfinite(output[name]).all() for name in ('pred_latents', 'pred_boxes', 'pred_keypoints', 'pred_logits')):
                raise FloatingPointError('Nonfinite V46 evaluation output')
            valid = output['valid_mask'][0]
            if valid.any():
                selected = {name: output[name][:, valid] for name in ('pred_logits', 'pred_boxes', 'pred_keypoints')}
                processed = postprocess_crowdpose(selected, batch.get('original_img_hw', batch['img_hw']).to(sizes),
                    num_body_points=model.num_body_points, nms_target_sizes=batch['img_hw'].to(sizes))
                records.extend(crowdpose_result_records(image_id, processed))
            # GT becomes available only after image-only generation is complete.
            people = _diagnostics(model, context, output, batch, collection, seed=seed+image_id,
                                  collect_trajectory=collect_trajectory)
            counts.append(abs(int(output['person_counts'][0])-people))
            eos_rows.append(int(output['eos_row'][0]))
            capacity_stops.append(int(output['stop_reason'][0]) == 2)
            if len(seen) % 100 == 0:
                (directory/'progress.json').write_text(json.dumps(dict(images=len(seen), total=len(image_ids))))
    if seen != image_ids:
        raise ValueError('V46 VAL coverage/order differs from manifest')
    (directory/'predictions.json').write_text(json.dumps(records, allow_nan=False))
    actual_eos = [value for value in eos_rows if value]
    report = dict(images=len(seen), image_ids=seen, rows=model.rows, capacity=model.capacity,
        metrics=metric_evaluator(records, annotations, image_ids), detections=len(records),
        count_mae=sum(counts)/max(len(counts), 1), mean_eos_row=sum(actual_eos)/max(len(actual_eos), 1),
        capacity_stop_fraction=sum(capacity_stops)/max(len(capacity_stops), 1),
        inference_seconds=inference_seconds, inference_images_per_second=len(seen)/max(inference_seconds, 1e-9),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if next(model.parameters()).is_cuda else 0,
        seconds_including_diagnostics=time.perf_counter()-started,
        refinement={name: sum(values)/len(values) for name, values in collection.items() if not name.startswith('trajectory/')},
        refinement_counts={name: len(values) for name, values in collection.items() if not name.startswith('trajectory/')},
        split='val', annotations=str(annotations), seed=seed, sampler=model.sampler_description(sampler),
        evaluation_sampler=sampler,
        postprocessing='native CrowdPose pose NMS; birth person-probability confidence; invalid slots excluded',
        diagnostic_scope='Direct canonical-rank errors; teacher_forced uses GT prefixes, generated covers emitted ranks only; no Hungarian matching')
    if collect_trajectory:
        report['trajectory'] = {name: sum(values)/len(values) for name, values in collection.items() if name.startswith('trajectory/')}
        report['trajectory_counts'] = {name: len(values) for name, values in collection.items() if name.startswith('trajectory/')}
    (directory/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def run_v46_evaluation(model, data, root, *, epoch, count=32, seed=42, sampler='renoise', collect_trajectory=False):
    from src.data.datasets import PoseDataset
    root = Path(root)
    dataset = PoseDataset.from_path(data.val_ann_path, img_root=data.img_dir, num_keypoints=14)
    candidates = sorted(int(sample['image_id']) for sample in dataset.samples)
    ids = candidates if count is None else select_test_images(candidates, root/'subset.json',
        count=count, seed=seed, annotations=data.val_ann_path)
    evaluation_data = copy(data)
    evaluation_data.trainer = SimpleNamespace(training=False)
    report = evaluate_v46_batches(model, v21_test_batches(evaluation_data, dataset, ids, next(model.parameters()).device),
        root/f'epoch{epoch:03d}', data.val_ann_path, ids, seed=seed, sampler=sampler,
        collect_trajectory=collect_trajectory)
    report.update(epoch=epoch, full_validation=count is None, resize=[data.eval_resize_scale, data.eval_max_size])
    (root/f'epoch{epoch:03d}'/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
