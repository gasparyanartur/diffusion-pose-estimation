import json
import time
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import torch
from scipy.optimize import linear_sum_assignment

from ..diffusionregpose.training import DiffusionRegPoseMatcher
from .quick_eval_v13 import evaluate_records, evaluation_context, select_test_images
from .quick_eval_v21 import v21_test_batches
from .quick_eval_v50 import _output_records, _rank_errors
from .rows_v53 import matched_training_rows


def tensorboard_metrics(report):
    values = {f'quick-test/{name}': value for name, value in report['metrics'].items()}
    values.update({f'quick-test/{name}': report[name] for name in ('count_mae', 'mean_eos_row',
        'mean_generation_steps', 'capacity_stop_fraction', 'inference_images_per_second', 'peak_cuda_allocated_bytes')})
    values.update({f'diagnostics/{name}': value for name, value in report['refinement'].items()})
    return values


def birth_diagnostics(model, context, output, batch, collection, seed):
    matcher = DiffusionRegPoseMatcher().to(context['memory'].device)
    rows, sources, geometry, targets = matched_training_rows(model, matcher, batch, context,
        generator=torch.Generator().manual_seed(seed+100000))
    people = int(sources['counts'][0])
    sizes = batch['img_hw'].flip(-1).to(context['memory'])
    teacher = model.forward_rows(context, **{name: rows[name] for name in model.row_input_fields})
    ranks = torch.arange(people, device=sizes.device)
    _rank_errors(collection, ('teacher_forced', 'birth'), teacher['pred_coordinates'][-1, 0, ranks], ranks, sources, geometry, sizes)
    generated = int(output['person_counts'][0])
    if generated and people:
        coordinates = output['pred_coordinates'][0, :generated]
        difference = torch.where(sources['coordinate_mask'][0, None, :people],
            coordinates[:, None]-sources['coordinates'][0, None, :people], 0.)
        costs = difference.square().sum(-1)/sources['coordinate_mask'][0, None, :people].sum(-1).clamp_min(1)
        predicted, target = linear_sum_assignment(costs.cpu().numpy())
        predicted = torch.as_tensor(predicted, device=sizes.device)
        target = torch.as_tensor(target, device=sizes.device)
        aligned = dict(sources)
        for name in ('coordinates', 'coordinate_mask', 'target_indices'):
            aligned[name] = sources[name].clone()
            aligned[name][0, predicted] = sources[name][0, target]
        _rank_errors(collection, ('generated_final_assignment', 'birth'), coordinates[predicted],
                     predicted, aligned, geometry, sizes)
    return people


def evaluate_v53_batches(model, batches, directory, annotations, image_ids, *, seed=42,
                         metric_evaluator=evaluate_records, split='val'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records, seen, collection, per_image = [], [], {}, []
    errors, eos_rows, steps, capacities = [], [], [], []
    inference_seconds, peak_memory = 0., 0
    started = time.perf_counter()
    with evaluation_context(model):
        for batch in batches:
            if len(batch['image']) != 1:
                raise ValueError('V53 evaluation expects one image per batch')
            image_id = int(batch['image_id'][0])
            seen.append(image_id)
            sizes = batch['img_hw'].flip(-1).to(batch['image'])
            if sizes.is_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            inference_start = time.perf_counter()
            context = model.prepare_context(batch['image'], batch.get('padding_mask'))
            output = model.predict(context, sizes)
            if sizes.is_cuda:
                torch.cuda.synchronize()
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
            inference_seconds += time.perf_counter()-inference_start
            if not all(torch.isfinite(output[name]).all() for name in ('pred_coordinates', 'pred_boxes', 'pred_keypoints', 'pred_logits')):
                raise FloatingPointError('Nonfinite V53 evaluation geometry')
            records.extend(_output_records(model, output, batch, sizes, image_id))
            people = birth_diagnostics(model, context, output, batch, collection, seed+image_id)
            count = int(output['person_counts'][0])
            per_image.append(dict(image_id=image_id, people=people, predicted_people=count))
            errors.append(abs(count-people))
            eos_rows.append(int(output['eos_row'][0]))
            steps.append(int(output['generation_steps'][0]))
            capacities.append(int(output['stop_reason'][0]) == 2)
            if len(seen) % 100 == 0:
                (directory/'progress.json').write_text(json.dumps(dict(images=len(seen), total=len(image_ids))))
    if seen != image_ids:
        raise ValueError('V53 evaluation image coverage differs from manifest')
    actual_eos = [value for value in eos_rows if value]
    report = dict(images=len(seen), image_ids=seen, rows=model.rows, capacity=model.capacity,
        metrics=metric_evaluator(records, annotations, image_ids), detections=len(records),
        count_mae=sum(errors)/max(len(errors), 1), exact_count_images=sum(value == 0 for value in errors),
        per_image_counts=per_image, mean_eos_row=sum(actual_eos)/max(len(actual_eos), 1),
        mean_generation_steps=sum(steps)/max(len(steps), 1), capacity_stop_fraction=sum(capacities)/max(len(capacities), 1),
        inference_seconds=inference_seconds, inference_images_per_second=len(seen)/max(inference_seconds, 1e-9),
        peak_cuda_allocated_bytes=peak_memory, seconds_including_diagnostics=time.perf_counter()-started,
        refinement={name: sum(values)/len(values) for name, values in collection.items()},
        refinement_counts={name: len(values) for name, values in collection.items()},
        split=split, annotations=str(annotations), seed=seed, architecture=getattr(model, 'architecture', 'V53'),
        sampler='deterministic_birth_only;fixed_clean_prefix;immediate_EOS_or_capacity;no_noise_no_refinement',
        postprocessing='native CrowdPose pose NMS; native objectness times birth nonEOS confidence',
        diagnostic_scope='Teacher-forced random clean-prefix birth; generated fixed Hungarian birth assignment; no refinement metrics')
    (directory/'predictions.json').write_text(json.dumps(records, allow_nan=False))
    (directory/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def run_v53_evaluation(model, data, root, *, epoch, count=32, seed=42, split='test'):
    from src.data.datasets import PoseDataset
    if split not in ('test', 'val'):
        raise ValueError('Evaluation split must be test or val')
    annotations = Path(getattr(data, f'{split}_ann_path'))
    if split == 'test' and annotations.resolve() == Path(data.val_ann_path).resolve():
        raise ValueError('TEST annotations must not alias VAL')
    root = Path(root)
    dataset = PoseDataset.from_path(annotations, img_root=data.img_dir, num_keypoints=14)
    candidates = sorted(int(sample['image_id']) for sample in dataset.samples)
    ids = candidates if count is None else select_test_images(candidates, root/'subset.json', count=count,
                                                             seed=seed, annotations=annotations)
    evaluation_data = copy(data)
    evaluation_data.trainer = SimpleNamespace(training=False)
    report = evaluate_v53_batches(model, v21_test_batches(evaluation_data, dataset, ids, next(model.parameters()).device),
                                  root/f'epoch{epoch:03d}', annotations, ids, seed=seed, split=split)
    report.update(epoch=epoch, full_test=split == 'test' and count is None,
                  full_validation=split == 'val' and count is None,
                  resize=[data.eval_resize_scale, data.eval_max_size])
    (root/f'epoch{epoch:03d}'/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
