import contextlib
import io
import json
from pathlib import Path

import torch

from ..diffusionregpose.evaluation import (
    crowdpose_result_records, evaluate_crowdpose_results, postprocess_crowdpose,
)
from .quick_eval_v13 import evaluation_context, select_test_images
from .quick_eval_v21 import v21_test_batches


@torch.no_grad()
def coverage_diagnostics(output, batch):
    from src.eval.metrics import CROWDPOSE_SIGMAS
    from src.eval.poseflow_diagnostics import stage_pose_oks
    from src.eval.poseflow_training_diagnostics import _image_targets

    target = _image_targets(batch, 0)
    oks = stage_pose_oks(output, target, output['pred_boxes'].new_tensor(CROWDPOSE_SIGMAS))
    accepted = output['existence_logits'][0].sigmoid() >= .5
    hits = (oks >= .5) & accepted[:, None]
    covered = hits.any(0).sum().item()
    if oks.shape[1]:
        best_score, best_person = oks.max(1)
        good = accepted & (best_score >= .5)
        distinct = best_person[good].unique().numel()
        duplicates = int(good.sum())-distinct
    else:
        duplicates = 0
    return dict(gt_people=oks.shape[1], covered_people=covered,
                duplicate_predictions=duplicates, accepted_predictions=int(accepted.sum()))


def run_v26_evaluation(model, data, root, *, epoch, global_step, count=32, seed=42):
    from src.data.datasets import PoseDataset

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = PoseDataset.from_path(data.test_ann_path, img_root=data.img_dir,
                                    num_keypoints=data.num_keypoints)
    candidates = sorted(int(sample['image_id']) for sample in dataset.samples)
    image_ids = (candidates if count is None else select_test_images(candidates, root/'subset.json',
        count=count, seed=seed, annotations=data.test_ann_path))
    directory = root/f'epoch{epoch:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    conditions = {f'noise_var_100/steps_{steps}': dict(steps=steps, records=[],
        diagnostics=dict(gt_people=0, covered_people=0, duplicate_predictions=0, accepted_predictions=0)) for steps in (1, 4)}
    previous_steps = model.flow_sampling_steps
    device = next(model.parameters()).device
    seen = []
    try:
        with evaluation_context(model):
            for batch in v21_test_batches(data, dataset, image_ids, device):
                image_id = int(batch['image_id'][0])
                seen.append(image_id)
                for condition in conditions.values():
                    model.flow_sampling_steps = condition['steps']
                    generator = torch.Generator(device=device).manual_seed(seed+image_id)
                    output = model.sample_keypoints(batch['image'], padding_mask=batch.get('padding_mask'),
                                                    generator=generator)
                    if not all(torch.isfinite(output[key]).all() for key in ('pred_logits', 'pred_boxes', 'pred_keypoints')):
                        raise ValueError('V26 evaluation produced nonfinite predictions')
                    sizes = batch.get('original_img_hw', batch['img_hw']).to(batch['image'])
                    results = postprocess_crowdpose(output, sizes, num_body_points=model.num_body_points,
                                                    nms_target_sizes=batch['img_hw'].to(batch['image']))
                    condition['records'].extend(crowdpose_result_records(image_id, results))
                    for name, value in coverage_diagnostics(output, batch).items():
                        condition['diagnostics'][name] += value
                if len(seen) % 100 == 0 or len(seen) == len(image_ids):
                    progress = dict(completed=len(seen), total=len(image_ids), last_image_id=image_id)
                    (directory/'progress.json').write_text(json.dumps(progress)+'\n')
                    print('V26 EVALUATION', json.dumps(progress), flush=True)
    finally:
        model.flow_sampling_steps = previous_steps
    if seen != image_ids:
        raise ValueError('V26 evaluation image coverage differs from manifest')
    for condition in conditions.values():
        records = condition.pop('records')
        path = directory/f"k{condition['steps']}_predictions.json"
        path.write_text(json.dumps(records, allow_nan=False))
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = (evaluate_crowdpose_results(path, data.test_ann_path, image_ids=image_ids)
                if records else dict(AP=0., AP_E=0., AP_M=0., AP_H=0., **{'bbox/AP':0., 'bbox/AP50':0., 'bbox/AP75':0., 'bbox/AR':0.}))
        condition.update(predictions=str(path), metrics=metrics, detections=len(records))
        summary = condition['diagnostics']
        summary['coverage_oks50_score50'] = summary['covered_people']/max(summary['gt_people'], 1)
    report = dict(epoch=epoch, global_step=global_step, image_ids=image_ids, images=len(image_ids),
        annotations=str(data.test_ann_path), seed=seed, per_image_seed='seed + image_id',
        resize=[data.eval_resize_scale, data.eval_max_size], sampler=model.sampler_kind,
        scope='full held-out test' if count is None else f'fixed{len(image_ids)} held-out diagnostic',
        conditioning='Gaussian latents; no GT-derived sampling state',
        precision='FP32', matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32, conditions=conditions)
    (directory/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
