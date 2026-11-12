import json
from pathlib import Path

import torch

from ..diffusionregpose.training import build_diffusionregpose_clean_targets
from .quick_eval_v13 import evaluation_context
from .quick_eval_v21 import v21_test_batches
from .training_v42_paired_proposal import build_proposal_paired_pose_inputs


def reconstruction_statistics(predicted, target, areas, sigmas):
    points = len(sigmas)
    errors = (predicted - target).reshape(-1, points, 2).norm(dim=-1)
    oks = torch.exp(-errors.square() / (areas[:, None] * (2 * sigmas.to(errors)).square()[None] * 2)).mean(-1)
    return dict(coordinate_mae=float((predicted - target).abs().mean()), paired_oks=float(oks.mean()),
                paired_oks_p05=float(torch.quantile(oks, .05)), joint_error_p95=float(torch.quantile(errors.flatten(), .95)))


@torch.no_grad()
def evaluate_task_pose_diagnostics(pipeline, data, image_ids, directory, *, max_images=4):
    from src.data.datasets import PoseDataset

    model, teacher = pipeline.model, pipeline.teacher
    device = next(model.parameters()).device
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset = PoseDataset.from_path(data.test_ann_path, img_root=data.img_dir, num_keypoints=data.num_keypoints)
    selected_ids = image_ids[:max_images]
    codes, targets_xy, areas, teacher_codes = [], [], [], []
    state_values = {'correct': [], 'zero': [], 'swapped': []}
    points = model.transformer.decoder.num_body_points
    with evaluation_context(model):
        for batch in v21_test_batches(data, dataset, selected_ids, device):
            targets = build_diffusionregpose_clean_targets(batch, num_body_points=points)
            context = model.prepare_context(batch['image'], batch.get('padding_mask'))
            times = torch.full((1,), 100, dtype=torch.long, device=device)
            correct, matches = build_proposal_paired_pose_inputs(context['pred_logits'], context['boxes'], times, targets,
                diffusion=model, matcher=pipeline.criterion.matcher, noise_seed=42)
            queries, people = matches[0]
            complete = targets[0]['keypoints'][people, 2 * points:].bool().all(-1)
            if not complete.any():
                continue
            rows = targets[0]['keypoints'][people][complete]
            xy, area = rows[:, :2 * points], targets[0]['area'][people][complete]
            codes.append(model.pose_codec.encode(xy))
            teacher_codes.append(teacher(xy))
            targets_xy.append(xy)
            areas.append(area)
            for name, state in (('correct', correct), ('zero', torch.zeros_like(correct)), ('swapped', correct.roll(1, 1))):
                prediction = model.denoise_coordinates(state, times, context)
                state_values[name].append(prediction['pred_keypoints'][0, queries[complete], :2 * points])
        report = dict(image_ids=selected_ids, complete_people=sum(map(len, codes)),
            scope='GT-conditioned complete-person diagnostics at t100; not native detector AP', perturbations={})
        if codes:
            online, clean, area, target = torch.cat(codes), torch.cat(targets_xy), torch.cat(areas), torch.cat(teacher_codes)
            sigmas = pipeline.criterion.matcher.sigmas
            report['online_clean'] = reconstruction_statistics(model.pose_codec.decode(online), clean, area, sigmas)
            report['teacher_clean'] = reconstruction_statistics(model.pose_codec.decode(target), clean, area, sigmas)
            report['online_variance'] = float(online.var(0, unbiased=False).mean())
            report['teacher_variance'] = float(target.var(0, unbiased=False).mean())
            report['teacher_lag'] = float((online - target).square().mean())
            generator = torch.Generator(device=device).manual_seed(42)
            direction = torch.randn(online.shape, device=device, generator=generator)
            for level in (0., .01, .025, .05):
                perturbed = online if level == 0 else model.pose_codec.norm(online + level * direction)
                report['perturbations'][str(level)] = dict(
                    latent_mse=float((perturbed - online).square().mean()),
                    **reconstruction_statistics(model.pose_codec.decode(perturbed), clean, area, sigmas))
            report['state_use'] = {name: reconstruction_statistics(torch.cat(values), clean, area, sigmas)
                                   for name, values in state_values.items()}
    (directory / 'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
