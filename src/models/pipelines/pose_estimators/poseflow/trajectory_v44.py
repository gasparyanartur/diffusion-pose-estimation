import json
from pathlib import Path

import torch

from src.data.datasets import PoseDataset
from src.models.pipelines.pose_autoencoder.evaluation import paired_metrics
from .latent_v43 import assign_latent_sources, geometry_targets
from .quick_eval_v13 import evaluation_context
from .quick_eval_v21 import v21_test_batches


def evaluate_trajectory_gap(module, data, directory, image_ids, *, seed=1042):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset = PoseDataset.from_path(data.val_ann_path, img_root=data.img_dir, num_keypoints=14)
    model = module.model
    predictions = {'forward': [], 'generated': []}
    errors = {'forward': [], 'generated': []}
    targets = []
    with evaluation_context(model):
        for batch in v21_test_batches(data, dataset, image_ids, next(model.parameters()).device):
            geometry, match_targets = geometry_targets(batch)
            context = model.prepare_context(batch['image'], batch.get('padding_mask'))
            matches = module.slot_matcher(dict(pred_boxes=context['boxes'], pred_logits=context['pred_logits']),
                match_targets, include_keypoints=False)
            sources = assign_latent_sources(model.codec, geometry, match_targets, matches,
                slots=model.num_proposals, generator=torch.Generator(), posterior_mode='mean')
            positive, clean = sources['positive'], sources['latents']
            rows = sources['target_indices'][positive]
            targets.append({name: value[rows].cpu() for name, value in geometry.items()})
            sizes = batch['img_hw'].flip(-1).to(clean)
            generator = torch.Generator(device=clean.device).manual_seed(seed + int(batch['image_id'][0]))
            noise = torch.randn(clean.shape, generator=generator, device=clean.device)
            state = noise.clone()
            for time, next_time in ((999, 749), (749, 499), (499, 249)):
                times = torch.full((len(clean),), time, device=clean.device, dtype=torch.long)
                output = model.denoise(state, times, context, sizes)
                epsilon = model.predict_noise_from_start(state, times, output['pred_latents'])
                state = model.ddim_step(output['pred_latents'], epsilon, time, next_time, noise=torch.zeros_like(state))
            times = torch.full((len(clean),), 249, device=clean.device, dtype=torch.long)
            forward = model.q_sample(clean, times, noise)
            for name, inputs in (('forward', forward), ('generated', state)):
                output = model.denoise(inputs, times, context, sizes)
                predictions[name].append({key: output[key][positive].cpu()
                    for key in ('keypoints_px', 'boxes_cxcywh_px')})
                errors[name].append((output['pred_latents']-clean).square().mean(-1)[positive].cpu())
    target = {name: torch.cat([row[name] for row in targets]) for name in targets[0]}
    branches = {}
    for name in predictions:
        prediction = {key: torch.cat([row[key] for row in predictions[name]]) for key in predictions[name][0]}
        branches[name] = dict(paired_metrics(prediction, target), foreground_latent_mse=float(torch.cat(errors[name]).mean()))
    report = dict(scope='fixed VAL paired-person diagnostic, no detector AP', image_ids=image_ids, seed=seed,
        people=len(target['joint_mask']), time=249, branches=branches,
        OKS_gap=branches['forward']['paired_OKS_mean']-branches['generated']['paired_OKS_mean'])
    (directory/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
