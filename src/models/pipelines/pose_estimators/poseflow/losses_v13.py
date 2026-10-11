"""Per-layer set supervision for persistent latent person slots."""

import torch
from torch import Tensor

from .latent_v12 import prediction_loss


def real_person_targets(targets: dict[str, Tensor]) -> list[dict[str, Tensor]]:
    result = []
    for index, included in enumerate(targets['real']):
        boxes = targets['boxes'][index, included]
        joints = targets['joints'][index, included]
        result.append({
            'labels': torch.zeros(boxes.shape[0], device=boxes.device, dtype=torch.long),
            'boxes': boxes,
            'keypoints': torch.cat((joints.flatten(-2), targets['labeled'][index, included].float()), -1),
            'area': (boxes[:, 2] * boxes[:, 3]).clamp_min(1e-6),
        })
    return result


def matched_layer_losses(output, targets, clean, codec, matcher, weights):
    matches = matcher(output, real_person_targets(targets))
    aligned = {name: value.clone() for name, value in targets.items()}
    aligned['real'].zero_()
    aligned['existence'].zero_()
    aligned['labeled'].zero_()
    latent_targets = clean.detach().clone()
    with torch.no_grad():
        zeros = torch.zeros_like(targets['existence'])
        empty_existence = codec.clean_norm(
            codec.existence_encoder(zeros) + codec.roles.weight[0]
        )
        latent_targets[:, :, 0] = empty_existence
    for index, (outputs, people) in enumerate(matches):
        source_slots = targets['real'][index].nonzero(as_tuple=False).flatten()[people]
        aligned['real'][index, outputs] = True
        aligned['existence'][index, outputs] = 1
        for name in ('boxes', 'joints', 'labeled'):
            aligned[name][index, outputs] = targets[name][index, source_slots]
        latent_targets[index, outputs] = clean.detach()[index, source_slots]

    values = prediction_loss(output, aligned, weights=weights)
    slot_mean_classification = values['loss_ce']
    normalizer = aligned['real'].sum().clamp_min(1)
    values['loss_ce'] = slot_mean_classification * output['pred_logits'].numel() / normalizer
    values['total'] = values['total'] + weights.cls * (values['loss_ce'] - slot_mean_classification)
    squared_error = (output['pred_latents'] - latent_targets).square()
    included = aligned['real']
    geometry_terms = [
        squared_error[:, :, role][included].mean() if included.any()
        else squared_error[:, :, role].sum() * 0
        for role in (slice(1, 2), slice(2, 3), slice(3, None))
    ]
    values['loss_latent'] = torch.stack([
        squared_error[:, :, 0].mean(), *geometry_terms,
    ]).mean()
    values['positive_assignments'] = aligned['real'].sum().float()
    values['target_people'] = targets['real'].sum().float()
    values['assignments_per_person'] = values['positive_assignments'] / values['target_people'].clamp_min(1)
    return values
