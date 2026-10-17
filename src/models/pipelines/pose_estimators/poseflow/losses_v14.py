import torch

from .latent_v12 import prediction_loss
from .losses_v13 import real_person_targets


def matched_v14_layer_losses(output, targets, clean, codec, matcher, weights):
    matches = matcher(output, real_person_targets(targets))
    aligned = {name: value.clone() for name, value in targets.items()}
    aligned['real'].zero_()
    aligned['existence'].zero_()
    aligned['labeled'].zero_()
    latent_targets = torch.zeros_like(clean)
    for index, (outputs, people) in enumerate(matches):
        source_slots = targets['real'][index].nonzero(as_tuple=False).flatten()[people]
        aligned['real'][index, outputs] = True
        aligned['existence'][index, outputs] = 1
        for name in ('boxes', 'joints', 'labeled'):
            aligned[name][index, outputs] = targets[name][index, source_slots]
        latent_targets[index, outputs] = clean.detach()[index, source_slots]
    values = prediction_loss(output, aligned, weights=weights)
    previous_classification = values['loss_ce']
    positives = aligned['real'].sum().clamp_min(1)
    values['loss_ce'] = previous_classification * output['pred_logits'].numel() / positives
    values['total'] = values['total'] + weights.cls * (values['loss_ce']-previous_classification)
    squared_error = (output['pred_latents']-latent_targets).square()
    box_mask = aligned['real']
    joint_mask = aligned['real'][..., None] & aligned['labeled']
    box_roles = getattr(codec, 'box_role_count', 1)
    box_terms = []
    for role in range(box_roles):
        box_error = squared_error[:, :, role][box_mask]
        term = box_error.sum() / max(box_error.numel(), 1)
        box_terms.append(term)
        if box_roles > 1:
            name = codec.latent_role_slices[role][0]
            values['loss_latent_'+name] = term
    joint_error = squared_error[:, :, box_roles:][joint_mask]
    values['loss_latent_box'] = torch.stack(box_terms).mean()
    values['loss_latent_joints'] = joint_error.sum() / max(joint_error.numel(), 1)
    values['loss_latent'] = (values['loss_latent_box']+values['loss_latent_joints']) / 2
    values['positive_assignments'] = aligned['real'].sum().float()
    values['target_people'] = targets['real'].sum().float()
    values['assignments_per_person'] = values['positive_assignments'] / values['target_people'].clamp_min(1)
    return values
