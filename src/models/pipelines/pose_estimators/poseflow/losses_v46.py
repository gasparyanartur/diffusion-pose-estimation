import torch
from torch.nn import functional as F

from src.models.pipelines.pose_autoencoder.model import geometry_loss_per_person


def task_average(errors, mask):
    per_person = (errors.mean(0)*mask).sum(1)/mask.sum(1).clamp_min(1)
    valid = mask.any(1)
    return per_person[valid].mean() if valid.any() else errors.sum()*0


def row_losses(output, rows, sources, geometry, codec, image_wh, *, latent_weight=10., geometry_weight=.5):
    prediction = output['pred_latents']  # L layers, B images, R rows, Q tokens, D channels.
    layers, batch, row_count, queries, _ = prediction.shape
    target = F.pad(sources['latents'], (0, 0, 0, queries-sources['latents'].shape[1]))
    latent_errors = (prediction-target[None, :, None]).square().mean(-1)
    supervised = rows['supervised']
    geometry_errors = prediction.new_zeros(layers, batch, row_count, queries)
    if supervised.any():
        images, _, ranks = supervised.nonzero(as_tuple=True)
        indices = sources['target_indices'][images, ranks]
        target_geometry = {name: value[indices] for name, value in geometry.items()}
        for layer in range(layers):
            decoded = codec.decode_latents(prediction[layer][supervised], image_wh[images])
            errors = geometry_loss_per_person(decoded, target_geometry)['loss']
            geometry_errors[layer] = geometry_errors[layer].masked_scatter(supervised, errors)
    # Empty geometry objectives remain connected to the decoder without training codec parameters.
    geometry_errors = geometry_errors+prediction.sum()*0
    losses = {}
    for name, mask in (('birth', rows['birth_mask']), ('refine', rows['refinement_mask'])):
        losses[f'latent_{name}'] = task_average(latent_errors, mask)
        losses[f'geometry_{name}'] = task_average(geometry_errors, mask)
    indices = rows['predict_indices'][None, ..., None].expand(layers, -1, -1, -1)
    logits = output['eos_logits'].gather(-1, indices).squeeze(-1)
    labels = rows['eos_targets']
    bce = F.binary_cross_entropy_with_logits(logits, labels[None].expand_as(logits).to(logits), reduction='none').mean(0)
    eos = bce.new_zeros(batch)
    groups = bce.new_zeros(batch)
    for mask in (labels, ~labels):
        count = mask.sum(-1)
        eos = eos+(bce*mask).sum(-1)/count.clamp_min(1)
        groups = groups+(count > 0)
    losses['eos'] = (eos/groups).mean()
    losses['total'] = latent_weight*(losses['latent_birth']+losses['latent_refine'])/2+(
        geometry_weight*(losses['geometry_birth']+losses['geometry_refine'])/2)+losses['eos']
    return losses
