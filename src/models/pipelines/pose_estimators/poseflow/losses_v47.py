import torch
from torch.nn import functional as F

from src.models.pipelines.pose_autoencoder.model import geometry_loss_per_person
from .coordinates_v47 import coordinate_geometry
from .losses_v46 import task_average


def balanced_eos_loss(logits, rows):
    layers = len(logits)
    indices = rows['predict_indices'][None, ..., None].expand(layers, -1, -1, -1)
    predicted = logits.gather(-1, indices).squeeze(-1)
    labels = rows['eos_targets']
    errors = F.binary_cross_entropy_with_logits(predicted, labels[None].expand_as(predicted).to(predicted),
                                               reduction='none').mean(0)
    total, groups = errors.new_zeros(len(labels)), errors.new_zeros(len(labels))
    for mask in (labels, ~labels):
        count = mask.sum(-1)
        total = total+(errors*mask).sum(-1)/count.clamp_min(1)
        groups = groups+(count > 0)
    return (total/groups).mean()


def row_losses(output, rows, sources, geometry, image_wh, *, coordinate_weight=10., geometry_weight=.5):
    prediction = output['pred_coordinates']  # <L,B,R,Q,32>
    layers, batch, row_count, queries, _ = prediction.shape
    target = F.pad(sources['coordinates'], (0, 0, 0, queries-sources['coordinates'].shape[1]))
    mask = F.pad(sources['coordinate_mask'], (0, 0, 0, queries-sources['coordinate_mask'].shape[1]))
    difference = torch.where(mask[None, :, None], prediction-target[None, :, None], 0.)
    coordinate_errors = difference.square().sum(-1)/mask.sum(-1).clamp_min(1)[None, :, None]
    supervised = rows['supervised']
    geometry_errors = prediction.new_zeros(layers, batch, row_count, queries)
    if supervised.any():
        images, _, ranks = supervised.nonzero(as_tuple=True)
        target_indices = sources['target_indices'][images, ranks]
        targets = {name: value[target_indices] for name, value in geometry.items()}
        for layer in range(layers):
            decoded = coordinate_geometry(prediction[layer][supervised], image_wh[images])
            errors = geometry_loss_per_person(decoded, targets)['loss']
            geometry_errors[layer] = geometry_errors[layer].masked_scatter(supervised, errors)
    geometry_errors = geometry_errors+prediction.sum()*0
    losses = {}
    for name, task_mask in (('birth', rows['birth_mask']), ('refine', rows['refinement_mask'])):
        losses[f'coordinate_{name}'] = task_average(coordinate_errors, task_mask)
        losses[f'geometry_{name}'] = task_average(geometry_errors, task_mask)
    losses['eos'] = balanced_eos_loss(output['eos_logits'], rows)
    losses['total'] = coordinate_weight*(losses['coordinate_birth']+losses['coordinate_refine'])/2+(
        geometry_weight*(losses['geometry_birth']+losses['geometry_refine'])/2)+losses['eos']
    return losses
