import torch
from torch.nn import functional as F

from src.models.pipelines.pose_autoencoder.model import geometry_loss_per_person
from .coordinates_v47 import coordinate_geometry


def prediction_readouts(values, indices):
    # Values <L,B,R,Q,D>; one PREDICT readout for each independent row.
    gather = indices[None, ..., None, None].expand(len(values), -1, -1, 1, values.shape[-1])
    return values.gather(3, gather).squeeze(3)


def balanced_eos_loss(logits, rows):
    predicted = logits.gather(-1, rows['predict_indices'][None, ..., None].expand(len(logits), -1, -1, 1)).squeeze(-1)
    targets = rows['eos_targets']
    errors = F.binary_cross_entropy_with_logits(predicted, targets[None].expand_as(predicted).to(predicted),
                                               reduction='none').mean(0)
    total, groups = errors.new_zeros(len(targets)), errors.new_zeros(len(targets))
    for mask in (targets, rows['birth_rows']):
        count = mask.sum(-1)
        total = total+(errors*mask).sum(-1)/count.clamp_min(1)
        groups = groups+(count > 0)
    return (total/groups).mean()


def row_losses(output, rows, sources, geometry, image_wh, *, coordinate_weight=10., geometry_weight=.5):
    prediction = prediction_readouts(output['pred_coordinates'], rows['predict_indices'])
    birth = rows['birth_rows']
    zero = prediction.sum()*0
    if birth.any():
        images, ranks = birth.nonzero(as_tuple=True)
        coordinates = prediction[:, images, ranks]
        target = sources['coordinates'][images, ranks]
        mask = sources['coordinate_mask'][images, ranks]
        difference = torch.where(mask[None], coordinates-target[None], 0.)
        coordinate_loss = (difference.square().sum(-1)/mask.sum(-1).clamp_min(1)[None]).mean()
        targets = {name: value[sources['target_indices'][images, ranks]] for name, value in geometry.items()}
        geometry_loss = torch.stack([geometry_loss_per_person(
            coordinate_geometry(layer, image_wh[images]), targets)['loss'].mean() for layer in coordinates]).mean()
    else:
        coordinate_loss = geometry_loss = zero
    eos = balanced_eos_loss(output['eos_logits'], rows)
    return dict(coordinate_birth=coordinate_loss, geometry_birth=geometry_loss, eos=eos,
                total=coordinate_weight*coordinate_loss+geometry_weight*geometry_loss+eos)
