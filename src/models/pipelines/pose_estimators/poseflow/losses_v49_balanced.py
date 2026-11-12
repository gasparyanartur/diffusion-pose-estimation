import torch
from torch.nn import functional as F

from src.models.pipelines.pose_autoencoder.model import aligned_box_metrics, joint_exponents
from .coordinates_v47 import coordinate_geometry
from .losses_v46 import task_average


TASK_NAMES = ('pose_birth', 'box_birth', 'pose_refine', 'box_refine')


def robust_square(squared):
    return torch.where(squared <= 1., squared, 2*squared.clamp_min(1e-12).sqrt()-1)


def robust_geometry_errors(prediction, target):
    mask = target['joint_mask']
    pose = (robust_square(joint_exponents(prediction, target))*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    boxes, predicted = target['boxes_cxcywh_px'], prediction['boxes_cxcywh_px']
    center = robust_square(((predicted[:, :2]-boxes[:, :2])/boxes[:, 2:]).square()).mean(-1)
    size = robust_square((predicted[:, 2:]/boxes[:, 2:]).log().square()).mean(-1)
    _, giou = aligned_box_metrics(predicted, boxes)
    return dict(pose=pose, box_center=center, box_size=size, giou=1-giou)


def refinement_buckets(mask):
    rows, queries = mask.shape[1:]
    ages = torch.arange(rows, device=mask.device)[:, None]-torch.arange(queries, device=mask.device)[None]
    return tuple(mask & selected[None] for selected in (ages == 1, ages == 2, ages >= 3))


def age_balanced_average(errors, mask, *, weights=(1., 1., 1.)):
    # Average layers, rows within each person's age bucket, present buckets, then people.
    per_person = errors.new_zeros(mask.shape[0], mask.shape[2])
    present = torch.zeros_like(per_person)
    for selected, weight in zip(refinement_buckets(mask), weights):
        counts = selected.sum(1)
        per_person = per_person+weight*(errors.mean(0)*selected).sum(1)/counts.clamp_min(1)
        present = present+(counts > 0)
    valid = present > 0
    return (per_person[valid]/present[valid]).mean() if valid.any() else errors.sum()*0


def boundary_eos_loss(logits, rows):
    indices = rows['predict_indices'][None, ..., None].expand(len(logits), -1, -1, -1)
    predicted = logits.gather(-1, indices).squeeze(-1)
    labels = rows['eos_targets']
    errors = F.binary_cross_entropy_with_logits(predicted, labels[None].expand_as(predicted).to(predicted),
                                               reduction='none').mean(0)
    first = labels & (labels.long().cumsum(-1) == 1)
    total, denominator = errors.new_zeros(len(labels)), errors.new_zeros(len(labels))
    for selected, weight in ((~labels, .5), (first, .25), (labels & ~first, .25)):
        count = selected.sum(-1)
        total = total+weight*(errors*selected).sum(-1)/count.clamp_min(1)
        denominator = denominator+weight*(count > 0)
    probability = predicted[-1].sigmoid()
    return dict(eos=(total/denominator).mean(),
        eos_first_probability=probability[first].mean(),
        eos_first_accuracy=(probability[first] >= .5).float().mean())


def prediction_errors(output, rows, sources, geometry, image_wh):
    prediction = output['pred_coordinates']  # <L,B,R,Q,32>
    layers, batch, row_count, queries, _ = prediction.shape
    target = F.pad(sources['coordinates'], (0, 0, 0, queries-sources['coordinates'].shape[1]))
    mask = F.pad(sources['coordinate_mask'], (0, 0, 0, queries-sources['coordinate_mask'].shape[1]))
    supervised = rows['supervised']
    valid = supervised[None, ..., None] & mask[None, :, None]
    difference = torch.where(valid, prediction-target[None, :, None], 0.)
    errors = dict(coordinate_joint=difference[..., :28].square().sum(-1)/mask[..., :28].sum(-1).clamp_min(1)[None, :, None],
                  coordinate_box=difference[..., 28:].square().mean(-1))
    zero = prediction[supervised[None].expand(layers, -1, -1, -1)].sum()*0
    geometry_errors = {name: [] for name in ('pose', 'box_center', 'box_size', 'giou')}
    images, _, ranks = supervised.nonzero(as_tuple=True)
    targets = {name: value[sources['target_indices'][images, ranks]] for name, value in geometry.items()}
    for layer in range(layers):
        values = robust_geometry_errors(coordinate_geometry(prediction[layer][supervised], image_wh[images]), targets)
        for name in geometry_errors:
            geometry_errors[name].append(prediction.new_zeros(batch, row_count, queries).masked_scatter(supervised, values[name])+zero)
    errors.update({name: torch.stack(value) for name, value in geometry_errors.items()})
    return errors


def row_losses(output, rows, sources, geometry, image_wh, *, scales, coordinate_weight=10., geometry_weight=.5):
    if scales.shape != (4,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Expected four finite positive fixed gradient scales')
    errors = prediction_errors(output, rows, sources, geometry, image_wh)
    pose = coordinate_weight*errors['coordinate_joint']+geometry_weight*errors['pose']
    box = coordinate_weight*errors['coordinate_box']+geometry_weight*(errors['box_center']+errors['box_size']+errors['giou'])
    birth, refinement = rows['birth_mask'], rows['refinement_mask']
    losses = dict(pose_birth=task_average(pose, birth), box_birth=task_average(box, birth),
        pose_refine=age_balanced_average(pose, refinement), box_refine=age_balanced_average(box, refinement))
    losses['pose_birth_contribution'] = .5*.25*losses['pose_birth']/scales[0]
    losses['box_birth_contribution'] = .5*.75*losses['box_birth']/scales[1]
    losses['pose_refine_contribution'] = .5*age_balanced_average(pose, refinement, weights=(.5,.75,.75))/scales[2]
    losses['box_refine_contribution'] = .5*age_balanced_average(box, refinement, weights=(.5,.25,.25))/scales[3]
    for name, error in errors.items():
        label = name if name.startswith('coordinate') else f'geometry_{name}'
        losses[f'{label}_birth'] = task_average(error, birth)
        losses[f'{label}_refine'] = age_balanced_average(error, refinement)
    for index, selected in enumerate(refinement_buckets(refinement)):
        for name, error in (('pose',pose),('box',box)):
            losses[f'{name}_age_bucket_{index+1}'] = task_average(error, selected)
    losses.update(boundary_eos_loss(output['eos_logits'], rows))
    losses['total'] = sum(losses[f'{name}_contribution'] for name in TASK_NAMES)+losses['eos']
    return losses
