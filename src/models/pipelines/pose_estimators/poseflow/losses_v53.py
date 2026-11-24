import torch
from torch.nn import functional as F

from .coordinates_v47 import coordinate_geometry
from .losses_v49_balanced import robust_geometry_errors


TASK_NAMES = ('pose_birth', 'box_birth')


def row_losses(output, rows, sources, geometry, image_wh, *, scales, coordinate_weight=10., geometry_weight=.5):
    prediction = output['pred_coordinates']  # <L,B,R,32>, newborn readouts only.
    zero = prediction.sum()*0
    birth = rows['birth_rows']
    errors = {name: zero for name in ('coordinate_joint', 'coordinate_box', 'geometry_pose',
                                      'geometry_box_center', 'geometry_box_size', 'geometry_giou')}
    if birth.any():
        images, ranks = birth.nonzero(as_tuple=True)
        predicted = prediction[:, images, ranks]
        target = sources['coordinates'][images, ranks]
        mask = sources['coordinate_mask'][images, ranks]
        difference = torch.where(mask[None], predicted-target[None], 0.)
        errors['coordinate_joint'] = (difference[..., :28].square().sum(-1)/mask[..., :28].sum(-1).clamp_min(1)[None]).mean()
        errors['coordinate_box'] = difference[..., 28:].square().mean()
        targets = {name: value[sources['target_indices'][images, ranks]] for name, value in geometry.items()}
        per_layer = [robust_geometry_errors(coordinate_geometry(layer, image_wh[images]), targets) for layer in predicted]
        for name in ('pose', 'box_center', 'box_size', 'giou'):
            errors[f'geometry_{name}'] = torch.stack([layer[name].mean() for layer in per_layer]).mean()
    pose = coordinate_weight*errors['coordinate_joint']+geometry_weight*errors['geometry_pose']
    box = coordinate_weight*errors['coordinate_box']+geometry_weight*sum(errors[f'geometry_{name}'] for name in ('box_center', 'box_size', 'giou'))
    losses = {f'{name}_birth': value for name, value in errors.items()}
    losses.update(pose_birth=pose, box_birth=box)
    logits = output['eos_logits'].mean(0)
    eos_errors = F.binary_cross_entropy_with_logits(logits, rows['eos_targets'].to(logits), reduction='none')
    total, groups = eos_errors.new_zeros(len(logits)), eos_errors.new_zeros(len(logits))
    for selected in (birth, rows['eos_targets']):
        count = selected.sum(-1)
        total = total+(eos_errors*selected).sum(-1)/count.clamp_min(1)
        groups = groups+(count > 0)
    losses['eos'] = (total/groups.clamp_min(1)).mean()
    for index, name in enumerate(TASK_NAMES):
        losses[f'{name}_contribution'] = .5*losses[name]/scales[index]
    losses['total'] = sum(losses[f'{name}_contribution'] for name in TASK_NAMES)+losses['eos']
    return losses
