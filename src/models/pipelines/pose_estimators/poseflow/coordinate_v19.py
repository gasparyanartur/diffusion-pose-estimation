import torch
from torch import nn


class CoordinateState(nn.Module):
    def __init__(self, joints, epsilon=1e-5):
        super().__init__()
        self.joints = joints
        self.epsilon = epsilon
        self.width = 4 + 2*joints

    def encode(self, targets):
        geometry = torch.cat((targets['boxes'], targets['joints'].flatten(-2)), -1)
        return torch.logit(geometry.clamp(self.epsilon, 1-self.epsilon))

    def decode(self, coordinates):
        geometry = coordinates.sigmoid()
        points = geometry[..., 4:]
        return {'pred_boxes': geometry[..., :4],
                'pred_keypoints': torch.cat((points, torch.ones_like(points[..., :self.joints])), -1)}


def matched_coordinate_loss(output, targets, matcher, weights, *, matches=None):
    from .latent_v12 import prediction_loss
    from .losses_v13 import real_person_targets

    if matches is None:
        matches = matcher(output, real_person_targets(targets))
    aligned = {name:value.clone() for name,value in targets.items()}
    for name in ('real', 'existence', 'labeled'):
        aligned[name].zero_()
    for index, (candidates, people) in enumerate(matches):
        slots = targets['real'][index].nonzero(as_tuple=False).flatten()[people]
        aligned['real'][index, candidates] = True
        aligned['existence'][index, candidates] = 1
        for name in ('boxes', 'joints', 'labeled'):
            aligned[name][index, candidates] = targets[name][index, slots]
    values = prediction_loss(output, aligned, weights=weights)
    previous = values['loss_ce']
    positives = aligned['real'].sum()
    values['loss_ce'] = previous*output['pred_logits'].numel()/positives.clamp_min(1)
    values['total'] = values['total']+weights.cls*(values['loss_ce']-previous)
    values['positive_assignments'] = positives.float()
    values['target_people'] = targets['real'].sum().float()
    values['assignments_per_person'] = positives/values['target_people'].clamp_min(1)
    return values
