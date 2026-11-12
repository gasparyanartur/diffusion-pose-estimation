import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as functional
from torchvision.ops import generalized_box_iou

from ..diffusionregpose.evaluation import box_cxcywh_to_xyxy
from ..diffusionregpose.training import (
    DiffusionRegPoseOTAMatcher, _diffusionregpose_pairwise_matching_cost, _pairwise_box_iou,
)


def grid_anchors(rows, columns):
    y, x = torch.meshgrid((torch.arange(rows)+.5)/rows,
                          (torch.arange(columns)+.5)/columns, indexing='ij')
    return torch.stack((x, y), -1).flatten(0, 1)


def build_v35_targets(batch, *, proposals=50, generator=None, anchors=None, perturb_pixels=5.):
    if proposals < 1 or perturb_pixels < 0:
        raise ValueError('Positive proposal budget and nonnegative perturbation required')
    device = batch['gt_boxes_canon'].device
    boxes = batch['gt_boxes_canon'].detach().cpu()
    valid = batch['person_mask'].detach().cpu().bool()
    valid &= batch.get('gt_box_labeled', batch['person_mask']).detach().cpu().bool()
    valid &= torch.isfinite(boxes).all(-1) & (boxes[..., 2:] > 0).all(-1)
    sizes = batch['img_hw'].detach().cpu().flip(-1)
    image_ids = batch.get('image_id', torch.arange(len(boxes))).detach().cpu()
    pool = valid.nonzero()
    batch_size = len(boxes)
    result = dict(input_boxes=torch.zeros(batch_size, proposals, 4),
        boxes=torch.zeros(batch_size, proposals, 4),
        positive=torch.zeros(batch_size, proposals, dtype=torch.bool),
        perturbed=torch.zeros(batch_size, proposals, dtype=torch.bool),
        zero_fallback=torch.zeros(batch_size, proposals, dtype=torch.bool),
        target_indices=torch.full((batch_size, proposals), -1, dtype=torch.long),
        source_batch=torch.full((batch_size, proposals), -1, dtype=torch.long),
        source_person=torch.full((batch_size, proposals), -1, dtype=torch.long),
        source_image_ids=torch.full((batch_size, proposals), -1, dtype=torch.long),
        unique_people=valid.sum(-1),
        canonical_slots=torch.full((batch_size, proposals), -1, dtype=torch.long),
        slot_to_original=torch.arange(proposals).expand(batch_size, -1).clone())
    for image in range(batch_size):
        originals = valid[image].nonzero().flatten()
        count = len(originals)
        if count > proposals:
            raise ValueError(f'Person count {count} exceeds proposal budget {proposals}')
        foreign = pool[image_ids[pool[:, 0]] != image_ids[image]]
        for slot in range(proposals):
            wants_positive = slot < count+(proposals-count)//2
            if slot < count:
                person = originals[slot]
                source_image = image
                positive = True
            elif count and (wants_positive or not len(foreign)):
                person = originals[torch.randint(count, (), generator=generator)]
                source_image = image
                positive = True
            elif len(foreign):
                donor = foreign[torch.randint(len(foreign), (), generator=generator)]
                source_image, person = donor.tolist()
                positive = False
            else:
                result['zero_fallback'][image, slot] = True
                continue
            candidate = boxes[source_image, person].clone()
            if positive and slot >= count:
                candidate[:2] += torch.randn(2, generator=generator)*perturb_pixels/sizes[image]
                result['perturbed'][image, slot] = True
            result['input_boxes'][image, slot] = candidate
            result['positive'][image, slot] = positive
            result['source_batch'][image, slot] = source_image
            result['source_person'][image, slot] = person
            result['source_image_ids'][image, slot] = image_ids[source_image]
            if positive:
                result['boxes'][image, slot] = boxes[image, person]
                result['target_indices'][image, slot] = person
        if anchors is not None:
            if anchors.shape != (proposals, 2):
                raise ValueError('Grid shape disagrees with proposal budget')
            distance = torch.cdist(anchors.detach().cpu(), result['input_boxes'][image, :, :2]).square()
            rows, columns = linear_sum_assignment(distance.numpy())
            permutation = torch.empty(proposals, dtype=torch.long)
            permutation[rows] = torch.as_tensor(columns)
            for name, value in result.items():
                if name not in ('unique_people', 'canonical_slots'):
                    value[image] = value[image, permutation].clone()
        permutation = result['slot_to_original'][image]
        inverse = permutation.argsort()
        result['canonical_slots'][image, :count] = inverse[:count]
    result['input_pixels'] = result['input_boxes']*sizes[:, None].repeat(1, 1, 2)
    result['synthetic'] = result['perturbed'] | result['zero_fallback']
    return {name: value.to(device) for name, value in result.items()}


def box_task_loss(output, targets, *, weights=(2., 5., 2.)):
    positive = targets['positive']
    logits = output['existence_logits']
    existence = positive.to(logits)
    probability = logits.sigmoid()
    correct = probability*existence+(1-probability)*(1-existence)
    alpha = .25*existence+.75*(1-existence)
    count = positive.sum().to(logits).clamp_min(1)
    ce = functional.binary_cross_entropy_with_logits(logits, existence, reduction='none')
    predicted, target = output['pred_boxes'][positive], targets['boxes'][positive]
    losses = dict(loss_ce=(alpha*ce*(1-correct).square()).sum()/count,
                  loss_bbox=functional.l1_loss(predicted, target, reduction='sum')/count)
    if predicted.numel():
        giou = generalized_box_iou(box_cxcywh_to_xyxy(predicted), box_cxcywh_to_xyxy(target))
        losses['loss_giou'] = (1-giou.diag()).sum()/count
    else:
        losses['loss_giou'] = predicted.sum()
    losses['total'] = sum(w*v for w, v in zip(weights, losses.values()))
    return losses


def unique_box_targets(constructed):
    result = []
    for image, count in enumerate(constructed['unique_people'].tolist()):
        slots = constructed['canonical_slots'][image, :count]
        boxes = constructed['boxes'][image, slots]
        result.append(dict(boxes=boxes, labels=torch.ones(count, dtype=torch.long, device=boxes.device)))
    return result


class BoxOTAMatcher(DiffusionRegPoseOTAMatcher):
    def __init__(self, *, anchors=None, spatial_cost_weight=0., **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('query_anchors', anchors.clone() if anchors is not None else None)
        self.spatial_cost_weight = spatial_cost_weight
        self.diagnostics = {}

    def cost_components(self, output, target, image):
        boxes = output['pred_boxes'][image]
        cost = _diffusionregpose_pairwise_matching_cost(output['pred_logits'][image], boxes,
            None, target, num_body_points=self.num_body_points, sigmas=self.sigmas, weights=self.weights)
        penalty = self._candidate_penalties(boxes, target['boxes'])
        spatial = (torch.cdist(self.query_anchors.to(boxes), target['boxes'][:, :2]).square()
                   if self.query_anchors is not None else torch.zeros_like(cost))
        iou = _pairwise_box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target['boxes']))
        return cost, penalty, spatial, iou

    @torch.no_grad()
    def forward(self, output, targets, *, spatial_multipliers=None):
        boxes = output['pred_boxes']
        if spatial_multipliers is not None:
            if spatial_multipliers.shape != (len(boxes),):
                raise ValueError('Spatial multipliers must have shape [B]')
            if not torch.isfinite(spatial_multipliers).all() or ((spatial_multipliers < 0) | (spatial_multipliers > 1)).any():
                raise ValueError('Spatial multipliers must be finite and in [0, 1]')
            spatial_multipliers = spatial_multipliers.to(boxes)
        effective_weights = boxes.new_full((len(boxes),), self.spatial_cost_weight)
        if spatial_multipliers is not None:
            effective_weights = effective_weights*spatial_multipliers
        matches, spatial_values, weighted_values, matched_distances = [], [], [], []
        for image, target in enumerate(targets):
            count = len(target['labels'])
            if not count:
                empty = target['labels'].new_empty(0)
                matches.append((empty, empty))
                continue
            if len(output['pred_boxes'][image]) < count:
                raise ValueError('OTA requires at least one query per GT person')
            cost, penalty, spatial, iou = self.cost_components(output, target, image)
            weight = self.spatial_cost_weight if spatial_multipliers is None else effective_weights[image]
            match = self._dynamic_k_assignment(cost+penalty+weight*spatial, iou)
            matches.append(match)
            spatial_values.append(spatial.mean())
            weighted_values.append((weight*spatial).mean())
            matched_distances.append(spatial[match].mean())
        zero = output['pred_boxes'].new_zeros(())
        raw = torch.stack(spatial_values).mean() if spatial_values else zero
        weighted = (torch.stack(weighted_values).mean() if weighted_values else zero)
        if spatial_multipliers is None:
            weighted = raw*self.spatial_cost_weight
        self.diagnostics = dict(spatial_cost_raw=raw, spatial_cost_weighted=weighted,
            matched_anchor_distance=(torch.stack(matched_distances).mean().sqrt() if matched_distances else zero))
        if spatial_multipliers is not None:
            self.diagnostics.update(spatial_weight_mean=effective_weights.mean() if len(boxes) else zero,
                spatial_weight_min=effective_weights.min() if len(boxes) else zero,
                spatial_weight_max=effective_weights.max() if len(boxes) else zero)
        return matches


@torch.no_grad()
def assign_box_targets(clean, recovered, constructed, matcher, *, spatial_multipliers=None):
    targets = unique_box_targets(constructed)
    matches = (matcher(recovered, targets) if spatial_multipliers is None
               else matcher(recovered, targets, spatial_multipliers=spatial_multipliers))
    latents = clean['pred_latents'].detach()
    latent_targets = torch.zeros_like(latents)
    assigned = dict(positive=torch.zeros_like(constructed['positive']), boxes=torch.zeros_like(constructed['boxes']))
    assigned_people = torch.full_like(constructed['target_indices'], -1)
    null_count = 0
    for image, (queries, people) in enumerate(matches):
        slots = constructed['canonical_slots'][image, people]
        assigned['positive'][image, queries] = True
        assigned['boxes'][image, queries] = constructed['boxes'][image, slots]
        assigned_people[image, queries] = people
        latent_targets[image, queries] = latents[image, slots]
        unmatched = (~assigned['positive'][image]).nonzero().flatten()
        bank = (~constructed['positive'][image]).nonzero().flatten()
        if len(unmatched) and len(bank):
            distance = torch.cdist(recovered['pred_latents'][image, unmatched], latents[image, bank])
            selected = bank[distance.argmin(-1)]
            latent_targets[image, unmatched] = latents[image, selected]
        elif len(unmatched):
            null_count += len(unmatched)
    metadata = dict(matched_person=assigned_people,
        assigned_positive=assigned['positive'].sum(-1).float().mean(),
        unique_gt_coverage=latents.new_tensor(sum(p.unique().numel() for _, p in matches)/max(int(constructed['unique_people'].sum()), 1)),
        zero_background_latent_fraction=latents.new_tensor(null_count/assigned['positive'].numel()))
    return assigned, latent_targets, metadata
