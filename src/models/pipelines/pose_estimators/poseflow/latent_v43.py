import torch
from torchvision.ops import box_iou, sigmoid_focal_loss

from .codec_v43 import geometry_from_batch
from ..diffusionregpose.evaluation import box_cxcywh_to_xyxy


def geometry_targets(batch):
    geometry, locations = geometry_from_batch(batch)
    targets = []
    for image in range(len(batch['img_hw'])):
        indices = (locations[:, 0] == image).nonzero().flatten()
        sizes = geometry['image_wh'][indices]
        points = geometry['keypoints_px'][indices]/sizes[:, None]
        boxes = geometry['boxes_cxcywh_px'][indices]/sizes.repeat(1, 2)
        targets.append(dict(boxes=boxes, labels=torch.ones(len(indices), dtype=torch.long, device=boxes.device),
            keypoints=torch.cat((points.flatten(1), geometry['joint_mask'][indices].to(points)), -1),
            area=boxes[:, 2:].prod(-1), geometry_indices=indices))
    return geometry, targets


@torch.no_grad()
def assign_latent_sources(codec, geometry, targets, matches, *, slots, generator, posterior_mode='sampled'):
    if any(len(target['labels']) > slots for target in targets):
        raise ValueError('Valid people exceed selected proposal capacity')
    if len(targets) != len(matches):
        raise ValueError('Assignments disagree with image count')
    if posterior_mode == 'sampled':
        codes = codec.sample_latents(geometry, generator=generator)
    elif posterior_mode == 'mean':
        mean, _ = codec.encode_posterior(geometry)
        codes = codec.standardize(mean)
    else:
        raise ValueError('Unknown posterior mode')
    source = codes.new_zeros(len(targets), slots, codec.latent_dim)
    positive = torch.zeros(len(targets), slots, dtype=torch.bool, device=codes.device)
    indices = torch.full((len(targets), slots), -1, dtype=torch.long, device=codes.device)
    for image, ((queries, people), target) in enumerate(zip(matches, targets)):
        if (len(people) != len(target['labels']) or len(queries.unique()) != len(queries)
                or not torch.equal(people.sort().values, torch.arange(len(people), device=people.device))):
            raise ValueError('Assignment must cover each target exactly once')
        rows = target['geometry_indices'][people]
        source[image, queries] = codes[rows]
        positive[image, queries] = True
        indices[image, queries] = rows
    return dict(latents=source, positive=positive, target_indices=indices)


def balanced_latent_loss(predicted, target, positive):
    error = (predicted-target.detach()).square().mean(-1)
    foreground = error[positive].mean() if positive.any() else predicted.sum()*0
    background = error[~positive].mean() if (~positive).any() else predicted.sum()*0
    return dict(positive=foreground, background=background, total=foreground+background)


def slot_focal_loss(logits, positive):
    return sigmoid_focal_loss(logits, positive.to(logits), alpha=.25, gamma=2.,
                              reduction='sum')/positive.sum().clamp_min(1)


def proposal_targets(candidates, targets, matches):
    boxes = torch.zeros_like(candidates['pred_boxes'])
    positive = torch.zeros_like(candidates['existence_logits'], dtype=torch.bool)
    for image, ((queries, people), target) in enumerate(zip(matches, targets)):
        boxes[image, queries] = target['boxes'][people]
        positive[image, queries] = True
    return dict(boxes=boxes, positive=positive)


@torch.no_grad()
def reference_statistics(candidates, references, targets, matches):
    measurements = dict(candidate_recall_iou50=[], selected_recall_iou50=[],
                        selected_matched_iou=[], selected_matched_l1=[])
    for image, (target, (queries, people)) in enumerate(zip(targets, matches)):
        if not len(target['boxes']):
            continue
        truth = box_cxcywh_to_xyxy(target['boxes'])
        candidate_iou = box_iou(box_cxcywh_to_xyxy(candidates['pred_boxes'][image]), truth)
        selected_iou = box_iou(box_cxcywh_to_xyxy(references['boxes'][image]), truth)
        measurements['candidate_recall_iou50'].append((candidate_iou.amax(0) >= .5).float())
        measurements['selected_recall_iou50'].append((selected_iou.amax(0) >= .5).float())
        measurements['selected_matched_iou'].append(selected_iou[queries, people])
        measurements['selected_matched_l1'].append(
            (references['boxes'][image, queries]-target['boxes'][people]).abs().mean(-1))
    return {name: torch.cat(rows).mean() for name, rows in measurements.items() if rows}
