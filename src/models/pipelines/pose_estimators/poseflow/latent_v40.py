import torch
from torchvision.ops import sigmoid_focal_loss

from .latent_v35 import box_task_loss


def proposal_detection_loss(candidates, targets, *, weights):
    losses = box_task_loss(candidates, targets, weights=weights)
    labels = torch.zeros_like(candidates['pred_logits'])
    labels[..., 1] = targets['positive'].to(labels)
    losses['loss_ce'] = sigmoid_focal_loss(candidates['pred_logits'], labels,
        alpha=.25, gamma=2., reduction='sum')/targets['positive'].sum().clamp_min(1)
    losses['total'] = sum(weight*losses[name] for weight, name in
                          zip(weights, ('loss_ce', 'loss_bbox', 'loss_giou')))
    return losses


def gather_references(candidates, indices):
    return dict(indices=indices,
        features=candidates['features'].gather(1, indices[..., None].expand(-1, -1, candidates['features'].shape[-1])),
        boxes=candidates['pred_boxes'].gather(1, indices[..., None].expand(-1, -1, 4)))


def select_training_references(candidates, batch, matcher, *, slots):
    boxes = batch['gt_boxes_canon']
    valid = batch['person_mask'].bool() & batch.get('gt_box_labeled', batch['person_mask']).bool()
    valid = valid & torch.isfinite(boxes).all(-1) & (boxes[..., 2:] > 0).all(-1)
    if (valid.sum(-1) > slots).any():
        raise ValueError('GT person count exceeds latent slot budget')
    targets = [dict(boxes=row[mask], labels=torch.ones(int(mask.sum()), dtype=torch.long, device=boxes.device))
               for row, mask in zip(boxes, valid)]
    matches = matcher(candidates, targets, include_keypoints=False)
    candidate_boxes = torch.zeros_like(candidates['pred_boxes'])
    positive = torch.zeros_like(candidates['existence_logits'], dtype=torch.bool)
    original_people = torch.full_like(positive, -1, dtype=torch.long)
    selected = []
    with torch.no_grad():
        for image, ((queries, people), mask) in enumerate(zip(matches, valid)):
            candidate_boxes[image, queries] = targets[image]['boxes'][people]
            positive[image, queries] = True
            original_people[image, queries] = mask.nonzero().flatten()[people]
            ranking = candidates['pred_logits'][image].amax(-1).argsort(descending=True, stable=True)
            remainder = ranking[~positive[image, ranking]][:slots-len(queries)]
            keep = torch.zeros_like(positive[image])
            keep[queries] = True
            keep[remainder] = True
            selected.append(ranking[keep[ranking]])
        indices = torch.stack(selected)
        slot_boxes = candidate_boxes.gather(1, indices[..., None].expand(-1, -1, 4))
        slot_positive = positive.gather(1, indices)
        sizes = batch['img_hw'].flip(-1).to(slot_boxes).repeat(1, 2)
        slot_targets = dict(boxes=slot_boxes, input_pixels=slot_boxes*sizes[:, None],
            positive=slot_positive, target_indices=original_people.gather(1, indices),
            unique_people=valid.sum(-1))
    references = gather_references(candidates, indices)
    return references, slot_targets, dict(boxes=candidate_boxes, positive=positive)
