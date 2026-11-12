import torch


def build_coordinate_targets(batch, *, slots, scale, generator):
    boxes = batch['gt_boxes_canon']
    valid = batch['person_mask'].bool() & batch.get('gt_box_labeled', batch['person_mask']).bool()
    valid &= torch.isfinite(boxes).all(-1) & (boxes[..., 2:] > 0).all(-1)
    if (valid.sum(-1) > slots).any():
        raise ValueError('GT person count exceeds coordinate query budget')
    clean = boxes.new_zeros(len(boxes), slots, 4)
    positive = torch.zeros(len(boxes), slots, dtype=torch.bool, device=boxes.device)
    source = torch.randn(len(boxes), slots, 4, generator=generator).to(boxes)*scale
    targets = []
    for image, mask in enumerate(valid):
        truth = boxes[image, mask]
        count = len(truth)
        clean[image, :count] = truth
        positive[image, :count] = True
        source[image, :count] = scale*(2*truth-1)
        targets.append(dict(boxes=truth, labels=torch.ones(count, dtype=torch.long, device=boxes.device)))
    return dict(boxes=clean, positive=positive, source_coordinates=source,
                unique_people=valid.sum(-1), targets=targets)


def assign_detection_targets(output, targets, matcher):
    matches = matcher(output, targets['targets'], include_keypoints=False)
    indices = torch.full(output['pred_boxes'].shape[:2], -1, device=output['pred_boxes'].device,
                         dtype=torch.long)
    for image, (queries, people) in enumerate(matches):
        indices[image, queries] = people
    positive = indices >= 0
    boxes = targets['boxes'].gather(1, indices.clamp_min(0)[..., None].expand(-1, -1, 4))
    return dict(boxes=boxes, positive=positive), indices


def assign_coordinate_targets(output, clean_embeddings, targets, matcher):
    assigned, indices = assign_detection_targets(output, targets, matcher)
    embeddings = clean_embeddings.gather(1,
        indices.clamp_min(0)[..., None].expand(-1, -1, clean_embeddings.shape[-1]))
    return assigned, embeddings, indices


def matched_embedding_loss(prediction, target, positive):
    difference = (prediction[positive]-target[positive]).square()
    return difference.sum()/(positive.sum().clamp_min(1)*prediction.shape[-1])
