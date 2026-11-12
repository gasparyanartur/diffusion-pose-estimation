import torch
from torchvision.ops import box_iou

from ..diffusionregpose.evaluation import box_cxcywh_to_xyxy


def build_fixed_source_targets(batch, *, proposals=50, generator=None):
    if proposals < 1:
        raise ValueError('Positive proposal budget required')
    device = batch['gt_boxes_canon'].device
    boxes = batch['gt_boxes_canon'].detach().cpu()
    valid = batch['person_mask'].detach().cpu().bool()
    valid &= batch.get('gt_box_labeled', batch['person_mask']).detach().cpu().bool()
    valid &= torch.isfinite(boxes).all(-1) & (boxes[..., 2:] > 0).all(-1)
    sizes = batch['img_hw'].detach().cpu().flip(-1)
    if not torch.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError('Finite positive image dimensions required')
    image_ids = batch.get('image_id', torch.arange(len(boxes))).detach().cpu()
    pool = valid.nonzero()
    shape = (len(boxes), proposals)
    result = dict(input_boxes=torch.zeros(*shape, 4), boxes=torch.zeros(*shape, 4),
        positive=torch.zeros(shape, dtype=torch.bool),
        random_negative=torch.zeros(shape, dtype=torch.bool),
        target_indices=torch.full(shape, -1, dtype=torch.long),
        source_batch=torch.full(shape, -1, dtype=torch.long),
        source_person=torch.full(shape, -1, dtype=torch.long),
        source_image_ids=torch.full(shape, -1, dtype=torch.long),
        unique_people=valid.sum(-1), canonical_slots=torch.full(shape, -1, dtype=torch.long),
        slot_to_original=torch.arange(proposals).expand(len(boxes), -1).clone(),
        negative_gt_iou=torch.zeros(shape))
    for image in range(len(boxes)):
        people = valid[image].nonzero().flatten()
        count = len(people)
        if count > proposals:
            raise ValueError(f'Person count {count} exceeds proposal budget {proposals}')
        foreign = pool[image_ids[pool[:, 0]] != image_ids[image]]
        for slot in range(proposals):
            if slot < count:
                source_image, person = image, int(people[slot])
                result['positive'][image, slot] = True
                result['target_indices'][image, slot] = person
                result['boxes'][image, slot] = boxes[image, person]
            elif len(foreign):
                source_image, person = foreign[torch.randint(len(foreign), (), generator=generator)].tolist()
            else:
                size = .05+.45*torch.rand(2, generator=generator)
                center = size/2+(1-size)*torch.rand(2, generator=generator)
                result['input_boxes'][image, slot] = torch.cat((center, size))
                result['random_negative'][image, slot] = True
                continue
            result['input_boxes'][image, slot] = boxes[source_image, person]
            result['source_batch'][image, slot] = source_image
            result['source_person'][image, slot] = person
            result['source_image_ids'][image, slot] = image_ids[source_image]
        if count and count < proposals:
            overlap = box_iou(box_cxcywh_to_xyxy(result['input_boxes'][image, count:]),
                              box_cxcywh_to_xyxy(boxes[image, people]))
            result['negative_gt_iou'][image, count:] = overlap.amax(-1)
        permutation = torch.randperm(proposals, generator=generator)
        for name, value in result.items():
            if name not in ('unique_people', 'canonical_slots'):
                value[image] = value[image, permutation].clone()
        result['canonical_slots'][image, :count] = permutation.argsort()[:count]
    result['input_pixels'] = result['input_boxes']*sizes[:, None].repeat(1, 1, 2)
    return {name: value.to(device) for name, value in result.items()}


def positive_latent_mse(predicted, target, positive):
    # Latents: <B, P, D>; positive: <B, P>.
    difference = (predicted[positive]-target.detach()[positive]).square()
    return difference.sum()/(positive.sum().clamp_min(1)*predicted.shape[-1])
