import torch
from torch import Tensor
from torch.nn import functional as functional
from torchvision.ops import generalized_box_iou

from src.eval.metrics import CROWDPOSE_SIGMAS
from ..diffusionregpose.evaluation import box_cxcywh_to_xyxy


def _sample_indices(count: int, size: int, generator: torch.Generator | None) -> Tensor:
    return torch.randint(count, (size,), generator=generator)


def _pose_pool(points: Tensor, labeled: Tensor, people: Tensor):
    indices = (people.bool() & labeled.any(-1)).nonzero()
    filled = torch.where(labeled[..., None], (points + 1) / 2, .5)
    return indices, filled


def build_v26_targets(batch: dict[str, Tensor], *, proposals: int = 50,
                      generator: torch.Generator | None = None,
                      fallback_noise_pixels: float = 5.) -> dict[str, Tensor]:
    if proposals < 1 or fallback_noise_pixels < 0:
        raise ValueError('V26 requires a positive proposal budget and nonnegative fallback noise')
    device = batch['gt_keypoints'].device
    points = batch['gt_keypoints'].detach().cpu()
    labeled = batch['gt_labeled'].detach().cpu().bool()
    people = batch.get('person_mask', labeled.any(-1)).detach().cpu().bool()
    boxes = batch['gt_boxes_canon'].detach().cpu()
    sizes = batch['img_hw'].detach().cpu().flip(-1)
    image_ids = batch.get('image_id', torch.arange(len(points))).detach().cpu()
    pool, filled = _pose_pool(points, labeled, people)
    fallback_pool, fallback_points = pool, filled
    if not len(pool) and 'v26_original_points' in batch:
        fallback_pool, fallback_points = _pose_pool(
            batch['v26_original_points'].detach().cpu(),
            batch['v26_original_labeled'].detach().cpu().bool(),
            batch['v26_original_people'].detach().cpu().bool())
    if not len(fallback_pool):
        raise ValueError('V26 batch has no poses, including before augmentation')
    batch_size, _, joints, _ = points.shape
    targets = dict(
        input_pixels=torch.empty(batch_size, proposals, joints * 2),
        joints=torch.zeros(batch_size, proposals, joints, 2),
        boxes=torch.zeros(batch_size, proposals, 4),
        labeled=torch.zeros(batch_size, proposals, joints, dtype=torch.bool),
        positive=torch.zeros(batch_size, proposals, dtype=torch.bool),
        synthetic=torch.zeros(batch_size, proposals, dtype=torch.bool),
        target_indices=torch.full((batch_size, proposals), -1, dtype=torch.long),
        source_batch=torch.full((batch_size, proposals, 2), -1, dtype=torch.long),
        source_person=torch.full((batch_size, proposals, 2), -1, dtype=torch.long),
        source_image_ids=torch.full((batch_size, proposals, 2), -1, dtype=torch.long),
        unique_people=torch.zeros(batch_size, dtype=torch.long),
    )
    for recipient in range(batch_size):
        original = (people[recipient] & labeled[recipient].any(-1)).nonzero().flatten()
        count = len(original)
        if count > proposals:
            raise ValueError(f'V26 person count {count} exceeds proposal budget {proposals}')
        duplicate_count = (proposals-count)//2 if count else 0
        selected = torch.cat((original, original[_sample_indices(count, duplicate_count, generator)])) if count else original
        positive_count = len(selected)
        targets['unique_people'][recipient] = count
        targets['positive'][recipient, :positive_count] = True
        targets['target_indices'][recipient, :positive_count] = selected
        targets['joints'][recipient, :positive_count] = filled[recipient, selected]
        targets['boxes'][recipient, :positive_count] = boxes[recipient, selected]
        targets['labeled'][recipient, :positive_count] = labeled[recipient, selected]
        targets['input_pixels'][recipient, :positive_count] = (
            filled[recipient, selected] * sizes[recipient]).flatten(-2)
        targets['source_batch'][recipient, :positive_count, 0] = recipient
        targets['source_person'][recipient, :positive_count, 0] = selected
        targets['source_image_ids'][recipient, :positive_count, 0] = image_ids[recipient]
        negative_count = proposals-positive_count
        if not negative_count:
            continue
        foreign = pool[image_ids[pool[:, 0]] != image_ids[recipient]]
        slots = slice(positive_count, proposals)
        if len(foreign):
            donors = foreign[_sample_indices(len(foreign), negative_count, generator)]
            negative_points = filled[donors[:, 0], donors[:, 1]] * sizes[recipient]
            targets['source_batch'][recipient, slots, 0] = donors[:, 0]
            targets['source_person'][recipient, slots, 0] = donors[:, 1]
            targets['source_image_ids'][recipient, slots, 0] = image_ids[donors[:, 0]]
        else:
            donor_ids = _sample_indices(len(fallback_pool), negative_count*2, generator).reshape(-1, 2)
            donors = fallback_pool[donor_ids]
            negative_points = fallback_points[donors[..., 0], donors[..., 1]].mean(1) * sizes[recipient]
            negative_points += torch.randn(negative_points.shape, generator=generator) * fallback_noise_pixels
            targets['synthetic'][recipient, slots] = True
            targets['source_batch'][recipient, slots] = donors[..., 0]
            targets['source_person'][recipient, slots] = donors[..., 1]
            targets['source_image_ids'][recipient, slots] = image_ids[donors[..., 0]]
        targets['input_pixels'][recipient, slots] = negative_points.flatten(-2)
    return {name: value.to(device) for name, value in targets.items()}


def v26_task_loss(output: dict[str, Tensor], targets: dict[str, Tensor], *,
                  weights=(2., 5., 2., 10., 4.)) -> dict[str, Tensor]:
    positive = targets['positive']
    logits = output['existence_logits']
    existence = positive.to(logits.dtype)
    probability = logits.sigmoid()
    correct_probability = probability * existence + (1-probability) * (1-existence)
    alpha = .25 * existence + .75 * (1-existence)
    count = positive.sum().to(logits).clamp_min(1)
    cross_entropy = functional.binary_cross_entropy_with_logits(logits, existence, reduction='none')
    losses = {'loss_ce': (alpha * cross_entropy * (1-correct_probability).square()).sum()/count}
    predicted_boxes = output['pred_boxes'][positive]
    target_boxes = targets['boxes'][positive]
    losses['loss_bbox'] = functional.l1_loss(predicted_boxes, target_boxes, reduction='sum')/count
    if predicted_boxes.numel():
        giou = generalized_box_iou(box_cxcywh_to_xyxy(predicted_boxes), box_cxcywh_to_xyxy(target_boxes))
        losses['loss_giou'] = (1-giou.diag()).sum()/count
    else:
        losses['loss_giou'] = predicted_boxes.sum()
    points = output['joint_xy']
    valid = targets['labeled'] & positive[..., None]
    distance = (points-targets['joints']).square().sum(-1)
    losses['loss_keypoints'] = ((points-targets['joints']).abs()*valid[..., None]).sum()/count
    sigmas = points.new_tensor(CROWDPOSE_SIGMAS[:points.shape[-2]])
    area = targets['boxes'][..., 2:].prod(-1).clamp_min(1e-6)
    oks = torch.exp(-distance / (2*area[..., None]*(2*sigmas).square()))
    people_with_joints = valid.any(-1)
    losses['loss_oks'] = ((1-(oks*valid).sum(-1)/valid.sum(-1).clamp_min(1))*people_with_joints).sum()/count
    losses['total'] = sum(weight*value for weight, value in zip(weights, losses.values()))
    return losses
