"""Fixed-slot targets, clean latent codec, and V12 diffusion losses."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou

from ..diffusionregpose.transformer import MLP
from ..diffusionregpose.evaluation import box_cxcywh_to_xyxy
from src.eval.metrics import CROWDPOSE_SIGMAS


def build_v12_targets(batch: dict[str, Tensor], *, groups: int,
                      generator: torch.Generator | None = None) -> dict[str, Tensor]:
    if groups < 1:
        raise ValueError('V12 requires at least one fixed person group')
    points = (batch['gt_keypoints'] + 1) / 2
    labeled = batch['gt_labeled'].bool()
    boxes = batch['gt_boxes_canon']
    if points.shape[:-1] != labeled.shape or boxes.shape[:2] != points.shape[:2]:
        raise ValueError('V12 target shapes disagree')
    batch_size, _, joints, _ = points.shape
    device, dtype = points.device, points.dtype
    empty_joints = torch.rand(batch_size, groups, joints, 2, device=device,
                              dtype=dtype, generator=generator)
    minimum, maximum = empty_joints.amin(-2), empty_joints.amax(-2)
    targets = {
        'existence': torch.zeros(batch_size, groups, 1, device=device, dtype=dtype),
        'real': torch.zeros(batch_size, groups, device=device, dtype=torch.bool),
        'joints': empty_joints,
        'boxes': torch.cat(((minimum + maximum) / 2, (maximum - minimum).clamp_min(1e-6)), -1),
        'labeled': torch.ones(batch_size, groups, joints, device=device, dtype=torch.bool),
        'overflow': torch.zeros(batch_size, device=device, dtype=torch.long),
    }
    people = batch.get('person_mask', labeled.any(-1)).bool()
    for index in range(batch_size):
        valid = people[index].nonzero(as_tuple=False).flatten()
        targets['overflow'][index] = max(valid.numel() - groups, 0)
        valid = valid[torch.randperm(valid.numel(), device=device, generator=generator)[:groups]]
        slots = torch.randperm(groups, device=device, generator=generator)[:valid.numel()]
        targets['real'][index, slots] = True
        targets['existence'][index, slots] = 1
        targets['boxes'][index, slots] = boxes[index, valid]
        mask = labeled[index, valid]
        targets['labeled'][index, slots] = mask
        targets['joints'][index, slots] = torch.where(mask[..., None], points[index, valid], 0.5)
    return targets


class V12StateCodec(nn.Module):
    def __init__(self, *, width: int, joints: int):
        super().__init__()
        self.joints = joints
        self.existence_encoder = MLP(1, width, width, 2)
        self.center_encoder = MLP(2, width, width, 2)
        self.size_encoder = MLP(2, width, width, 2)
        self.joint_encoder = MLP(2, width, width, 2)
        self.roles = nn.Embedding(joints + 3, width)
        self.clean_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.existence_head = MLP(width, width, 1, 3)
        self.point_head = MLP(width, width, 2, 3)
        self.size_head = MLP(width, width, 2, 3)

    def encode(self, targets: dict[str, Tensor]) -> Tensor:
        tokens = torch.cat((
            self.existence_encoder(targets['existence'])[:, :, None],
            self.center_encoder(targets['boxes'][..., :2])[:, :, None],
            self.size_encoder(targets['boxes'][..., 2:])[:, :, None],
            self.joint_encoder(targets['joints']),
        ), dim=2)
        return self.clean_norm(tokens + self.roles.weight[None, None])

    def decode(self, latents: Tensor) -> dict[str, Tensor]:
        points = self.point_head(latents[:, :, 3:]).sigmoid()
        center = self.point_head(latents[:, :, 1]).sigmoid()
        size = self.size_head(latents[:, :, 2]).sigmoid().clamp_min(1e-6)
        return {
            'pred_logits': self.existence_head(latents[:, :, 0]),
            'pred_boxes': torch.cat((center, size), -1),
            'pred_keypoints': torch.cat((points.flatten(-2), torch.ones_like(points[..., 0])), -1),
        }


def prediction_loss(output: dict[str, Tensor], targets: dict[str, Tensor], *,
                    reconstruct: bool = False, weights=None) -> dict[str, Tensor]:
    logits = output['pred_logits']
    probability = logits.sigmoid()
    existence = targets['existence']
    cross_entropy = F.binary_cross_entropy_with_logits(logits, existence, reduction='none')
    correct_probability = probability * existence + (1 - probability) * (1 - existence)
    alpha = 0.25 * existence + 0.75 * (1 - existence)
    values = {'loss_ce': (alpha * cross_entropy * (1 - correct_probability).square()).mean()}
    included = torch.ones_like(targets['real']) if reconstruct else targets['real']
    predicted_boxes, boxes = output['pred_boxes'][included], targets['boxes'][included]
    count = included.sum().clamp_min(1)
    values['loss_bbox'] = F.l1_loss(predicted_boxes, boxes, reduction='sum') / count
    if predicted_boxes.numel():
        giou = generalized_box_iou(box_cxcywh_to_xyxy(predicted_boxes), box_cxcywh_to_xyxy(boxes))
        values['loss_giou'] = (1 - giou.diag()).sum() / count
    else:
        values['loss_giou'] = predicted_boxes.sum()
    joints = targets['joints'].shape[-2]
    points = output['pred_keypoints'][..., :2*joints].reshape(*included.shape, joints, 2)
    squared_distance = (points - targets['joints']).square().sum(-1)
    valid = targets['labeled'] & included[..., None]
    values['loss_keypoints'] = ((points - targets['joints']).abs() * valid[..., None]).sum() / count
    sigmas = torch.as_tensor(CROWDPOSE_SIGMAS[:joints], device=logits.device, dtype=logits.dtype)
    area = (targets['boxes'][..., 2] * targets['boxes'][..., 3]).clamp_min(1e-6)
    oks = torch.exp(-squared_distance / (2 * area[..., None] * (2 * sigmas).square()))
    valid_people = valid.any(-1)
    values['loss_oks'] = ((1 - (oks * valid).sum(-1) / valid.sum(-1).clamp_min(1)) * valid_people).sum() / count
    coefficients = (2., 5., 2., 10., 4.) if weights is None else (
        weights.cls, weights.bbox, weights.giou, weights.keypoints, weights.oks,
    )
    values['total'] = sum(coefficient * values[name] for coefficient, name in zip(
        coefficients, ('loss_ce', 'loss_bbox', 'loss_giou', 'loss_keypoints', 'loss_oks'),
    ))
    return values


def latent_loss(predicted: Tensor, clean: Tensor) -> Tensor:
    difference = (predicted - clean).square()
    return torch.stack([difference[..., subset, :].mean() for subset in (
        slice(0, 1), slice(1, 2), slice(2, 3), slice(3, None),
    )]).mean()


def cosine_alpha(times: Tensor) -> Tensor:
    if not torch.isfinite(times).all() or ((times < 0) | (times > 1)).any():
        raise ValueError('V12 diffusion times must be finite and in [0,1]')
    alpha = torch.cos((times + .008) / 1.008 * math.pi / 2).square()
    alpha = (alpha / math.cos(.008 / 1.008 * math.pi / 2)**2).clamp(0, 1)
    return torch.where(times == 1, torch.zeros_like(alpha),
                       torch.where(times == 0, torch.ones_like(alpha), alpha))


def _broadcast_alpha(times: Tensor, latents: Tensor) -> Tensor:
    return cosine_alpha(times).reshape(-1, *([1] * (latents.ndim - 1)))


def corrupt_latents(clean: Tensor, noise: Tensor, times: Tensor) -> Tensor:
    alpha = _broadcast_alpha(times, clean)
    return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise


def ddim_update(latents: Tensor, predicted_clean: Tensor, times: Tensor,
                next_times: Tensor) -> Tensor:
    if (times <= 0).any() or (next_times > times).any():
        raise ValueError('DDIM requires a positive current time and descending next time')
    alpha = _broadcast_alpha(times, latents)
    next_alpha = _broadcast_alpha(next_times, latents)
    noise = (latents - alpha.sqrt() * predicted_clean) / (1 - alpha).sqrt().clamp_min(1e-8)
    return next_alpha.sqrt() * predicted_clean + (1 - next_alpha).sqrt() * noise
