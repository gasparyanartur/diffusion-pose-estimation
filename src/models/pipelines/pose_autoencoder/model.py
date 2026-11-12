from __future__ import annotations

import torch
from torch import Tensor, nn

from src.data.pose_geometry import normalize_geometry, restore_geometry
from src.eval.metrics import CROWDPOSE_SIGMAS
from src.models.pipelines.pose_vae.model import residual_trunk


class PoseBoxAutoencoder(nn.Module):
    def __init__(self, width: int = 256, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = residual_trunk(46, width)
        self.latent_head = nn.Linear(width, latent_dim)
        self.decoder = residual_trunk(latent_dim, width)
        self.box_head = nn.Linear(width, 4)
        self.joint_head = nn.Linear(width, 28)
        nn.init.zeros_(self.box_head.weight)
        with torch.no_grad():
            self.box_head.bias.copy_(torch.tensor([.5, .5, -1.5, -1.5]))

    def encode(self, keypoints_px: Tensor, boxes_cxcywh_px: Tensor,
               joint_mask: Tensor, image_wh: Tensor) -> Tensor:
        """Encode supplied pixel geometry into one deterministic latent per person."""
        relative, placement = normalize_geometry(keypoints_px, boxes_cxcywh_px, joint_mask, image_wh)
        features = torch.cat((relative.flatten(-2), placement, joint_mask.to(relative.dtype)), -1)
        return self.latent_head(self.encoder(features))

    def decode(self, z: Tensor, image_wh: Tensor) -> dict[str, Tensor]:
        """Decode joints and boxes from only the latent and image-size metadata."""
        features = self.decoder(z)
        box = self.box_head(features)
        placement = torch.cat((box[..., :2], box[..., 2:].clamp(-10., 2.)), -1)
        return restore_geometry(self.joint_head(features).unflatten(-1, (14, 2)), placement, image_wh)

    def freeze_encoder(self):
        self.encoder.requires_grad_(False).eval()
        self.latent_head.requires_grad_(False).eval()


def joint_exponents(prediction: dict[str, Tensor], target: dict[str, Tensor]) -> Tensor:
    mask = target['joint_mask']
    difference = torch.where(mask[..., None], prediction['keypoints_px'] - target['keypoints_px'], 0.)
    sigmas = difference.new_tensor(CROWDPOSE_SIGMAS)
    area = .53 * target['boxes_cxcywh_px'][:, 2:].prod(-1)
    # B people, K=14 joints. This is the native CrowdPose OKS exponent <B,K>.
    return difference.square().sum(-1) / (8 * sigmas.square()[None] * area[:, None])


def aligned_box_metrics(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    pred_low, pred_high = prediction[:, :2]-prediction[:, 2:]/2, prediction[:, :2]+prediction[:, 2:]/2
    true_low, true_high = target[:, :2]-target[:, 2:]/2, target[:, :2]+target[:, 2:]/2
    overlap = (torch.minimum(pred_high, true_high)-torch.maximum(pred_low, true_low)).clamp_min(0).prod(-1)
    union = prediction[:, 2:].prod(-1)+target[:, 2:].prod(-1)-overlap
    enclosing = (torch.maximum(pred_high, true_high)-torch.minimum(pred_low, true_low)).prod(-1)
    epsilon = torch.finfo(prediction.dtype).eps
    iou = overlap/union.clamp_min(epsilon)
    return iou, iou-(enclosing-union)/enclosing.clamp_min(epsilon)


def geometry_loss_per_person(prediction: dict[str, Tensor], target: dict[str, Tensor], *,
                  center_min_side_px: float = 0.) -> dict[str, Tensor]:
    mask = target['joint_mask']
    pose = (joint_exponents(prediction, target)*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    boxes, predicted_boxes = target['boxes_cxcywh_px'], prediction['boxes_cxcywh_px']
    center_scale = boxes[:, 2:].clamp_min(center_min_side_px) if center_min_side_px else boxes[:, 2:]
    center = ((predicted_boxes[:, :2]-boxes[:, :2])/center_scale).square().mean(-1)
    size = (predicted_boxes[:, 2:]/boxes[:, 2:]).log().square().mean(-1)
    _, giou = aligned_box_metrics(predicted_boxes, boxes)
    return dict(loss=pose+center+size+(1-giou), pose=pose,
                box_center=center, box_size=size, giou=1-giou)


def geometry_loss(prediction: dict[str, Tensor], target: dict[str, Tensor], *,
                  center_min_side_px: float = 0.) -> dict[str, Tensor]:
    return {name: value.mean() for name, value in geometry_loss_per_person(
        prediction, target, center_min_side_px=center_min_side_px).items()}
