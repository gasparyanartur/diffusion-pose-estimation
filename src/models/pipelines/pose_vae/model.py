from __future__ import annotations

import torch
from torch import Tensor, nn

from src.data.pose_geometry import normalize_geometry, restore_geometry


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                    nn.Linear(width, width))

    def forward(self, features: Tensor) -> Tensor:
        return features + self.layers(features)


def residual_trunk(input_dim: int, width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(),
                         ResidualBlock(width), ResidualBlock(width), nn.LayerNorm(width))


class PoseBoxVAE(nn.Module):
    def __init__(self, width: int = 256, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = residual_trunk(46, width)
        self.posterior_mean = nn.Linear(width, latent_dim)
        self.posterior_logvar = nn.Linear(width, latent_dim)
        self.decoder = residual_trunk(latent_dim, width)
        self.box_head = nn.Linear(width, 4)
        self.joint_head = nn.Linear(width, 28)
        nn.init.constant_(self.posterior_logvar.bias, -4.)
        nn.init.zeros_(self.box_head.weight)
        with torch.no_grad():
            self.box_head.bias.copy_(torch.tensor([.5, .5, -1.5, -1.5]))

    def encode(self, keypoints_px: Tensor, boxes_cxcywh_px: Tensor,
               joint_mask: Tensor, image_wh: Tensor) -> tuple[Tensor, Tensor]:
        """Encode pixel geometry using the supplied box, without any clean reference."""
        relative, placement = normalize_geometry(keypoints_px, boxes_cxcywh_px, joint_mask, image_wh)
        features = self.encoder(torch.cat((relative.flatten(-2), placement,
                                           joint_mask.to(relative.dtype)), -1))
        return self.posterior_mean(features), self.posterior_logvar(features).clamp(-10., 2.)

    def decode(self, z: Tensor, image_wh: Tensor) -> dict[str, Tensor]:
        """Decode pixel geometry using only a latent and image-size metadata."""
        features = self.decoder(z)
        raw_box = self.box_head(features)
        placement = torch.cat((raw_box[..., :2], raw_box[..., 2:].clamp(-10., 2.)), -1)
        relative = self.joint_head(features).unflatten(-1, (14, 2))
        return restore_geometry(relative, placement, image_wh)


def reconstruction_loss(prediction: dict[str, Tensor], target: dict[str, Tensor]) -> dict[str, Tensor]:
    boxes = target['boxes_cxcywh_px']
    scale = boxes[:, 2:].amax(-1)
    mask = target['joint_mask']
    difference = (prediction['keypoints_px'] - target['keypoints_px']) / scale[:, None, None]
    difference = torch.where(mask[..., None], difference, 0.)
    pose = difference.square().sum((1, 2)) / (2 * mask.sum(-1).clamp_min(1))
    center = ((prediction['boxes_cxcywh_px'][:, :2] - boxes[:, :2]) / scale[:, None]).square().mean(-1)
    size = (prediction['boxes_cxcywh_px'][:, 2:] / boxes[:, 2:]).log().square().mean(-1)
    return dict(pose=pose.mean(), box=(center + size).mean(), reconstruction=(pose + center + size).mean())


def prior_kl(mean: Tensor, logvar: Tensor) -> Tensor:
    return .5 * (mean.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
