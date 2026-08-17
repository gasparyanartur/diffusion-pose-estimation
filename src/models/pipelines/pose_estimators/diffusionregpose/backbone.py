"""ED-Pose-compatible ResNet feature hierarchy for DiffusionRegPose."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter


class PositionEmbeddingSineHW(nn.Module):
    """Build normalized sine/cosine embeddings with separate H/W temperatures."""

    def __init__(self, num_pos_feats=128, temperature_h=20.0, temperature_w=20.0, normalize=True, scale=None):
        super().__init__()
        if scale is not None and not normalize:
            raise ValueError("normalize must be True when scale is provided")
        self.num_pos_feats = num_pos_feats
        self.temperature_h = temperature_h
        self.temperature_w = temperature_w
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, image: Tensor, padding_mask: Tensor) -> Tensor:
        if image.ndim != 4 or padding_mask.shape != image.shape[:1] + image.shape[2:]:
            raise ValueError("image must be [B,C,H,W] and padding_mask must be [B,H,W]")
        not_mask = ~padding_mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_x = self.temperature_w ** (2 * (torch.arange(self.num_pos_feats, device=image.device) // 2) / self.num_pos_feats)
        dim_y = self.temperature_h ** (2 * (torch.arange(self.num_pos_feats, device=image.device) // 2) / self.num_pos_feats)
        pos_x = x_embed[..., None] / dim_x
        pos_y = y_embed[..., None] / dim_y
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class FrozenBatchNorm2d(nn.Module):
    """Batch normalization with fixed affine parameters and running statistics."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + 1e-5).rsqrt()
        bias = bias - running_mean * scale
        return inputs * scale + bias


class DiffusionRegPoseResNetBackbone(nn.Module):
    """Return ResNet-50 layer2/layer3/layer4 feature maps in order."""

    _CHANNELS = {1: 256, 2: 512, 3: 1024, 4: 2048}

    def __init__(
        self,
        *,
        pretrained: bool = True,
        return_interm_indices: Sequence[int] = (1, 2, 3),
    ) -> None:
        super().__init__()
        indices = tuple(return_interm_indices)
        if indices not in ((0, 1, 2, 3), (1, 2, 3), (3,)):
            raise ValueError(f"unsupported ResNet feature indices: {indices}")
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.body = IntermediateLayerGetter(
            resnet,
            return_layers={f"layer{index + 1}": str(index) for index in indices},
        )
        self.num_channels = [self._CHANNELS[index + 1] for index in indices]

    def forward(self, image: Tensor, padding_mask: Tensor | None = None) -> list[Tensor]:
        features = list(self.body(image).values())
        if padding_mask is None:
            return features
        return [
            feature * (~F.interpolate(
                padding_mask[:, None].float(), size=feature.shape[-2:]
            ).to(torch.bool))
            for feature in features
        ]
