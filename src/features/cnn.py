"""Plain CNN feature extractor (stride 16).

Trainable from scratch, used as the baseline backbone in the set-prediction
benchmark. Image is normalised with ImageNet statistics inside the module so
upstream code feeds raw [0, 1] tensors.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.embedders.spatial_attention import make_patch_centers
from src.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _conv_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CNNFeatureExtractor(BaseFeatureExtractor):
    """Plain 4-stage CNN with total stride 16. Output dict key: ``"spatial"``."""

    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        self.stage1 = _conv_block(3, 32, stride=2)
        self.stage2 = _conv_block(32, 64, stride=2)
        self.stage3 = _conv_block(64, 128, stride=2)
        self.stage4 = _conv_block(128, out_channels, stride=2)
        self.out_channels = out_channels
        self.register_buffer("_imagenet_mean", _IMAGENET_MEAN)
        self.register_buffer("_imagenet_std", _IMAGENET_STD)

    def forward(
        self,
        image: Tensor | None = None,
        image_ids: list[int] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        if image is None:
            raise ValueError("CNNFeatureExtractor requires `image`; got None")
        x = (image - self._imagenet_mean) / self._imagenet_std
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return {
            "spatial": x,
            "patch_centers": make_patch_centers(x.shape[2], x.shape[3], x.device, x.dtype),
        }

    def get_output_size(self, input_size: int) -> int:
        return input_size // 16


class CNNFeatureExtractorConfig(BaseFeatureExtractorConfig):
    out_channels: int = 128

    def build(self) -> CNNFeatureExtractor:
        return CNNFeatureExtractor(out_channels=self.out_channels)
