"""Identity feature extractor: returns the image tensor unchanged."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.embedders.spatial_attention import make_patch_centers
from src.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig


class IdentityFeatureExtractor(BaseFeatureExtractor):
    """Returns the raw image under ``"spatial"`` with no processing.

    Output has ``out_channels = 3`` (RGB).  Useful for probing whether a probe
    architecture fits in memory when given full-resolution pixel features.
    """

    out_channels: int = 3

    def forward(
        self,
        image: Tensor | None = None,
        image_ids: list[int] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        if image is None:
            raise ValueError("IdentityFeatureExtractor requires `image`; got None")
        return {
            "spatial": image,
            "patch_centers": make_patch_centers(image.shape[2], image.shape[3], image.device, image.dtype),
        }

    def get_output_size(self, input_size: int) -> int:
        return input_size


class IdentityFeatureExtractorConfig(BaseFeatureExtractorConfig):
    def build(self) -> IdentityFeatureExtractor:
        return IdentityFeatureExtractor()
