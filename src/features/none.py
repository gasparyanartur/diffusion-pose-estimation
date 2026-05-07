"""No-op feature extractor: emits an empty dict."""

from __future__ import annotations

from typing import Any

import torch

from src.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig


class NoneFeatureExtractor(BaseFeatureExtractor):
    """Returns an empty dict; use when no image conditioning is required."""

    def forward(
        self,
        image: torch.Tensor | None = None,
        image_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {}


class NoneFeatureExtractorConfig(BaseFeatureExtractorConfig):
    def build(self) -> NoneFeatureExtractor:
        return NoneFeatureExtractor()
