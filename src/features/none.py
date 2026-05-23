"""No-op feature extractor: emits an empty dict."""

from __future__ import annotations

from typing import Any

from src.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig


class NoneFeatureExtractor(BaseFeatureExtractor):
    """Returns an empty dict; use when no image conditioning is required."""

    def forward(self, image_ids: list[int]) -> dict[str, Any]:
        return {}


class NoneFeatureExtractorConfig(BaseFeatureExtractorConfig):
    def build(self) -> NoneFeatureExtractor:
        return NoneFeatureExtractor()
