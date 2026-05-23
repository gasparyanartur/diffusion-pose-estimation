"""No-op embedder: returns an empty dict."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from src.embedders.base import BaseFeatureEmbedder, BaseFeatureEmbedderConfig


class NoneEmbedder(BaseFeatureEmbedder):
    """Emits an empty dict (no node features, no global features)."""

    def forward(
        self,
        features: dict[str, Any],
        kpt_pixel: Tensor,
        img_hw: Tensor,
        joint_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return {}


class NoneEmbedderConfig(BaseFeatureEmbedderConfig):
    def build(self) -> NoneEmbedder:
        return NoneEmbedder()
