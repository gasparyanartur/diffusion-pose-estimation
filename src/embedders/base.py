"""Feature embedders: raw feature dict + keypoints → per-node embeddings.

Output contract:
    "node_feats":   Tensor[B, N, K, D_node]   (required)
    "global_feats": Tensor[B, D_global]       (optional)

Children may add extra keys; consumers ignore unknown keys.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from torch import Tensor, nn

from src.base_config import BaseConfig


class BaseFeatureEmbedder(nn.Module):
    """Maps raw image features + keypoint locations to per-node embeddings."""

    @abstractmethod
    def forward(
        self,
        features: dict[str, Any],
        kpt_pixel: Tensor,
        img_hw: Tensor,
        joint_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        ...


class BaseFeatureEmbedderConfig(BaseConfig):
    """Polymorphic config; subclasses build their concrete embedder."""

    def build(self) -> BaseFeatureEmbedder:
        raise NotImplementedError
