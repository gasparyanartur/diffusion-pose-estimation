"""Feature embedders: raw feature dict + keypoints → per-node embeddings.

Output contract (all keys optional unless required by the downstream consumer):
    "feature":      Tensor[B, D]           — global vector (set-prediction head)
    "tokens":       Tensor[B, T, D]        — spatial tokens (no pooling)
    "node_feats":   Tensor[B, N, K, D]     — per-node (refinement decoder)
    "global_feats": Tensor[B, D]           — per-batch global (refinement)

Children may add extra keys; consumers ignore unknown keys.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from torch import Tensor, nn

from src.base_config import BaseConfig


class BaseFeatureEmbedder(nn.Module):
    """Maps raw image features (+ optional keypoint locations) to embeddings.

    All keypoint-related arguments are optional: set-prediction embedders
    ignore them, refinement embedders read them.
    """

    @abstractmethod
    def forward(
        self,
        features: dict[str, Any],
        kpt_pixel: Tensor | None = None,
        img_hw: Tensor | None = None,
        joint_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        ...


class BaseFeatureEmbedderConfig(BaseConfig):
    """Polymorphic config; subclasses build their concrete embedder."""

    def build(self) -> BaseFeatureEmbedder:
        raise NotImplementedError
