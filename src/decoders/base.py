"""Decoders: feature embeddings (+ optional graph / initial keypoints) → predictions.

A decoder is the third stage of a :class:`~src.pose_estimator.PoseEstimator`.
It consumes embeddings from a feature embedder and produces a prediction dict.

Standard output contract (see ``docs/assumptions.md``):
    Required:
        "keypoints":       Tensor[B, N, K, 2]   in canonical (-1, 1) image space.
    Optional:
        "scores":          Tensor[B, N]         per-person/per-slot confidence in [0, 1].
        "keypoint_scores": Tensor[B, N, K]      per-keypoint confidence in [0, 1].

The ``graph`` and ``initial_keypoints`` kwargs are only required by refinement
decoders (e.g. graph diffusion); set-prediction decoders ignore them.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from torch import Tensor, nn

from src.base_config import BaseConfig
from src.data.graph import PoseGraphBatch


class BaseDecoder(nn.Module):
    """Final stage of a :class:`PoseEstimator`."""

    @abstractmethod
    def forward(
        self,
        embeddings: dict[str, Tensor],
        graph: PoseGraphBatch | None = None,
        initial_keypoints: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Return a prediction dict (see module docstring for contract)."""


class BaseDecoderConfig(BaseConfig):
    """Polymorphic config; subclasses build their decoder."""

    def build(self) -> BaseDecoder:
        raise NotImplementedError
