"""Decoders: feature embeddings (+ optional initial keypoints) → predicted keypoints.

A decoder is the third stage of a :class:`~src.pose_estimator.PoseEstimator`.
It consumes per-node embeddings emitted by a feature embedder, optionally an
initial keypoint state, and produces predicted keypoints in canonical space.
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
        graph: PoseGraphBatch,
        embeddings: dict[str, Tensor],
        initial_keypoints: Tensor | None = None,
    ) -> Tensor:
        """Return predicted keypoints in canonical space, shape ``(V, 2)``."""


class BaseDecoderConfig(BaseConfig):
    """Polymorphic config; subclasses build their decoder."""

    def build(self) -> BaseDecoder:
        raise NotImplementedError
