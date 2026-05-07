"""PoseEstimator: feature extractor → embedder → decoder.

Three-stage pipeline that produces predicted keypoints (and per-slot /
per-keypoint scores) from images and optional refinement state.

Set prediction (default):
  image -> features -> embeddings -> decoder dict
    {"keypoints":[B,N,K,2], "scores":[B,N], "keypoint_scores":[B,N,K]}

Refinement:
  image (or image_ids) + initial_keypoints + graph -> same dict.

Keypoints are always returned in canonical ``(-1, 1)`` image space; see
``docs/assumptions.md``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.base_config import BaseConfig
from src.data.graph import PoseGraphBatch
from src.decoders.base import BaseDecoder, BaseDecoderConfig
from src.embedders.base import BaseFeatureEmbedder, BaseFeatureEmbedderConfig
from src.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig


class PoseEstimator(nn.Module):
    """End-to-end pose estimator composing extractor + embedder + decoder."""

    def __init__(
        self,
        feature_extractor: BaseFeatureExtractor,
        feature_embedder: BaseFeatureEmbedder,
        decoder: BaseDecoder,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_embedder = feature_embedder
        self.decoder = decoder

    def encode(
        self,
        image: Tensor | None = None,
        image_ids: list[int] | None = None,
        kpt_pixel: Tensor | None = None,
        img_hw: Tensor | None = None,
        joint_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Run extractor + embedder, returning the embedding dict for the decoder."""
        features = self.feature_extractor(image=image, image_ids=image_ids)
        return self.feature_embedder(
            features=features,
            kpt_pixel=kpt_pixel,
            img_hw=img_hw,
            joint_mask=joint_mask,
        )

    def forward(
        self,
        image: Tensor | None = None,
        image_ids: list[int] | None = None,
        kpt_pixel: Tensor | None = None,
        img_hw: Tensor | None = None,
        initial_keypoints: Tensor | None = None,
        graph: PoseGraphBatch | None = None,
        joint_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        embeddings = self.encode(
            image=image,
            image_ids=image_ids,
            kpt_pixel=kpt_pixel,
            img_hw=img_hw,
            joint_mask=joint_mask,
        )
        return self.decoder(
            embeddings=embeddings,
            graph=graph,
            initial_keypoints=initial_keypoints,
        )


class PoseEstimatorConfig(BaseConfig):
    feature_extractor: BaseFeatureExtractorConfig
    feature_embedder: BaseFeatureEmbedderConfig
    decoder: BaseDecoderConfig

    def build(self) -> PoseEstimator:
        return PoseEstimator(
            feature_extractor=self.feature_extractor.build(),
            feature_embedder=self.feature_embedder.build(),
            decoder=self.decoder.build(),
        )
