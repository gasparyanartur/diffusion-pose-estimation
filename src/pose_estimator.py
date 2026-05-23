"""PoseEstimator: feature extractor → embedder → decoder.

Three-stage pipeline that produces predicted keypoints from images and an
optional initial keypoint state.

Stages:
  1. Feature extractor: image_ids → raw feature dict (e.g. DINO maps).
  2. Feature embedder: raw features + per-joint pixel coords → per-node embeddings.
  3. Decoder: embeddings + optional initial keypoints → predicted keypoints.

Pose estimation is the base task. Refinement is the special case where
``initial_keypoints`` is provided (e.g. detector output).
"""

from __future__ import annotations

from typing import Any

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
        graph: PoseGraphBatch,
        image_ids: list[int],
        kpt_pixel: Tensor,        # (B, N, K, 2) pixel-space (x, y)
        img_hw: Tensor,           # (B, 2) original image (H, W)
        joint_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run extractor + embedder, returning the embedding dict for the decoder."""
        features = self.feature_extractor(image_ids)
        return self.feature_embedder(
            features=features,
            kpt_pixel=kpt_pixel,
            img_hw=img_hw,
            joint_mask=joint_mask,
        )

    def forward(
        self,
        graph: PoseGraphBatch,
        image_ids: list[int],
        kpt_pixel: Tensor,
        img_hw: Tensor,
        initial_keypoints: Tensor | None = None,
        joint_mask: Tensor | None = None,
    ) -> Tensor:
        embeddings = self.encode(graph, image_ids, kpt_pixel, img_hw, joint_mask)
        return self.decoder(graph, embeddings, initial_keypoints=initial_keypoints)


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
