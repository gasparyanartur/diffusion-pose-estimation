"""Feature embedders: raw feature dict + keypoints → per-node embedding dict."""

from src.embedders.base import BaseFeatureEmbedder, BaseFeatureEmbedderConfig
from src.embedders.bilinear import BilinearSampleEmbedder, BilinearSampleEmbedderConfig
from src.embedders.cross_attention import (
    CrossAttentionEmbedder,
    CrossAttentionEmbedderConfig,
)
from src.embedders.none import NoneEmbedder, NoneEmbedderConfig

__all__ = [
    "BaseFeatureEmbedder",
    "BaseFeatureEmbedderConfig",
    "BilinearSampleEmbedder",
    "BilinearSampleEmbedderConfig",
    "CrossAttentionEmbedder",
    "CrossAttentionEmbedderConfig",
    "NoneEmbedder",
    "NoneEmbedderConfig",
]
