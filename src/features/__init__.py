"""Feature extractors: image-level features as a dict."""

from src.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig
from src.features.cached_dino import (
    CachedDinoFeatureExtractor,
    CachedDinoFeatureExtractorConfig,
)
from src.features.cnn import CNNFeatureExtractor, CNNFeatureExtractorConfig
from src.features.dino import (
    DinoFeatureExtractor,
    DinoFeatureExtractorConfig,
    DinoViTFeatureExtractor,
    DinoViTFeatureExtractorConfig,
)
from src.features.identity import IdentityFeatureExtractor, IdentityFeatureExtractorConfig
from src.features.none import NoneFeatureExtractor, NoneFeatureExtractorConfig
from src.features.resnet import ResNetFeatureExtractor, ResNetFeatureExtractorConfig

__all__ = [
    "BaseFeatureExtractor",
    "BaseFeatureExtractorConfig",
    "CNNFeatureExtractor",
    "CNNFeatureExtractorConfig",
    "CachedDinoFeatureExtractor",
    "CachedDinoFeatureExtractorConfig",
    "DinoFeatureExtractor",
    "DinoFeatureExtractorConfig",
    "DinoViTFeatureExtractor",
    "DinoViTFeatureExtractorConfig",
    "IdentityFeatureExtractor",
    "IdentityFeatureExtractorConfig",
    "NoneFeatureExtractor",
    "NoneFeatureExtractorConfig",
    "ResNetFeatureExtractor",
    "ResNetFeatureExtractorConfig",
]
