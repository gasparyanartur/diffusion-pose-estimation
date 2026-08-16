"""Feature extractors: image-level features as a dict."""

from src.models.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig
from src.models.features.cnn import CNNFeatureExtractor, CNNFeatureExtractorConfig
from src.models.features.concat import ConcatFeatureExtractor, ConcatFeatureExtractorConfig
from src.models.features.dino import (
    DinoFeatureExtractor,
    DinoFeatureExtractorConfig,
    DinoViTFeatureExtractor,
    DinoViTFeatureExtractorConfig,
)
from src.models.features.identity import IdentityFeatureExtractor, IdentityFeatureExtractorConfig
from src.models.features.none import NoneFeatureExtractor, NoneFeatureExtractorConfig
from src.models.features.radio import (
    CRadioV3BFeatureExtractorConfig,
    CRadioV4HFeatureExtractorConfig,
    RadioFeatureExtractor,
    RadioFeatureExtractorConfig,
)
from src.models.features.resnet import ResNetFeatureExtractor, ResNetFeatureExtractorConfig
from src.models.features.vit import ViTFeatureExtractor, ViTFeatureExtractorConfig

__all__ = [
    "BaseFeatureExtractor",
    "BaseFeatureExtractorConfig",
    "CNNFeatureExtractor",
    "CNNFeatureExtractorConfig",
    "ConcatFeatureExtractor",
    "ConcatFeatureExtractorConfig",
    "CRadioV3BFeatureExtractorConfig",
    "CRadioV4HFeatureExtractorConfig",
    "DinoFeatureExtractor",
    "DinoFeatureExtractorConfig",
    "DinoViTFeatureExtractor",
    "DinoViTFeatureExtractorConfig",
    "IdentityFeatureExtractor",
    "IdentityFeatureExtractorConfig",
    "NoneFeatureExtractor",
    "NoneFeatureExtractorConfig",
    "RadioFeatureExtractor",
    "RadioFeatureExtractorConfig",
    "ResNetFeatureExtractor",
    "ResNetFeatureExtractorConfig",
    "ViTFeatureExtractor",
    "ViTFeatureExtractorConfig",
]
