"""Tests for ViTFeatureExtractor."""

from __future__ import annotations

import torch

from src.features.vit import ViTFeatureExtractor


def test_output_shape():
    extractor = ViTFeatureExtractor(pretrained=False)
    image = torch.rand(2, 3, 224, 224)
    out = extractor(image=image)
    assert out["spatial"].shape == (2, 768, 14, 14)
