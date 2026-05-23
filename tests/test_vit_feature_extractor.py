"""Tests for ViTFeatureExtractor."""

from __future__ import annotations

import torch

from src.features.vit import ViTFeatureExtractor


def test_output_shape():
    extractor = ViTFeatureExtractor(pretrained=False)
    image = torch.rand(2, 3, 224, 224)
    out = extractor(image=image)
    assert out["spatial"].shape == (2, 768, 14, 14)


def test_output_shape_non_224():
    """Non-224 inputs are resized internally; output is always 14\u00d714."""
    extractor = ViTFeatureExtractor(pretrained=False)
    image = torch.rand(2, 3, 128, 192)
    out = extractor(image=image)
    assert out["spatial"].shape == (2, 768, 14, 14)
