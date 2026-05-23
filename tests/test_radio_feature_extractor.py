"""Tests for RadioFeatureExtractor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from src.features.radio import RadioFeatureExtractor


def _make_extractor(out_channels: int = 64, patch_size: int = 16) -> RadioFeatureExtractor:
    """Build a RadioFeatureExtractor backed by a lightweight fake model."""

    def fake_forward(x):
        B = x.shape[0]
        T = (x.shape[2] // patch_size) * (x.shape[3] // patch_size)
        summary = torch.zeros(B, out_channels)
        spatial = torch.rand(B, T, out_channels)
        return summary, spatial

    fake_model = MagicMock()
    fake_model.side_effect = fake_forward
    fake_model.parameters = lambda: iter([torch.nn.Parameter(torch.zeros(1))])

    with patch("transformers.AutoModel.from_pretrained", return_value=fake_model):
        return RadioFeatureExtractor(model_name="fake/model", patch_size=patch_size)


def test_output_shape():
    extractor = _make_extractor(out_channels=64, patch_size=16)
    image = torch.rand(2, 3, 128, 96)
    out = extractor(image=image)
    assert out["spatial"].shape == (2, 64, 8, 6)


def test_raw_image_passed_to_model():
    """Extractor must pass raw pixels to the model (it normalises internally)."""
    received: list[torch.Tensor] = []

    def fake_forward(x):
        received.append(x.clone())
        B, _, H, W = x.shape
        return torch.zeros(B, 64), torch.rand(B, (H // 16) * (W // 16), 64)

    fake_model = MagicMock()
    fake_model.side_effect = fake_forward
    fake_model.parameters = lambda: iter([torch.nn.Parameter(torch.zeros(1))])

    with patch("transformers.AutoModel.from_pretrained", return_value=fake_model):
        extractor = RadioFeatureExtractor(model_name="fake/model")

    image = torch.rand(1, 3, 224, 224) * 0.9 + 0.05  # pixels in (0.05, 0.95)
    extractor(image=image)

    assert torch.allclose(received[-1], image), "model did not receive raw pixels"
