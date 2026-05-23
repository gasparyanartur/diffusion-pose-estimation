"""Tests for ResNetFeatureExtractor."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.resnet import ResNetFeatureExtractor, ResNetFeatureExtractorConfig, _OUT_CHANNELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image(B: int = 1, H: int = 256, W: int = 192) -> torch.Tensor:
    return torch.rand(B, 3, H, W)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("output_layer,expected_ch,expected_h,expected_w", [
    ("layer3", 256, 16, 12),  # stride 16 on 256×192
    ("layer4", 512, 8,  6),   # stride 32 on 256×192
])
def test_resnet18_output_shape(output_layer, expected_ch, expected_h, expected_w):
    m = ResNetFeatureExtractor(variant="resnet18", output_layer=output_layer, pretrained=False)
    m.eval()
    with torch.no_grad():
        out = m(image=_image())
    assert "spatial" in out
    feat = out["spatial"]
    assert feat.shape == (1, expected_ch, expected_h, expected_w), feat.shape


def test_out_channels_matches_output():
    for variant in ("resnet18", "resnet34", "resnet50"):
        for layer in ("layer3", "layer4"):
            m = ResNetFeatureExtractor(variant=variant, output_layer=layer, pretrained=False)
            m.eval()
            with torch.no_grad():
                feat = m(image=_image())["spatial"]
            assert m.out_channels == feat.shape[1], f"{variant}/{layer}: {m.out_channels} != {feat.shape[1]}"


def test_output_key_is_spatial():
    m = ResNetFeatureExtractor(pretrained=False)
    out = m(image=_image())
    assert list(out.keys()) == ["spatial"]


def test_requires_image_not_none():
    m = ResNetFeatureExtractor(pretrained=False)
    with pytest.raises(ValueError, match="requires `image`"):
        m(image=None)


def test_batch_size_gt1():
    m = ResNetFeatureExtractor(pretrained=False)
    m.eval()
    with torch.no_grad():
        out = m(image=_image(B=4))
    assert out["spatial"].shape[0] == 4


# ---------------------------------------------------------------------------
# Config / factory
# ---------------------------------------------------------------------------

def test_config_build():
    cfg = ResNetFeatureExtractorConfig(variant="resnet18", output_layer="layer3", pretrained=False)
    m = cfg.build()
    assert isinstance(m, ResNetFeatureExtractor)
    assert m.out_channels == 256


def test_config_out_channels_table():
    for variant, layers in _OUT_CHANNELS.items():
        for layer, expected in layers.items():
            cfg = ResNetFeatureExtractorConfig(variant=variant, output_layer=layer, pretrained=False)
            m = cfg.build()
            assert m.out_channels == expected
