"""Tests for DinoFeatureExtractor (ConvNeXt) and DinoViTFeatureExtractor (ViT).

Both models are mocked so no network access or large checkpoints are needed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.features.dino import (
    DinoFeatureExtractor,
    DinoFeatureExtractorConfig,
    DinoViTFeatureExtractor,
    DinoViTFeatureExtractorConfig,
    _MODEL_NAME,
    _STAGE_OUT_CHANNELS,
    _STAGE_STRIDE,
    _VIT_S16_MODEL_NAME,
    _VIT_S16_OUT_CHANNELS,
    _VIT_S16_PATCH_SIZE,
)


# ---------------------------------------------------------------------------
# Fake backbone helpers
# ---------------------------------------------------------------------------

def _make_convnext_model(stage_channels: dict[int, int] | None = None):
    """Return a minimal mock that mimics ConvNeXt HuggingFace output."""
    if stage_channels is None:
        stage_channels = _STAGE_OUT_CHANNELS

    class _FakeConvNeXt(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MagicMock()

        def forward(self, pixel_values, **kwargs):
            B, _, H, W = pixel_values.shape
            hidden_states = [None]  # index 0 unused
            for stage, (ch, stride) in enumerate(
                zip(stage_channels.values(), _STAGE_STRIDE.values()), start=1
            ):
                hidden_states.append(torch.zeros(B, ch, H // stride, W // stride))
            out = MagicMock()
            out.hidden_states = hidden_states
            return out

    model = _FakeConvNeXt()
    model.parameters = lambda: iter([nn.Parameter(torch.zeros(1))])
    return model


def _make_vit_model(hidden_dim: int = _VIT_S16_OUT_CHANNELS, patch_size: int = _VIT_S16_PATCH_SIZE, n_registers: int = 0):
    """Return a minimal mock that mimics ViT HuggingFace output."""

    class _FakeViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MagicMock()
            self.config.num_register_tokens = n_registers

        def forward(self, pixel_values, **kwargs):
            B, _, H, W = pixel_values.shape
            G_H = H // patch_size
            G_W = W // patch_size
            n_tokens = 1 + n_registers + G_H * G_W
            out = MagicMock()
            out.last_hidden_state = torch.zeros(B, n_tokens, hidden_dim)
            return out

    model = _FakeViT()
    model.parameters = lambda: iter([nn.Parameter(torch.zeros(1))])
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def convnext_extractor():
    fake = _make_convnext_model()
    with patch("transformers.AutoModel.from_pretrained", return_value=fake):
        yield DinoFeatureExtractor(model_name=_MODEL_NAME, stage=4, frozen=False)


@pytest.fixture
def vit_extractor():
    fake = _make_vit_model()
    with patch("transformers.AutoModel.from_pretrained", return_value=fake):
        yield DinoViTFeatureExtractor(
            model_name=_VIT_S16_MODEL_NAME,
            patch_size=_VIT_S16_PATCH_SIZE,
            out_channels=_VIT_S16_OUT_CHANNELS,
            frozen=False,
        )


def _image(B: int = 1, H: int = 128, W: int = 128) -> torch.Tensor:
    return torch.rand(B, 3, H, W)


# ---------------------------------------------------------------------------
# DinoFeatureExtractor (ConvNeXt) tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_convnext_output_shape(stage):
    fake = _make_convnext_model()
    with patch("transformers.AutoModel.from_pretrained", return_value=fake):
        m = DinoFeatureExtractor(stage=stage, frozen=False)
    m.eval()
    H, W = 128, 128
    with torch.no_grad():
        out = m(image=_image(H=H, W=W))
    assert "spatial" in out
    stride = _STAGE_STRIDE[stage]
    expected_ch = _STAGE_OUT_CHANNELS[stage]
    assert out["spatial"].shape == (1, expected_ch, H // stride, W // stride)


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_convnext_out_channels_matches_output(stage):
    fake = _make_convnext_model()
    with patch("transformers.AutoModel.from_pretrained", return_value=fake):
        m = DinoFeatureExtractor(stage=stage, frozen=False)
    m.eval()
    with torch.no_grad():
        out = m(image=_image())
    assert m.out_channels == out["spatial"].shape[1]


def test_convnext_get_output_size():
    fake = _make_convnext_model()
    with patch("transformers.AutoModel.from_pretrained", return_value=fake):
        m = DinoFeatureExtractor(stage=4, frozen=False)
    assert m.get_output_size(128) == 128 // _STAGE_STRIDE[4]


def test_convnext_requires_image(convnext_extractor):
    with pytest.raises(AssertionError):
        convnext_extractor(image=None)


def test_convnext_batch(convnext_extractor):
    convnext_extractor.eval()
    with torch.no_grad():
        out = convnext_extractor(image=_image(B=4))
    assert out["spatial"].shape[0] == 4


def test_convnext_invalid_stage():
    with pytest.raises(ValueError, match="stage must be 1-4"):
        with patch("transformers.AutoModel.from_pretrained", return_value=_make_convnext_model()):
            DinoFeatureExtractor(stage=0, frozen=False)


def test_convnext_config_build():
    fake = _make_convnext_model()
    with patch("transformers.AutoModel.from_pretrained", return_value=fake):
        cfg = DinoFeatureExtractorConfig(model_name=_MODEL_NAME, stage=4, frozen=False)
        m = cfg.build()
    assert isinstance(m, DinoFeatureExtractor)
    assert m.out_channels == _STAGE_OUT_CHANNELS[4]


# ---------------------------------------------------------------------------
# DinoViTFeatureExtractor tests
# ---------------------------------------------------------------------------

def test_vit_output_shape(vit_extractor):
    vit_extractor.eval()
    H, W = 128, 128
    G = H // _VIT_S16_PATCH_SIZE
    with torch.no_grad():
        out = vit_extractor(image=_image(H=H, W=W))
    assert "spatial" in out
    assert out["spatial"].shape == (1, _VIT_S16_OUT_CHANNELS, G, G)


def test_vit_out_channels_matches_output(vit_extractor):
    vit_extractor.eval()
    with torch.no_grad():
        out = vit_extractor(image=_image())
    assert vit_extractor.out_channels == out["spatial"].shape[1]


def test_vit_get_output_size(vit_extractor):
    assert vit_extractor.get_output_size(128) == 128 // _VIT_S16_PATCH_SIZE


def test_vit_requires_image(vit_extractor):
    with pytest.raises(AssertionError):
        vit_extractor(image=None)


def test_vit_batch():
    fake = _make_vit_model()
    with patch("transformers.AutoModel.from_pretrained", return_value=fake):
        m = DinoViTFeatureExtractor(frozen=False)
    m.eval()
    with torch.no_grad():
        out = m(image=_image(B=4))
    assert out["spatial"].shape[0] == 4


def test_vit_register_tokens_skipped():
    """Extra register tokens should be dropped; output grid size should be unchanged."""
    n_registers = 4
    fake = _make_vit_model(n_registers=n_registers)
    with patch("transformers.AutoModel.from_pretrained", return_value=fake):
        m = DinoViTFeatureExtractor(frozen=False)
    m.eval()
    H, W = 128, 128
    G = H // _VIT_S16_PATCH_SIZE
    with torch.no_grad():
        out = m(image=_image(H=H, W=W))
    assert out["spatial"].shape == (1, _VIT_S16_OUT_CHANNELS, G, G)


def test_vit_non_square_image():
    """Non-square inputs must produce a non-square spatial grid (not crash)."""
    fake = _make_vit_model()
    with patch("transformers.AutoModel.from_pretrained", return_value=fake):
        m = DinoViTFeatureExtractor(frozen=False)
    m.eval()
    H, W = 192, 256  # 12 × 16 patches
    G_H = H // _VIT_S16_PATCH_SIZE
    G_W = W // _VIT_S16_PATCH_SIZE
    with torch.no_grad():
        out = m(image=_image(H=H, W=W))
    assert out["spatial"].shape == (1, _VIT_S16_OUT_CHANNELS, G_H, G_W)


def test_vit_config_build():
    fake = _make_vit_model()
    with patch("transformers.AutoModel.from_pretrained", return_value=fake):
        cfg = DinoViTFeatureExtractorConfig(model_name=_VIT_S16_MODEL_NAME, frozen=False)
        m = cfg.build()
    assert isinstance(m, DinoViTFeatureExtractor)
    assert m.out_channels == _VIT_S16_OUT_CHANNELS
