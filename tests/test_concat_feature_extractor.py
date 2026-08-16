"""Tests for ConcatFeatureExtractor and ConcatFeatureExtractorConfig."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.features.cnn import CNNFeatureExtractor, CNNFeatureExtractorConfig
from src.models.features.concat import ConcatFeatureExtractor, ConcatFeatureExtractorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_cnn_extractor(dim: int = 32) -> ConcatFeatureExtractor:
    b1 = CNNFeatureExtractor(out_channels=64)
    b2 = CNNFeatureExtractor(out_channels=32)
    return ConcatFeatureExtractor(backbones=[b1, b2], dim=dim)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_output_shape_two_backbones():
    extractor = _two_cnn_extractor(dim=32)
    image = torch.rand(2, 3, 64, 48)
    out = extractor(image=image)
    assert "spatial" in out
    B, T, D = out["spatial"].shape
    assert B == 2
    assert D == 32
    # CNN stride=16 → 64//16=4 and 48//16=3 → 4*3=12 tokens each, 12+12=24 total
    assert T == 24


def test_output_shape_single_backbone():
    b = CNNFeatureExtractor(out_channels=16)
    extractor = ConcatFeatureExtractor(backbones=[b], dim=8)
    image = torch.rand(1, 3, 32, 32)
    out = extractor(image=image)
    B, T, D = out["spatial"].shape
    assert B == 1
    assert D == 8
    # 32//16=2 → 2*2=4 tokens
    assert T == 4


# ---------------------------------------------------------------------------
# Output equals manual concatenation
# ---------------------------------------------------------------------------


def test_output_equals_manual_concat():
    """ConcatFeatureExtractor output must equal manually applying each step."""
    torch.manual_seed(0)
    b1 = CNNFeatureExtractor(out_channels=64)
    b2 = CNNFeatureExtractor(out_channels=32)
    extractor = ConcatFeatureExtractor(backbones=[b1, b2], dim=16)
    extractor.eval()

    image = torch.rand(2, 3, 64, 48)

    with torch.no_grad():
        result = extractor(image=image)["spatial"]  # [B, T, dim]

        # Manual backbone 0
        feat0 = b1(image=image)["spatial"]          # [B, 64, H, W]
        t0 = feat0.flatten(2).permute(0, 2, 1)      # [B, H*W, 64]
        t0 = extractor.norms[0](t0)
        t0 = extractor.projs[0](t0)
        t0 = t0 + extractor.modality_embeds[0]

        # Manual backbone 1
        feat1 = b2(image=image)["spatial"]
        t1 = feat1.flatten(2).permute(0, 2, 1)
        t1 = extractor.norms[1](t1)
        t1 = extractor.projs[1](t1)
        t1 = t1 + extractor.modality_embeds[1]

        expected = torch.cat([t0, t1], dim=1)

    assert torch.allclose(result, expected), "tokens do not match manual concatenation"


# ---------------------------------------------------------------------------
# Learned modality embeddings
# ---------------------------------------------------------------------------


def test_modality_embeds_are_learned():
    extractor = _two_cnn_extractor()
    for i, embed in enumerate(extractor.modality_embeds):
        assert embed.requires_grad, f"modality_embeds[{i}] should require grad"


def test_modality_embeds_shape():
    extractor = _two_cnn_extractor(dim=16)
    for embed in extractor.modality_embeds:
        assert embed.shape == (1, 1, 16)


# ---------------------------------------------------------------------------
# get_output_size
# ---------------------------------------------------------------------------


def test_get_output_size_is_sum_of_children():
    b1 = CNNFeatureExtractor(out_channels=64)   # stride 16
    b2 = CNNFeatureExtractor(out_channels=32)   # stride 16
    extractor = ConcatFeatureExtractor(backbones=[b1, b2], dim=8)
    n = 64
    expected = b1.get_output_size(n) + b2.get_output_size(n)
    assert extractor.get_output_size(n) == expected


# ---------------------------------------------------------------------------
# out_channels equals dim
# ---------------------------------------------------------------------------


def test_out_channels_equals_dim():
    extractor = _two_cnn_extractor(dim=48)
    assert extractor.out_channels == 48


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_empty_backbones_raises():
    with pytest.raises(ValueError, match="at least one"):
        ConcatFeatureExtractor(backbones=[], dim=32)


def test_config_build_wrong_type_raises():
    cfg = ConcatFeatureExtractorConfig(backbones=["not_a_config"], dim=16)
    with pytest.raises(TypeError, match="BaseFeatureExtractorConfig"):
        cfg.build()


# ---------------------------------------------------------------------------
# Config build round-trip
# ---------------------------------------------------------------------------


def test_config_builds_extractor():
    cfg = ConcatFeatureExtractorConfig(
        backbones=[
            CNNFeatureExtractorConfig(out_channels=64),
            CNNFeatureExtractorConfig(out_channels=32),
        ],
        dim=16,
    )
    extractor = cfg.build()
    assert isinstance(extractor, ConcatFeatureExtractor)
    assert extractor.out_channels == 16
    assert len(extractor.backbones) == 2


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_to_projections():
    extractor = _two_cnn_extractor(dim=16)
    image = torch.rand(2, 3, 32, 32)
    out = extractor(image=image)
    out["spatial"].sum().backward()
    for i in range(2):
        for name, p in [
            (f"projs[{i}].weight", extractor.projs[i].weight),
            (f"modality_embeds[{i}]", extractor.modality_embeds[i]),
        ]:
            assert p.grad is not None, f"No gradient for {name}"
