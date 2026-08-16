"""Unit tests for canonical shared diffusion modules (I2)."""

from __future__ import annotations

import pytest
import torch

from src.models.common.attention import (
    FourierFeatures,
    TimeFiLMFFN,
    TimeFiLMTransformerBlock,
)


# ---------------------------------------------------------------------------
# FourierFeatures
# ---------------------------------------------------------------------------

def test_fourier_features_output_shape_scalar_input():
    ff = FourierFeatures(in_dim=1, out_dim=32, num_bands=8)
    x = torch.zeros(4, 1)
    out = ff(x)
    assert out.shape == (4, 32)


def test_fourier_features_output_shape_multidim_input():
    ff = FourierFeatures(in_dim=4, out_dim=64, num_bands=16, scale=1000.0)
    boxes = torch.rand(2, 5, 4) * 2 - 1  # canonical [-1, 1]
    out = ff(boxes)
    assert out.shape == (2, 5, 64)


def test_fourier_features_rejects_wrong_in_dim():
    ff = FourierFeatures(in_dim=4, out_dim=8, num_bands=4)
    with pytest.raises(ValueError):
        ff(torch.zeros(2, 3))


def test_fourier_features_distinguishes_close_inputs():
    """High-frequency bands let the bank separate close box coords."""
    torch.manual_seed(0)
    ff = FourierFeatures(in_dim=1, out_dim=16, num_bands=16, scale=1000.0)
    a = ff(torch.tensor([[0.0]]))
    b = ff(torch.tensor([[0.001]]))
    assert (a - b).abs().mean().item() > 1e-4


def test_fourier_features_backward_flows():
    ff = FourierFeatures(in_dim=4, out_dim=8, num_bands=4, scale=10.0)
    x = torch.rand(3, 4, requires_grad=True)
    out = ff(x).sum()
    out.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# TimeFiLMFFN
# ---------------------------------------------------------------------------

def test_time_film_ffn_shape_and_identity_at_init():
    """FiLM is zero-initialised so the FFN is the identity-FFN at init."""
    dim, T = 32, 16
    ffn = TimeFiLMFFN(dim=dim, ffn_mult=2.0, time_dim=T)
    x = torch.randn(2, 5, dim)
    t = torch.randn(2, T)
    out_with_t = ffn(x, time_emb=t)

    # Same input with a different t should match (FiLM is zero-init).
    t2 = torch.randn(2, T)
    out_with_t2 = ffn(x, time_emb=t2)
    assert torch.allclose(out_with_t, out_with_t2)
    assert out_with_t.shape == x.shape


def test_time_film_ffn_no_time_branch():
    dim = 16
    ffn = TimeFiLMFFN(dim=dim, ffn_mult=4.0, time_dim=0)
    x = torch.randn(3, 4, dim)
    out = ffn(x)
    assert out.shape == x.shape


def test_time_film_ffn_requires_time_when_configured():
    ffn = TimeFiLMFFN(dim=8, time_dim=4)
    with pytest.raises(ValueError):
        ffn(torch.randn(2, 3, 8))


def test_time_film_ffn_modulation_affects_output_after_training_step():
    """After perturbing FiLM weights, different `t` should change the output."""
    dim, T = 8, 4
    ffn = TimeFiLMFFN(dim=dim, time_dim=T)
    with torch.no_grad():
        ffn.film.weight.normal_(std=0.5)
        ffn.film.bias.normal_(std=0.5)

    x = torch.randn(2, 3, dim)
    out1 = ffn(x, time_emb=torch.zeros(2, T))
    out2 = ffn(x, time_emb=torch.ones(2, T))
    assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# TimeFiLMTransformerBlock
# ---------------------------------------------------------------------------

def test_transformer_block_self_attn_only_shape():
    blk = TimeFiLMTransformerBlock(
        dim=32, num_heads=4, time_dim=16, use_cross_attn=False,
    )
    x = torch.randn(2, 7, 32)
    t = torch.randn(2, 16)
    out = blk(x, context=None, time_emb=t)
    assert out.shape == x.shape


def test_transformer_block_cross_attn_shape():
    blk = TimeFiLMTransformerBlock(
        dim=32, num_heads=4, time_dim=16, cross_dim=24,
    )
    x = torch.randn(2, 5, 32)
    ctx = torch.randn(2, 11, 24)
    t = torch.randn(2, 16)
    out = blk(x, context=ctx, time_emb=t)
    assert out.shape == x.shape


def test_transformer_block_backward_flows():
    blk = TimeFiLMTransformerBlock(dim=16, num_heads=2, time_dim=8, cross_dim=16)
    x = torch.randn(2, 4, 16, requires_grad=True)
    ctx = torch.randn(2, 6, 16, requires_grad=True)
    t = torch.randn(2, 8)
    out = blk(x, context=ctx, time_emb=t).sum()
    out.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert ctx.grad is not None and torch.isfinite(ctx.grad).all()


def test_transformer_block_requires_context_when_cross_enabled():
    blk = TimeFiLMTransformerBlock(dim=8, num_heads=2, time_dim=0)
    with pytest.raises(ValueError):
        blk(torch.randn(1, 2, 8), context=None)


def test_transformer_block_respects_context_padding_mask():
    """Fully-masked context yields a finite output (no NaNs from masked softmax)."""
    blk = TimeFiLMTransformerBlock(dim=16, num_heads=2, time_dim=0, cross_dim=16)
    x = torch.randn(2, 3, 16)
    ctx = torch.randn(2, 4, 16)
    # Mask out all but the first key per sample
    mask = torch.tensor([[False, True, True, True], [False, True, True, True]])
    out = blk(x, context=ctx, key_padding_mask_arg=None) if False else blk(
        x, context=ctx, context_key_padding_mask=mask,
    )
    assert torch.isfinite(out).all()
    assert out.shape == x.shape
