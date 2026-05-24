"""Tests for BoxSetDiffusionDecoder forward path (I3)."""

from __future__ import annotations

import pytest
import torch

from src.decoders.diffusion.box_set import BoxSetDiffusionDecoder


def _make_decoder(**overrides) -> BoxSetDiffusionDecoder:
    defaults = dict(
        num_slots=8,
        dim=32,
        num_heads=4,
        num_blocks=2,
        time_dim=24,
        ffn_mult=2.0,
        image_token_dim=16,
        fourier_box_bands=4,
        fourier_sigma_bands=4,
    )
    defaults.update(overrides)
    return BoxSetDiffusionDecoder(**defaults)


def _make_inputs(B=2, N=8, T=9, C=16):
    noisy_boxes = torch.randn(B, N, 4) * 0.3
    sigma = torch.rand(B) * 0.5 + 0.01
    image_tokens = torch.randn(B, T, C)
    # T = 3x3 patch grid mapped to (-1, 1)
    ys = (torch.arange(3, dtype=torch.float32) + 0.5) * (2.0 / 3) - 1.0
    xs = (torch.arange(3, dtype=torch.float32) + 0.5) * (2.0 / 3) - 1.0
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    patch_centers = torch.stack([gy.flatten(), gx.flatten()], dim=-1)  # [T=9, 2]
    return noisy_boxes, sigma, image_tokens, patch_centers


# ---------------------------------------------------------------------------
# Forward path
# ---------------------------------------------------------------------------

def test_forward_output_shapes():
    decoder = _make_decoder()
    boxes, sigma, tokens, centers = _make_inputs()
    out = decoder(boxes, sigma, tokens, centers)
    assert set(out.keys()) == {"pred_boxes", "obj_logits"}
    assert out["pred_boxes"].shape == boxes.shape
    assert out["obj_logits"].shape == boxes.shape[:2]


def test_forward_outputs_finite():
    decoder = _make_decoder()
    boxes, sigma, tokens, centers = _make_inputs()
    out = decoder(boxes, sigma, tokens, centers)
    assert torch.isfinite(out["pred_boxes"]).all()
    assert torch.isfinite(out["obj_logits"]).all()


def test_forward_handles_extreme_sigma():
    """Tiny σ (clamped via sigma_log_offset) should not produce NaNs."""
    decoder = _make_decoder()
    boxes, _, tokens, centers = _make_inputs()
    sigma = torch.tensor([1e-8, 1.0])
    out = decoder(boxes, sigma, tokens, centers)
    assert torch.isfinite(out["pred_boxes"]).all()
    assert torch.isfinite(out["obj_logits"]).all()


def test_forward_supports_image_key_padding_mask():
    decoder = _make_decoder()
    boxes, sigma, tokens, centers = _make_inputs()
    T = tokens.shape[1]
    mask = torch.zeros(boxes.shape[0], T, dtype=torch.bool)
    mask[:, T // 2 :] = True  # mask the tail half
    out = decoder(boxes, sigma, tokens, centers, image_key_padding_mask=mask)
    assert torch.isfinite(out["pred_boxes"]).all()


def test_forward_backward_flows():
    decoder = _make_decoder()
    boxes, sigma, tokens, centers = _make_inputs()
    boxes = boxes.detach().requires_grad_(True)
    tokens = tokens.detach().requires_grad_(True)
    out = decoder(boxes, sigma, tokens, centers)
    loss = out["pred_boxes"].pow(2).sum() + out["obj_logits"].pow(2).sum()
    loss.backward()
    assert boxes.grad is not None and torch.isfinite(boxes.grad).all()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    # at least some decoder parameters received a gradient
    grads = [p.grad for p in decoder.parameters() if p.grad is not None]
    assert any((g.abs().sum() > 0).item() for g in grads)


def test_forward_rejects_wrong_num_slots():
    decoder = _make_decoder(num_slots=8)
    boxes = torch.randn(1, 4, 4)
    sigma = torch.tensor([0.1])
    tokens = torch.randn(1, 9, 16)
    centers = torch.zeros(9, 2)
    with pytest.raises(ValueError):
        decoder(boxes, sigma, tokens, centers)


def test_init_rejects_invalid_dim():
    with pytest.raises(ValueError):
        BoxSetDiffusionDecoder(dim=30, image_token_dim=16)  # not div by 4


def test_different_sigma_changes_output():
    """Once trained, different σ would route via FiLM, but even at init the
    sinusoidal-time embedding should perturb activations through the second
    cross-attn / refine path. We simulate "trained" by perturbing FiLM scales."""
    torch.manual_seed(0)
    decoder = _make_decoder()
    with torch.no_grad():
        # perturb the FiLM linear in every TimeFiLMFFN so σ has an effect
        for m in decoder.modules():
            from src.training.diffusion_modules import TimeFiLMFFN
            if isinstance(m, TimeFiLMFFN) and m.use_film:
                m.film.weight.normal_(std=0.3)
                m.film.bias.normal_(std=0.3)

    boxes, _, tokens, centers = _make_inputs()
    out_low = decoder(boxes, torch.full((boxes.shape[0],), 1e-3), tokens, centers)
    out_high = decoder(boxes, torch.full((boxes.shape[0],), 0.5), tokens, centers)
    assert not torch.allclose(out_low["pred_boxes"], out_high["pred_boxes"])
