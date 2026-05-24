"""Tests for EDM Heun sampling on the BoxSetDiffusionDecoder (I5)."""

from __future__ import annotations

import torch

from src.decoders.diffusion.box_set import BoxSetDiffusionDecoder


def _make_decoder(num_slots: int = 4, dim: int = 32, image_token_dim: int = 16) -> BoxSetDiffusionDecoder:
    return BoxSetDiffusionDecoder(
        num_slots=num_slots,
        dim=dim,
        num_heads=4,
        num_blocks=2,
        time_dim=dim,
        image_token_dim=image_token_dim,
        fourier_box_bands=4,
        fourier_sigma_bands=4,
    )


def _make_inputs(B: int = 2, T: int = 9, image_token_dim: int = 16):
    image_tokens = torch.randn(B, T, image_token_dim)
    # 3x3 grid centres in [-1, 1].
    side = int(round(T ** 0.5))
    coords = torch.linspace(-1.0, 1.0, side)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    patch_centers = torch.stack([xx, yy], dim=-1).reshape(T, 2)
    return image_tokens, patch_centers


# ---------------------------------------------------------------------------
# edm_sigma_schedule
# ---------------------------------------------------------------------------


def test_edm_sigma_schedule_shape_and_endpoints():
    s = BoxSetDiffusionDecoder.edm_sigma_schedule(
        num_steps=8, sigma_min=0.01, sigma_max=1.0,
    )
    assert s.shape == (9,)
    assert torch.isclose(s[0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(s[-2], torch.tensor(0.01), atol=1e-5)
    assert s[-1].item() == 0.0


def test_edm_sigma_schedule_strictly_decreasing():
    s = BoxSetDiffusionDecoder.edm_sigma_schedule(num_steps=16)
    diffs = s[1:] - s[:-1]
    assert (diffs <= 0).all()


def test_edm_sigma_schedule_single_step():
    s = BoxSetDiffusionDecoder.edm_sigma_schedule(num_steps=1, sigma_max=1.0)
    assert s.shape == (2,)
    assert torch.isclose(s[0], torch.tensor(1.0), atol=1e-5)
    assert s[-1].item() == 0.0


# ---------------------------------------------------------------------------
# .sample()
# ---------------------------------------------------------------------------


def test_sample_single_step_shape_and_finite():
    torch.manual_seed(0)
    dec = _make_decoder()
    image_tokens, patch_centers = _make_inputs()
    out = dec.sample(
        image_tokens=image_tokens, patch_centers=patch_centers,
        num_inference_steps=1,
    )
    assert out["pred_boxes"].shape == (2, 4, 4)
    assert out["obj_logits"].shape == (2, 4)
    assert torch.isfinite(out["pred_boxes"]).all()
    assert torch.isfinite(out["obj_logits"]).all()


def test_sample_multi_step_heun_shape_and_finite():
    torch.manual_seed(0)
    dec = _make_decoder()
    image_tokens, patch_centers = _make_inputs()
    out = dec.sample(
        image_tokens=image_tokens, patch_centers=patch_centers,
        num_inference_steps=4,
    )
    assert out["pred_boxes"].shape == (2, 4, 4)
    assert out["obj_logits"].shape == (2, 4)
    assert torch.isfinite(out["pred_boxes"]).all()
    assert torch.isfinite(out["obj_logits"]).all()


def test_sample_is_deterministic_with_same_generator():
    torch.manual_seed(0)
    dec = _make_decoder()
    image_tokens, patch_centers = _make_inputs()

    g1 = torch.Generator(device="cpu").manual_seed(123)
    g2 = torch.Generator(device="cpu").manual_seed(123)
    out1 = dec.sample(
        image_tokens=image_tokens, patch_centers=patch_centers,
        num_inference_steps=3, generator=g1,
    )
    out2 = dec.sample(
        image_tokens=image_tokens, patch_centers=patch_centers,
        num_inference_steps=3, generator=g2,
    )
    assert torch.allclose(out1["pred_boxes"], out2["pred_boxes"])
    assert torch.allclose(out1["obj_logits"], out2["obj_logits"])


def test_sample_accepts_x_init():
    torch.manual_seed(0)
    dec = _make_decoder()
    image_tokens, patch_centers = _make_inputs()
    x_init = torch.zeros(2, 4, 4)
    out = dec.sample(
        image_tokens=image_tokens, patch_centers=patch_centers,
        num_inference_steps=2, x_init=x_init,
    )
    assert out["pred_boxes"].shape == (2, 4, 4)


def test_sample_zero_steps_raises():
    dec = _make_decoder()
    image_tokens, patch_centers = _make_inputs()
    try:
        dec.sample(
            image_tokens=image_tokens, patch_centers=patch_centers,
            num_inference_steps=0,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for num_inference_steps=0")


def test_sample_runs_under_no_grad():
    """Ensure forward pass on outputs does not require grad tracking."""
    dec = _make_decoder()
    image_tokens, patch_centers = _make_inputs()
    out = dec.sample(
        image_tokens=image_tokens, patch_centers=patch_centers,
        num_inference_steps=2,
    )
    assert not out["pred_boxes"].requires_grad
