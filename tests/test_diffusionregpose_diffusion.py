from __future__ import annotations

import math

import torch

from src.models.pipelines.pose_estimators.diffusionregpose.diffusion import (
    DiffusionRegPoseDiffusion,
    cosine_beta_schedule,
)


def _reference_cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], 0, 0.999)


def test_cosine_schedule_matches_diffusionregpose_reference() -> None:
    torch.testing.assert_close(
        cosine_beta_schedule(17),
        _reference_cosine_beta_schedule(17),
        rtol=0,
        atol=0,
    )


def test_diffusion_state_registers_reference_buffers() -> None:
    diffusion = DiffusionRegPoseDiffusion(timesteps=17, sampling_timesteps=5, scale=6.0)

    assert diffusion.num_timesteps == 17
    assert diffusion.sampling_timesteps == 5
    assert diffusion.is_ddim_sampling
    assert diffusion.scale == 6.0
    assert set(diffusion.state_dict()) == {
        "betas",
        "alphas_cumprod",
        "alphas_cumprod_prev",
        "sqrt_alphas_cumprod",
        "sqrt_one_minus_alphas_cumprod",
        "log_one_minus_alphas_cumprod",
        "sqrt_recip_alphas_cumprod",
        "sqrt_recipm1_alphas_cumprod",
        "posterior_variance",
        "posterior_log_variance_clipped",
        "posterior_mean_coef1",
        "posterior_mean_coef2",
    }


def test_q_sample_and_noise_conversion_are_consistent() -> None:
    diffusion = DiffusionRegPoseDiffusion(timesteps=32)
    x_start = torch.randn(2, 3, 28)
    noise = torch.randn_like(x_start)
    timesteps = torch.tensor([0, 13], dtype=torch.long)

    noisy = diffusion.q_sample(x_start, timesteps, noise)
    recovered_noise = diffusion.predict_noise_from_start(noisy, timesteps, x_start)

    assert recovered_noise.dtype == torch.float64
    torch.testing.assert_close(recovered_noise.float(), noise, rtol=1e-5, atol=1e-5)


def test_ddim_timestep_pairs_match_reference_order() -> None:
    diffusion = DiffusionRegPoseDiffusion(timesteps=1000, sampling_timesteps=4)

    assert diffusion.ddim_timestep_pairs() == [(999, 749), (749, 499), (499, 249), (249, -1)]


def test_ddim_step_matches_reference_update_and_terminal_state() -> None:
    diffusion = DiffusionRegPoseDiffusion(timesteps=32, sampling_timesteps=4, ddim_sampling_eta=0.0)
    x_start = torch.randn(2, 3, 28)
    pred_noise = torch.randn_like(x_start)
    noise = torch.randn_like(x_start)

    updated = diffusion.ddim_step(x_start, pred_noise, 20, 10, noise)
    alpha_next = diffusion.alphas_cumprod[10]
    expected = alpha_next.sqrt() * x_start + (1.0 - alpha_next).sqrt() * pred_noise

    torch.testing.assert_close(updated, expected)
    torch.testing.assert_close(diffusion.ddim_step(x_start, pred_noise, 20, -1, noise), x_start)
