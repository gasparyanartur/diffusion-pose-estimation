"""Discrete diffusion state and sampling helpers for pose proposals."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    """Build the cosine variance schedule for the forward diffusion process."""
    if timesteps < 1:
        raise ValueError(f"timesteps must be positive; got {timesteps}")
    steps = timesteps + 1
    positions = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((positions / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], 0, 0.999)


def extract(values: Tensor, timesteps: Tensor, target_shape: torch.Size) -> Tensor:
    """Select one schedule value per batch item and broadcast it to a tensor."""
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must have shape [B]; got {tuple(timesteps.shape)}")
    if timesteps.dtype != torch.long:
        raise TypeError(f"timesteps must be torch.long; got {timesteps.dtype}")
    if timesteps.numel() != target_shape[0]:
        raise ValueError(
            f"timesteps batch ({timesteps.numel()}) does not match target ({target_shape[0]})"
        )
    selected = values.to(timesteps.device).gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))


class DiffusionRegPoseDiffusion(nn.Module):
    """Store diffusion coefficients and expose proposal-state transitions."""

    def __init__(
        self,
        *,
        timesteps: int = 1000,
        sampling_timesteps: int | None = None,
        scale: float = 6.0,
        ddim_sampling_eta: float = 1.0,
    ) -> None:
        super().__init__()
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = (
            self.num_timesteps if sampling_timesteps is None else int(sampling_timesteps)
        )
        if not 1 <= self.sampling_timesteps <= self.num_timesteps:
            raise ValueError(
                "sampling_timesteps must be between 1 and timesteps; "
                f"got {self.sampling_timesteps} and {self.num_timesteps}"
            )
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        self.ddim_sampling_eta = float(ddim_sampling_eta)
        self.scale = float(scale)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20))
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} must match x_start {tuple(x_start.shape)}")
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_noise_from_start(self, x_t: Tensor, timesteps: Tensor, x_start: Tensor) -> Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t - x_start
        ) / extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape)

    def ddim_timestep_pairs(self) -> list[tuple[int, int]]:
        times = torch.linspace(
            -1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1
        )
        ordered_times = list(reversed(times.int().tolist()))
        return list(zip(ordered_times[:-1], ordered_times[1:]))

    def ddim_step(
        self,
        x_start: Tensor,
        pred_noise: Tensor,
        time: int,
        time_next: int,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Advance one deterministic DDIM step used by reference evaluation."""
        if x_start.shape != pred_noise.shape:
            raise ValueError("x_start and pred_noise must have identical shapes")
        if not 0 <= time < self.num_timesteps:
            raise ValueError(f"time must be in [0, {self.num_timesteps}); got {time}")
        if time_next < -1 or time_next >= time:
            raise ValueError(f"time_next must be in [-1, time); got {time_next} for time {time}")
        if time_next < 0:
            return x_start

        alpha = self.alphas_cumprod[time].to(dtype=x_start.dtype)
        alpha_next = self.alphas_cumprod[time_next].to(dtype=x_start.dtype)
        sigma = self.ddim_sampling_eta * (
            (1.0 - alpha / alpha_next) * (1.0 - alpha_next) / (1.0 - alpha)
        ).clamp_min(0).sqrt()
        coefficient = (1.0 - alpha_next - sigma.square()).clamp_min(0).sqrt()
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            raise ValueError("noise must have the same shape as x_start")
        return alpha_next.sqrt() * x_start + coefficient * pred_noise + sigma * noise
