"""Base diffusion decoder: VE noise schedule + timestep encoding + DDIM sampler.

Subclasses supply the per-step denoise function that maps the current noisy
keypoint state (plus conditioning) to a predicted update.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.base_config import BaseConfig
from src.data.graph import PoseGraphBatch
from src.decoders.base import BaseDecoder, BaseDecoderConfig


class TimestepSinusoidalEmbedding(nn.Module):
    """Sinusoidal positional encoding for a batch of integer timesteps."""

    def __init__(self, dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, timestep: Tensor) -> Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32, device=timestep.device)
            * np.log(self.max_period)
            / half_dim
        )
        args = timestep[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class BaseDiffusionDecoder(BaseDecoder):
    """Diffusion decoder skeleton.

    Holds the VE noise schedule and exposes ``add_noise`` / ``ddim_sample``.
    Subclasses implement ``predict_delta`` to map a noisy state to a predicted
    update at a specific timestep.
    """

    _sigmas: Tensor

    def __init__(
        self,
        num_train_steps: int = 100,
        sigma_min: float = 0.002,
        sigma_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_train_steps = num_train_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # sigmas[0] = 0 sentinel (clean output at the end of DDIM); sigmas[1..T]
        # increase log-linearly from sigma_min to sigma_max.
        sigmas = torch.zeros(num_train_steps + 1)
        sigmas[1:] = torch.exp(
            torch.linspace(math.log(sigma_min), math.log(sigma_max), num_train_steps)
        )
        self.register_buffer("_sigmas", sigmas)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    @abstractmethod
    def predict_delta(
        self,
        noisy_node_pos: Tensor,
        graph: PoseGraphBatch,
        embeddings: dict[str, Tensor],
        timestep: Tensor,            # (B,)
    ) -> Tensor:
        """Predict ``noisy - x0`` per node; shape ``(M, 2)``."""

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------
    def add_noise(
        self,
        node_pos: Tensor,
        timestep: Tensor,            # (B,)
        batch_idx: Tensor,           # (M,)
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        noise = torch.zeros_like(node_pos).normal_(generator=generator)
        sigma = self._sigmas[timestep[batch_idx]].unsqueeze(-1)
        return node_pos + sigma * noise, noise

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ddim_sample(
        self,
        graph: PoseGraphBatch,
        embeddings: dict[str, Tensor],
        num_nodes: int,
        num_inference_steps: int | None = None,
        start_t: int | None = None,
        device: torch.device | None = None,
        x_init: Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        T = self.num_train_steps
        t_start = start_t if start_t is not None else T
        n_steps = num_inference_steps if num_inference_steps is not None else t_start
        device = device or graph.edge_index.device

        timesteps = torch.linspace(
            t_start, 1, n_steps, dtype=torch.long, device=device
        )
        prev_timesteps = torch.cat([timesteps[1:], timesteps.new_zeros(1)])

        x = x_init.to(device) if x_init is not None else torch.randn(num_nodes, 2, device=device)
        B = int(graph.node_batch.max().item()) + 1

        has_cond = bool(embeddings)
        for t, t_prev in zip(timesteps, prev_timesteps):
            t_batch = t.expand(B)
            if guidance_scale != 1.0 and has_cond:
                cond_delta = self.predict_delta(x, graph, embeddings, t_batch)
                uncond_delta = self.predict_delta(
                    x, graph, _zero_embeddings(embeddings), t_batch
                )
                delta = uncond_delta + guidance_scale * (cond_delta - uncond_delta)
            else:
                delta = self.predict_delta(x, graph, embeddings, t_batch)
            x0 = x - delta

            sigma_t = self._sigmas[t.long()]
            sigma_prev = self._sigmas[t_prev.long()]
            eps = (x - x0) / sigma_t
            x = x0 + sigma_prev * eps

        return x

    # ------------------------------------------------------------------
    # Decoder API: run inference end-to-end
    # ------------------------------------------------------------------
    def forward(
        self,
        embeddings: dict[str, Tensor],
        graph: PoseGraphBatch | None = None,
        initial_keypoints: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Run DDIM sampling. ``initial_keypoints`` are canonical-space, ``(M, 2)``.

        Returns a dict with ``keypoints`` reshaped to ``(B, N, K, 2)`` in canonical
        ``(-1, 1)`` space. Diffusion decoders score every slot/joint as 1.0 (no
        per-slot confidence to report); set-prediction decoders override these.
        """
        if graph is None:
            raise ValueError("BaseDiffusionDecoder.forward requires graph")
        flat = self.ddim_sample(
            graph=graph,
            embeddings=embeddings,
            num_nodes=graph.num_nodes,
            x_init=initial_keypoints,
        )
        B, N, K = graph.B, graph.N, graph.K
        keypoints = flat.view(B, N, K, 2)
        return {
            "keypoints": keypoints,
            "scores": keypoints.new_ones(B, N),
            "keypoint_scores": keypoints.new_ones(B, N, K),
        }


def _zero_embeddings(embeddings: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: torch.zeros_like(v) for k, v in embeddings.items()}


class BaseDiffusionDecoderConfig(BaseDecoderConfig):
    num_train_steps: int = 100
    sigma_min: float = 0.002
    sigma_max: float = 0.5

    def build(self) -> BaseDiffusionDecoder:
        raise NotImplementedError
