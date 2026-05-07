"""Denoise networks: predict a per-node update from a noisy pose state.

A denoise network is the inner module used by a diffusion decoder. Given the
current node positions, edges, conditioning, and an optional global signal
(e.g. timestep, pooled image), it predicts a delta per node.
"""

from __future__ import annotations

from abc import abstractmethod

from torch import Tensor, nn

from src.base_config import BaseConfig


class BaseDenoiseNetwork(nn.Module):
    """Predicts ``(M, out_dim)`` from node state + edges + conditioning."""

    @abstractmethod
    def forward(
        self,
        h: Tensor,            # (M, hidden_dim)
        edge_index: Tensor,   # (2, E)
        e: Tensor,            # (E, dim_edge)
        v: Tensor,            # (M, dim_vis)
        g: Tensor,            # (M, dim_global)
    ) -> Tensor:
        ...


class BaseDenoiseNetworkConfig(BaseConfig):
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dim_edge: int = 64
    dim_vis: int = 128
    dim_global: int = 128
    out_dim: int = 2
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1

    def build(self) -> BaseDenoiseNetwork:
        raise NotImplementedError
