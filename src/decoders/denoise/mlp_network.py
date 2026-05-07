"""MLP denoise network: per-node MLP, no message passing."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.decoders.denoise.base import BaseDenoiseNetwork, BaseDenoiseNetworkConfig
from src.decoders.denoise.graph_network import FiLMLayer


class MlpDenoiseLayer(nn.Module):
    """FiLM conditioning + MLP update on each node independently."""

    def __init__(
        self,
        dim: int,
        dim_vis: int,
        dim_global: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.film = FiLMLayer(dim_global, dim)
        update_in = dim + dim_vis
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(update_in),
            nn.Linear(update_in, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, h: Tensor, v: Tensor, g: Tensor) -> Tensor:
        h = self.film(h, g)
        return h + self.update_mlp(torch.cat([h, v], dim=-1))


class MlpDenoiseNetwork(BaseDenoiseNetwork):
    """Per-node MLP stack — does not use ``edge_index`` or ``e``."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dim_vis: int,
        dim_global: int,
        out_dim: int = 2,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MlpDenoiseLayer(hidden_dim, dim_vis, dim_global, dropout_rate)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        self.output_proj[1].weight.data.zero_()
        self.output_proj[1].bias.data.zero_()

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,  # noqa: ARG002 — unused; kept for interface parity
        e: Tensor,           # noqa: ARG002
        v: Tensor,
        g: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            h = layer(h, v, g)
        return self.output_proj(h)


class MlpDenoiseNetworkConfig(BaseDenoiseNetworkConfig):
    def build(self) -> MlpDenoiseNetwork:
        return MlpDenoiseNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dim_vis=self.dim_vis,
            dim_global=self.dim_global,
            out_dim=self.out_dim,
            dropout_rate=self.dropout_rate,
        )
