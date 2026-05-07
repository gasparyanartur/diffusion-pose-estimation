"""Graph denoise network: stacked FiLM + attention message-passing layers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from src.decoders.denoise.base import BaseDenoiseNetwork, BaseDenoiseNetworkConfig


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: ``h → (1 + scale) * h + shift``."""

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feature_dim)

    def forward(self, h: Tensor, g: Tensor) -> Tensor:
        params = self.proj(g)
        scale, shift = params.chunk(2, dim=-1)
        return (1.0 + scale) * h + shift


class PoseGraphLayer(MessagePassing):
    """Graph attention layer: FiLM → attentive message passing → residual update."""

    def __init__(
        self,
        dim: int,
        dim_edge: int,
        dim_vis: int,
        dim_global: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        attn_dropout_rate: float = 0.1,
    ) -> None:
        super().__init__(aggr="add")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = attn_dropout_rate

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

        msg_in = dim + dim_edge + dim_vis
        self.msg_mlp = nn.Sequential(
            nn.LayerNorm(msg_in),
            nn.Linear(msg_in, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 2, dim),
        )
        self.edge_bias = nn.Linear(dim_edge, num_heads, bias=False)

        update_in = dim * 2 + dim_vis
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(update_in),
            nn.Linear(update_in, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 2, dim),
        )

        self.film = FiLMLayer(dim_global, dim)

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_state_j: Tensor,
        edge_feat: Tensor,
        vis_j: Tensor,
        edge_dst: Tensor,
    ) -> Tensor:
        E = q_i.shape[0]
        H, D = self.num_heads, self.head_dim

        msg = self.msg_mlp(torch.cat([v_state_j, edge_feat, vis_j], dim=-1))

        q_h = q_i.view(E, H, D)
        k_h = k_j.view(E, H, D)
        dot = (q_h * k_h).sum(-1) / math.sqrt(D)
        score = dot + self.edge_bias(edge_feat)
        alpha = softmax(score, index=edge_dst, num_nodes=self._num_nodes)
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

        msg_h = msg.view(E, H, D)
        return (msg_h * alpha.unsqueeze(-1)).view(E, self.dim)

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        e: Tensor,
        v: Tensor,
        g: Tensor,
    ) -> Tensor:
        h = self.film(h, g)
        q = self.W_q(h)
        k = self.W_k(h)
        v_state = self.W_v(h)
        _, dst = edge_index[0], edge_index[1]
        self._num_nodes = int(h.shape[0])
        agg = self.propagate(
            edge_index,
            q=q, k=k, v_state=v_state,
            edge_feat=e,
            vis=v,
            edge_dst=dst,
        )
        return h + self.update_mlp(torch.cat([h, agg, v], dim=-1))


class GraphDenoiseNetwork(BaseDenoiseNetwork):
    """Stacked :class:`PoseGraphLayer`s + zero-initialised output projection."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dim_edge: int,
        dim_vis: int,
        dim_global: int,
        num_heads: int = 4,
        out_dim: int = 2,
        dropout_rate: float = 0.1,
        attn_dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            PoseGraphLayer(
                dim=hidden_dim,
                dim_edge=dim_edge,
                dim_vis=dim_vis,
                dim_global=dim_global,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attn_dropout_rate=attn_dropout_rate,
            )
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        # Zero-init final linear so untrained model predicts zero displacement.
        self.output_proj[1].weight.data.zero_()
        self.output_proj[1].bias.data.zero_()

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        e: Tensor,
        v: Tensor,
        g: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            h = layer(h, edge_index, e, v, g)
        return self.output_proj(h)


class GraphDenoiseNetworkConfig(BaseDenoiseNetworkConfig):
    def build(self) -> GraphDenoiseNetwork:
        return GraphDenoiseNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dim_edge=self.dim_edge,
            dim_vis=self.dim_vis,
            dim_global=self.dim_global,
            num_heads=self.num_heads,
            out_dim=self.out_dim,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
        )
