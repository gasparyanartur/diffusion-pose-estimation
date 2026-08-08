"""Graph diffusion decoder: position/index/edge/time embeddings + denoise network."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.data.graph import PoseGraphBatch
from src.decoders.denoise.base import BaseDenoiseNetwork, BaseDenoiseNetworkConfig
from src.decoders.diffusion.base import (
    BaseDiffusionDecoder,
    BaseDiffusionDecoderConfig,
    TimestepSinusoidalEmbedding,
)


class GraphDiffusionRefinerDecoder(BaseDiffusionDecoder):
    """Legacy graph-structured pose-refinement decoder.

    Embeddings:
      h   (M, hidden)   = LN(pos_emb(noisy) + idx_emb(joint_type))
      e   (E, dim_edge) = edge_encoder(noisy[src] - noisy[dst])
      v   (M, dim_vis)  = ``embeddings["node_feats"]`` (zero if absent)
      g   (M, dim_g)    = MLP(timestep_emb [+ ``embeddings["global_feats"]``])

    The denoise network maps (h, edge_index, e, v, g) → predicted delta (M, 2).
    """

    def __init__(
        self,
        denoise_network: BaseDenoiseNetwork,
        num_keypoints: int,
        hidden_dim: int,
        dim_pos: int,
        dim_time: int,
        dim_edge: int,
        dim_vis: int,
        dim_global_image_cond: int = 0,  # 0 if no global conditioning
        dropout_rate: float = 0.1,
        num_train_steps: int = 100,
        sigma_min: float = 0.002,
        sigma_max: float = 0.5,
    ) -> None:
        super().__init__(
            num_train_steps=num_train_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim
        self.dim_vis = dim_vis
        self.dim_global_image_cond = dim_global_image_cond

        self.position_embedding = nn.Sequential(
            nn.Linear(2, dim_pos),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_pos, hidden_dim),
        )
        self.node_idx_embedding = nn.Embedding(num_keypoints, hidden_dim)
        self.node_init_norm = nn.LayerNorm(hidden_dim)

        self.timestep_embedding = TimestepSinusoidalEmbedding(dim_time)
        global_in = dim_time + dim_global_image_cond
        self.global_cond_mlp = nn.Sequential(
            nn.Linear(global_in, dim_time),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_time, dim_time),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(2, dim_edge),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_edge, dim_edge),
        )

        self.denoise_network = denoise_network

    def predict_delta(
        self,
        noisy_node_pos: Tensor,
        graph: PoseGraphBatch,
        embeddings: dict[str, Tensor],
        timestep: Tensor,
    ) -> Tensor:
        M = noisy_node_pos.shape[0]
        device = noisy_node_pos.device
        dtype = noisy_node_pos.dtype

        h = self.node_init_norm(
            self.position_embedding(noisy_node_pos)
            + self.node_idx_embedding(graph.flat_kpt_type_idx)
        )

        src, dst = graph.edge_index[0], graph.edge_index[1]
        e = self.edge_encoder(noisy_node_pos[src] - noisy_node_pos[dst])

        node_feats = embeddings.get("node_feats")
        if node_feats is not None:
            v = node_feats.reshape(M, -1)
        else:
            v = torch.zeros(M, self.dim_vis, device=device, dtype=dtype)

        t_emb = self.timestep_embedding(timestep)[graph.node_batch]
        global_feats = embeddings.get("global_feats")
        if global_feats is not None:
            g_in = torch.cat([t_emb, global_feats[graph.node_batch]], dim=-1)
        else:
            g_in = t_emb
        g = self.global_cond_mlp(g_in)

        return self.denoise_network(h, graph.edge_index, e, v, g)


class GraphDiffusionRefinerDecoderConfig(BaseDiffusionDecoderConfig):
    """Config for :class:`GraphDiffusionRefinerDecoder`.

    The inner denoise network is built from ``denoise_network`` (a polymorphic
    sub-config). ``hidden_dim`` etc. must match between this config and the
    denoise network config.
    """

    denoise_network: BaseDenoiseNetworkConfig
    num_keypoints: int = 14
    hidden_dim: int = 128
    dim_pos: int = 128
    dim_time: int = 128
    dim_edge: int = 64
    dim_vis: int = 128
    dim_global_image_cond: int = 0
    dropout_rate: float = 0.1

    def build(self) -> GraphDiffusionRefinerDecoder:
        return GraphDiffusionRefinerDecoder(
            denoise_network=self.denoise_network.build(),
            num_keypoints=self.num_keypoints,
            hidden_dim=self.hidden_dim,
            dim_pos=self.dim_pos,
            dim_time=self.dim_time,
            dim_edge=self.dim_edge,
            dim_vis=self.dim_vis,
            dim_global_image_cond=self.dim_global_image_cond,
            dropout_rate=self.dropout_rate,
            num_train_steps=self.num_train_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
        )


# Compatibility names retained for existing refinement configs and imports.
GraphDiffusionDecoder = GraphDiffusionRefinerDecoder
GraphDiffusionDecoderConfig = GraphDiffusionRefinerDecoderConfig
