"""Cross-attention embedder.

Architecture:
  1. Fuse spatial + upsampled semantic patches per image → patch tokens.
  2. Project keypoint positions → query tokens.
  3. Cross-attention (Q=nodes, KV=patches) with a per-head relative-position
     bias (patch_pos − node_pos → linear → num_heads); padded patches masked.
  4. Self-attention across the K joints of each person; absent joints masked.
  5. Linear → ``out_dim`` per node.

Patch padding: images have variable ``P_i`` due to aspect-ratio-preserving
resize. All images are zero-padded to ``max_P``; padded slots get ``-inf``
attention bias so they cannot influence outputs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.embedders._sampling import pixel_to_grid_sample_coords
from src.embedders.base import BaseFeatureEmbedder, BaseFeatureEmbedderConfig


class CrossAttentionEmbedder(BaseFeatureEmbedder):
    """Cross-attend keypoint queries against fused image patch tokens."""

    def __init__(
        self,
        spatial_dim: int = 96,
        hidden_dim: int = 768,
        out_dim: int = 128,
        inner_dim: int = 256,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        global_out_dim: int | None = None,
    ) -> None:
        super().__init__()
        if inner_dim % num_heads != 0:
            raise ValueError(
                f"inner_dim ({inner_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads

        self.patch_fusion = nn.Sequential(
            nn.Linear(spatial_dim + hidden_dim, inner_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(inner_dim * 2, inner_dim),
        )
        self.node_pos_proj = nn.Linear(2, inner_dim)
        self.cross_rel_pos_bias = nn.Linear(2, num_heads, bias=False)

        self.cross_attn = nn.MultiheadAttention(
            inner_dim, num_heads, batch_first=True, dropout=dropout_rate
        )
        self.cross_norm = nn.LayerNorm(inner_dim)
        self.cross_linear = nn.Linear(inner_dim, inner_dim)
        self.cross_ff_norm = nn.LayerNorm(inner_dim)

        self.joint_pos_proj = nn.Linear(2, inner_dim)

        self.self_attn = nn.MultiheadAttention(
            inner_dim, num_heads, batch_first=True, dropout=dropout_rate
        )
        self.self_norm = nn.LayerNorm(inner_dim)
        self.self_linear = nn.Linear(inner_dim, inner_dim)
        self.self_ff_norm = nn.LayerNorm(inner_dim)

        self.out_proj = nn.Linear(inner_dim, out_dim)

        # Optional projection of the pooled global vector (768-d) → global_out_dim.
        # Matches the old global_dino_proj in PoseDiffusionModule.
        self.global_proj: nn.Linear | None = None
        if global_out_dim is not None:
            self.global_proj = nn.Linear(hidden_dim, global_out_dim)

    def _fuse_patches(
        self,
        spatial_map: Tensor,   # (C_s, H_s, W_s)
        semantic_map: Tensor,  # (C_h, H_s/k, W_s/k)
    ) -> tuple[Tensor, Tensor]:  # (P, inner_dim), (P, 2)
        C_s, H_s, W_s = spatial_map.shape
        P = H_s * W_s

        semantic_up = F.interpolate(
            semantic_map.unsqueeze(0),
            size=(H_s, W_s),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        spatial_flat = spatial_map.permute(1, 2, 0).reshape(P, C_s)
        semantic_flat = semantic_up.permute(1, 2, 0).reshape(P, self.hidden_dim)
        combined = torch.cat([spatial_flat, semantic_flat], dim=-1)

        patch_feats = self.patch_fusion(combined)

        device = spatial_map.device
        ys = (torch.arange(H_s, device=device).float() + 0.5) / H_s * 2 - 1
        xs = (torch.arange(W_s, device=device).float() + 0.5) / W_s * 2 - 1
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        patch_pos = torch.stack([grid_x, grid_y], dim=-1).reshape(P, 2)
        return patch_feats, patch_pos

    def _build_patch_batch(
        self,
        all_patch_feats: list[Tensor],
        all_patch_pos: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        B = len(all_patch_feats)
        max_P = max(f.shape[0] for f in all_patch_feats)
        device = all_patch_feats[0].device
        dtype = all_patch_feats[0].dtype

        padded_feats = torch.zeros(B, max_P, self.inner_dim, device=device, dtype=dtype)
        padded_pos = torch.zeros(B, max_P, 2, device=device, dtype=dtype)
        patch_key_mask = torch.ones(B, max_P, dtype=torch.bool, device=device)

        for i, (feats, pos) in enumerate(zip(all_patch_feats, all_patch_pos)):
            P_i = feats.shape[0]
            padded_feats[i, :P_i] = feats
            padded_pos[i, :P_i] = pos
            patch_key_mask[i, :P_i] = False
        return padded_feats, padded_pos, patch_key_mask

    def _cross_attend(
        self,
        node_feats: Tensor,
        patch_feats: Tensor,
        node_pos: Tensor,
        patch_pos: Tensor,
        patch_key_mask: Tensor,
    ) -> Tensor:
        B, NK, _ = node_feats.shape
        max_P = patch_feats.shape[1]
        H = self.num_heads

        rel_pos = patch_pos.unsqueeze(1) - node_pos.unsqueeze(2)
        rel_bias = self.cross_rel_pos_bias(rel_pos)
        rel_bias = rel_bias.permute(0, 3, 1, 2).reshape(B * H, NK, max_P)

        pad_inf = (
            patch_key_mask
            .view(B, 1, 1, max_P)
            .expand(B, H, NK, max_P)
            .reshape(B * H, NK, max_P)
        )
        rel_bias = rel_bias.masked_fill(pad_inf, float("-inf"))

        attn_out, _ = self.cross_attn(node_feats, patch_feats, patch_feats, attn_mask=rel_bias)
        h = self.cross_norm(node_feats + attn_out)
        return self.cross_ff_norm(h + self.cross_linear(h))

    def _self_attend_joints(
        self,
        h: Tensor,
        joint_pos: Tensor,
        joint_mask: Tensor | None,
    ) -> Tensor:
        q = h + self.joint_pos_proj(joint_pos)

        safe_mask = None
        if joint_mask is not None:
            all_absent = joint_mask.all(dim=-1)
            safe_mask = joint_mask.clone()
            if all_absent.any():
                safe_mask[all_absent, 0] = False

        attn_out, _ = self.self_attn(q, q, q, key_padding_mask=safe_mask)
        h = self.self_norm(q + attn_out)
        return self.self_ff_norm(h + self.self_linear(h))

    def forward(
        self,
        features: dict[str, Any],
        kpt_pixel: Tensor | None = None,
        img_hw: Tensor | None = None,
        joint_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        if kpt_pixel is None or img_hw is None:
            raise ValueError(
                "CrossAttentionEmbedder requires kpt_pixel and img_hw"
            )
        spatial_maps = features["spatial"]
        semantic_maps = features["semantic"]

        B, N, K, _ = kpt_pixel.shape
        device = next(self.patch_fusion.parameters()).device

        all_patch_feats: list[Tensor] = []
        all_patch_pos: list[Tensor] = []
        for i in range(B):
            feats, pos = self._fuse_patches(spatial_maps[i].to(device), semantic_maps[i].to(device))
            all_patch_feats.append(feats)
            all_patch_pos.append(pos)

        padded_feats, padded_pos, patch_key_mask = self._build_patch_batch(
            all_patch_feats, all_patch_pos
        )

        node_pos_norm = pixel_to_grid_sample_coords(kpt_pixel, img_hw.view(B, 1, 1, 2))
        node_pos_flat = node_pos_norm.view(B, N * K, 2)
        h = self.node_pos_proj(node_pos_flat)

        h = h + self._cross_attend(h, padded_feats, node_pos_flat, padded_pos, patch_key_mask)

        joint_pos = node_pos_norm.view(B * N, K, 2)
        h = h.view(B * N, K, self.inner_dim)
        joint_mask_flat = joint_mask.view(B * N, K) if joint_mask is not None else None
        h = h + self._self_attend_joints(h, joint_pos, joint_mask_flat)

        out = self.out_proj(h)
        if joint_mask is not None:
            out = out.masked_fill(joint_mask_flat.unsqueeze(-1), 0.0)
        node_feats = out.view(B, N, K, self.out_dim)

        result: dict[str, Tensor] = {"node_feats": node_feats}
        if "global" in features:
            g = features["global"].to(device)
            if self.global_proj is not None:
                g = self.global_proj(g)
            result["global_feats"] = g
        return result


class CrossAttentionEmbedderConfig(BaseFeatureEmbedderConfig):
    spatial_dim: int = 96
    hidden_dim: int = 768
    out_dim: int = 128
    inner_dim: int = 256
    num_heads: int = 4
    dropout_rate: float = 0.1
    global_out_dim: int | None = None
    """If set, projects the 768-d pooled global vector to this dim before
    passing it to the decoder. Set to match decoder.dim_global_image_cond."""

    def build(self) -> CrossAttentionEmbedder:
        return CrossAttentionEmbedder(
            spatial_dim=self.spatial_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            inner_dim=self.inner_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            global_out_dim=self.global_out_dim,
        )
