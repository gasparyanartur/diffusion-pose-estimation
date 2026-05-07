"""Bilinear-sampling embedder.

Bilinearly samples spatial + semantic feature maps at each keypoint location,
concatenates them, and projects to ``out_dim`` per node. Also exposes the
pre-pooled global vector if present in the feature dict.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.embedders._sampling import pixel_to_grid_sample_coords
from src.embedders.base import BaseFeatureEmbedder, BaseFeatureEmbedderConfig


def _sample_at_keypoints(
    feat_maps: list[Tensor],   # B tensors, each (C, H_i, W_i)
    kpt_pixel: Tensor,         # (B, N, K, 2) pixel-space
    img_hw: Tensor,            # (B, 2)
) -> Tensor:                   # (B, N, K, C)
    """Bilinearly sample each per-image feature map at keypoint pixel locations."""
    B = len(feat_maps)
    N, K = kpt_pixel.shape[1], kpt_pixel.shape[2]
    device = kpt_pixel.device
    results = []
    for i in range(B):
        feat = feat_maps[i].to(device)
        C = feat.shape[0]
        coords = pixel_to_grid_sample_coords(
            kpt_pixel[i].unsqueeze(0),
            img_hw[i].to(device).view(1, 1, 1, 2),
        )
        grid = coords.reshape(1, N * K, 1, 2)
        sampled = F.grid_sample(
            feat.unsqueeze(0).contiguous(), grid,
            mode="bilinear", align_corners=False, padding_mode="zeros",
        )
        results.append(sampled.squeeze(0).squeeze(-1).permute(1, 0).view(N, K, C))
    return torch.stack(results)


class BilinearSampleEmbedder(BaseFeatureEmbedder):
    """Concatenate stage1 + stage4 samples per node and project to ``out_dim``.

    Expected feature-dict keys:
        ``"spatial"``  — list of B tensors ``(C_s, H_i, W_i)``
        ``"semantic"`` — list of B tensors ``(C_h, H_j, W_j)``
        ``"global"``   — optional tensor ``(B, C_h)``
    """

    def __init__(
        self,
        spatial_dim: int = 96,
        hidden_dim: int = 768,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(spatial_dim + hidden_dim, out_dim)

    def forward(
        self,
        features: dict[str, Any],
        kpt_pixel: Tensor | None = None,
        img_hw: Tensor | None = None,
        joint_mask: Tensor | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        if kpt_pixel is None or img_hw is None:
            raise ValueError(
                "BilinearSampleEmbedder requires kpt_pixel and img_hw"
            )
        spatial = features["spatial"]
        semantic = features["semantic"]
        spatial_feats = _sample_at_keypoints(spatial, kpt_pixel, img_hw)    # (B, N, K, C_s)
        semantic_feats = _sample_at_keypoints(semantic, kpt_pixel, img_hw)  # (B, N, K, C_h)
        node_feats = self.proj(torch.cat([spatial_feats, semantic_feats], dim=-1))

        out: dict[str, Tensor] = {"node_feats": node_feats}
        if "global" in features:
            out["global_feats"] = features["global"].to(node_feats.device)
        return out


class BilinearSampleEmbedderConfig(BaseFeatureEmbedderConfig):
    spatial_dim: int = 96
    hidden_dim: int = 768
    out_dim: int = 128

    def build(self) -> BilinearSampleEmbedder:
        return BilinearSampleEmbedder(
            spatial_dim=self.spatial_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
        )
