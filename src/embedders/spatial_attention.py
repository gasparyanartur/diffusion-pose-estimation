"""Spatial-attention embedder.

Pipeline:
    spatial feature map [B, C, H', W']
    -> 1x1 conv             [B, D, H', W']
    -> + 2D sinusoidal PE   [B, H'*W', D]
    -> single self-attention layer (residual + LayerNorm)
    -> mean-pool over tokens -> [B, D]

Output dict: ``{"feature": [B, D]}``. Size-independent: works for any (H', W')
because the PE is generated on the fly.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.embedders.base import BaseFeatureEmbedder, BaseFeatureEmbedderConfig


def make_2d_sinusoidal_pe(H: int, W: int, dim: int, device: torch.device,
                          dtype: torch.dtype = torch.float32) -> Tensor:
    """Fixed 2D sinusoidal positional encoding, shape ``[H*W, dim]``.

    ``dim`` is split into 4 equal banks: sin/cos for the H-axis and sin/cos
    for the W-axis.
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sinusoidal PE; got {dim}")
    d = dim // 4
    freq = 1.0 / (10000 ** (torch.arange(d, device=device, dtype=dtype) / d))
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    y_enc = y[:, None] * freq[None, :]
    x_enc = x[:, None] * freq[None, :]
    pe_h = torch.cat([y_enc.sin(), y_enc.cos()], dim=-1)  # [H, 2d]
    pe_w = torch.cat([x_enc.sin(), x_enc.cos()], dim=-1)  # [W, 2d]
    pe = torch.cat([
        pe_h[:, None, :].expand(H, W, -1),
        pe_w[None, :, :].expand(H, W, -1),
    ], dim=-1)  # [H, W, dim]
    return pe.reshape(H * W, dim)


class SpatialAttentionEmbedder(BaseFeatureEmbedder):
    """1x1 conv -> 2D sinusoidal PE -> self-attention -> mean-pool -> [B, D]."""

    def __init__(
        self,
        in_channels: int,
        feature_dim: int = 256,
        num_heads: int = 8,
        feature_key: str = "spatial",
    ) -> None:
        super().__init__()
        if feature_dim % 4 != 0:
            raise ValueError(
                f"feature_dim must be divisible by 4; got {feature_dim}"
            )
        self.project = nn.Conv2d(in_channels, feature_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.feature_dim = feature_dim
        self.feature_key = feature_key

    def forward(
        self,
        features: dict[str, Any],
        kpt_pixel: Tensor | None = None,  # noqa: ARG002
        img_hw: Tensor | None = None,     # noqa: ARG002
        joint_mask: Tensor | None = None, # noqa: ARG002
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        feat = features[self.feature_key]
        if not isinstance(feat, Tensor):
            raise TypeError(
                f"SpatialAttentionEmbedder expects features['{self.feature_key}'] "
                "as a Tensor [B, C, H, W]"
            )
        B, _, H, W = feat.shape
        x = self.project(feat)                                  # [B, D, H, W]
        pe = make_2d_sinusoidal_pe(H, W, self.feature_dim, x.device, x.dtype)
        tokens = x.flatten(2).permute(0, 2, 1)                   # [B, H*W, D]
        tokens = tokens + pe.unsqueeze(0)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        return {"feature": tokens.mean(dim=1)}                   # [B, D]


class SpatialAttentionEmbedderConfig(BaseFeatureEmbedderConfig):
    in_channels: int = 128
    feature_dim: int = 256
    num_heads: int = 8
    feature_key: str = "spatial"

    def build(self) -> SpatialAttentionEmbedder:
        return SpatialAttentionEmbedder(
            in_channels=self.in_channels,
            feature_dim=self.feature_dim,
            num_heads=self.num_heads,
            feature_key=self.feature_key,
        )
