"""Reusable building blocks for the box-set diffusion decoder.

This module is intentionally framework-agnostic: it contains plain
``nn.Module`` building blocks that do not depend on any project-specific
config plumbing. Higher-level decoders compose these modules.

Contents
--------
- :class:`FourierFeatures` — scalar Fourier feature bank with an MLP head.
- :class:`TimeFiLMFFN` — pre-norm FFN modulated by a time embedding (FiLM).
- :class:`TimeFiLMTransformerBlock` — transformer block with optional
  cross-attention and a FiLM-modulated FFN.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class FourierFeatures(nn.Module):
    """Random / log-spaced Fourier feature bank for scalar inputs.

    For an input tensor ``x`` with shape ``[..., D_in]``, each scalar entry is
    expanded via ``[sin(2π · scale · s · f_k), cos(2π · scale · s · f_k)]``
    where ``f_k`` are ``num_bands`` log-spaced frequencies in ``[1, max_freq]``.
    The resulting features (``D_in · 2 · num_bands`` channels) are then
    projected with ``Linear → GELU → Linear`` to ``out_dim``.

    Args:
        in_dim: Number of scalar input channels (1 for σ, 4 for box coords, …).
        out_dim: Output feature dimension.
        num_bands: Number of frequency bands.
        scale: Multiplicative scale applied before computing sin/cos. Use
            e.g. ``1000`` for box coordinates in ``[-1, 1]`` so the highest
            frequency resolves sub-pixel structure.
        max_freq: Maximum frequency (the lowest is 1).
        hidden_dim: Hidden dim of the projection MLP (defaults to ``out_dim``).
    """

    freqs: Tensor

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_bands: int = 16,
        scale: float = 1.0,
        max_freq: float = 256.0,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_bands < 1:
            raise ValueError(f"num_bands must be >= 1, got {num_bands}")
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_bands = num_bands
        self.scale = float(scale)

        freqs = torch.logspace(0.0, math.log10(max_freq), num_bands)  # [num_bands]
        self.register_buffer("freqs", freqs, persistent=False)

        feat_dim = in_dim * 2 * num_bands
        hidden = hidden_dim if hidden_dim is not None else out_dim
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map ``[..., in_dim]`` to ``[..., out_dim]``."""
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected last dim {self.in_dim}, got {x.shape[-1]}"
            )
        # phases: [..., in_dim, num_bands]
        phases = (x * self.scale).unsqueeze(-1) * self.freqs * (2.0 * math.pi)
        sins = torch.sin(phases)
        coss = torch.cos(phases)
        feats = torch.cat([sins, coss], dim=-1)            # [..., in_dim, 2·num_bands]
        feats = feats.flatten(start_dim=-2)                # [..., in_dim·2·num_bands]
        return self.proj(feats)


class TimeFiLMFFN(nn.Module):
    """Pre-norm FFN modulated by a time embedding via FiLM.

    Pipeline:
        ``x → LN → Linear(D, mD) → GELU → FiLM(γ, β | t) → Linear(mD, D)``
    where ``γ, β = chunk(Linear(time_dim → 2·mD)(t))``. ``γ`` is initialised
    to behave near identity (zero pre-scale; the FFN is unaffected at init).
    """

    def __init__(self, dim: int, ffn_mult: float = 4.0, time_dim: int = 0) -> None:
        super().__init__()
        hidden = int(round(dim * ffn_mult))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.use_film = time_dim > 0
        if self.use_film:
            self.film = nn.Linear(time_dim, 2 * hidden)
            # zero-init so the block is the identity FFN at init time
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x: Tensor, time_emb: Optional[Tensor] = None) -> Tensor:
        """Apply the FFN.

        Args:
            x: ``[..., dim]`` input.
            time_emb: ``[B, time_dim]`` time embedding (broadcast over any
                middle dims of ``x``). Required iff ``time_dim > 0``.
        """
        h = self.fc1(self.norm(x))
        h = self.act(h)
        if self.use_film:
            if time_emb is None:
                raise ValueError("time_emb is required when time_dim > 0")
            gamma, beta = self.film(time_emb).chunk(2, dim=-1)
            # broadcast B-dim FiLM over any sequence dim of h
            while gamma.ndim < h.ndim:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            h = h * (1.0 + gamma) + beta
        return self.fc2(h)


class TimeFiLMTransformerBlock(nn.Module):
    """Transformer block: self-attn + optional cross-attn + FiLM-FFN.

    All sub-blocks use pre-norm and residual connections. Attention layers
    are plain ``nn.MultiheadAttention`` (batch_first). The FFN is
    :class:`TimeFiLMFFN`, optionally modulated by a time embedding.

    Args:
        dim: Hidden dim for queries (and keys/values when not cross-attending).
        num_heads: Number of attention heads.
        time_dim: Time-embedding dim (0 disables FiLM).
        ffn_mult: FFN hidden ratio.
        cross_dim: Key/value dim for cross-attention (if used). Defaults
            to ``dim``. Pass ``None`` AND call ``forward(context=None)`` to
            skip cross-attn entirely.
        dropout: Attention/FFN dropout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        time_dim: int = 0,
        ffn_mult: float = 4.0,
        cross_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_cross_attn: bool = True,
    ) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )

        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.norm_cross_q = nn.LayerNorm(dim)
            self.norm_cross_kv = nn.LayerNorm(cross_dim if cross_dim is not None else dim)
            self.cross_attn = nn.MultiheadAttention(
                dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
                kdim=cross_dim if cross_dim is not None else dim,
                vdim=cross_dim if cross_dim is not None else dim,
            )

        self.ffn = TimeFiLMFFN(dim, ffn_mult=ffn_mult, time_dim=time_dim)

    def forward(
        self,
        x: Tensor,                                       # [B, N, dim]
        context: Optional[Tensor] = None,                # [B, T, cross_dim]
        time_emb: Optional[Tensor] = None,               # [B, time_dim]
        context_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        h = self.norm_self(x)
        sa, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + sa

        if self.use_cross_attn:
            if context is None:
                raise ValueError("context is required when use_cross_attn=True")
            q = self.norm_cross_q(x)
            kv = self.norm_cross_kv(context)
            ca, _ = self.cross_attn(
                q, kv, kv,
                key_padding_mask=context_key_padding_mask,
                need_weights=False,
            )
            x = x + ca

        x = x + self.ffn(x, time_emb=time_emb)
        return x
