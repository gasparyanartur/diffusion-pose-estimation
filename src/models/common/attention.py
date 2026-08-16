from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_bands: int = 16, scale: float = 1.0, max_freq: float = 256.0, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        if num_bands < 1:
            raise ValueError(f"num_bands must be >= 1, got {num_bands}")
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        self.in_dim = in_dim
        self.scale = float(scale)
        self.register_buffer("freqs", torch.logspace(0.0, math.log10(max_freq), num_bands), persistent=False)
        feature_dim = in_dim * 2 * num_bands
        hidden_dim = hidden_dim or out_dim
        self.proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"expected last dim {self.in_dim}, got {x.shape[-1]}")
        phases = (x * self.scale).unsqueeze(-1) * self.freqs * (2.0 * math.pi)
        features = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        return self.proj(features.flatten(start_dim=-2))


class TimeFiLMFFN(nn.Module):
    def __init__(self, dim: int, ffn_mult: float = 4.0, time_dim: int = 0) -> None:
        super().__init__()
        hidden_dim = int(round(dim * ffn_mult))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.use_film = time_dim > 0
        if self.use_film:
            self.film = nn.Linear(time_dim, 2 * hidden_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x: Tensor, time_emb: Optional[Tensor] = None) -> Tensor:
        hidden = self.activation(self.fc1(self.norm(x)))
        if self.use_film:
            if time_emb is None:
                raise ValueError("time_emb is required when time_dim > 0")
            scale, shift = self.film(time_emb).chunk(2, dim=-1)
            while scale.ndim < hidden.ndim:
                scale = scale.unsqueeze(1)
                shift = shift.unsqueeze(1)
            hidden = hidden * (1.0 + scale) + shift
        return self.fc2(hidden)


class TimeFiLMTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, time_dim: int = 0, ffn_mult: float = 4.0, cross_dim: Optional[int] = None, dropout: float = 0.0, use_cross_attn: bool = True) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            key_value_dim = cross_dim or dim
            self.norm_cross_q = nn.LayerNorm(dim)
            self.norm_cross_kv = nn.LayerNorm(key_value_dim)
            self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True, kdim=key_value_dim, vdim=key_value_dim)
        self.ffn = TimeFiLMFFN(dim, ffn_mult=ffn_mult, time_dim=time_dim)

    def forward(self, x: Tensor, context: Optional[Tensor] = None, time_emb: Optional[Tensor] = None, context_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        normalized = self.norm_self(x)
        self_attention, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
        x = x + self_attention
        if self.use_cross_attn:
            if context is None:
                raise ValueError("context is required when use_cross_attn=True")
            query = self.norm_cross_q(x)
            key_value = self.norm_cross_kv(context)
            cross_attention, _ = self.cross_attn(query, key_value, key_value, key_padding_mask=context_key_padding_mask, need_weights=False)
            x = x + cross_attention
        return x + self.ffn(x, time_emb=time_emb)
