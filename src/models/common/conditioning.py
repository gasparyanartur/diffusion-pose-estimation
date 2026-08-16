from __future__ import annotations

import torch
from torch import Tensor, nn


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation from a conditioning vector."""

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feature_dim)

    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.proj(condition).chunk(2, dim=-1)
        return (1.0 + scale) * features + shift
