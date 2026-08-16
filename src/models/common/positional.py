from __future__ import annotations

import torch
from torch import Tensor


def make_2d_sinusoidal_pe(height: int, width: int, dim: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sinusoidal PE; got {dim}")
    axis_dim = dim // 4
    frequencies = 1.0 / (10000 ** (torch.arange(axis_dim, device=device, dtype=dtype) / axis_dim))
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    y_encoding = y[:, None] * frequencies[None, :]
    x_encoding = x[:, None] * frequencies[None, :]
    y_features = torch.cat([y_encoding.sin(), y_encoding.cos()], dim=-1)
    x_features = torch.cat([x_encoding.sin(), x_encoding.cos()], dim=-1)
    features = torch.cat([y_features[:, None, :].expand(height, width, -1), x_features[None, :, :].expand(height, width, -1)], dim=-1)
    return features.reshape(height * width, dim)


def make_patch_centers(height: int, width: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([grid_y, grid_x], dim=-1)


def make_pe_from_centers(centers: Tensor, dim: int, scale: float = 1000.0) -> Tensor:
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for sinusoidal PE; got {dim}")
    axis_dim = dim // 4
    frequencies = 1.0 / (10000 ** (torch.arange(axis_dim, device=centers.device, dtype=centers.dtype) / axis_dim))
    y_encoding = centers[:, 0:1] * scale * frequencies
    x_encoding = centers[:, 1:2] * scale * frequencies
    return torch.cat([y_encoding.sin(), y_encoding.cos(), x_encoding.sin(), x_encoding.cos()], dim=-1)
