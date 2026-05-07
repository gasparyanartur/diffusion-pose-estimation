"""Shared sampling helper used by multiple embedders."""

from __future__ import annotations

from torch import Tensor


def pixel_to_grid_sample_coords(kpt_pixel: Tensor, img_hw: Tensor) -> Tensor:
    """Convert pixel-space ``(x, y)`` to ``grid_sample``-compatible coords in [-1, 1].

    Args:
        kpt_pixel: ``(..., 2)`` pixel-space (x, y).
        img_hw:    ``(..., 2)`` original image (H, W); broadcastable.
    """
    img_h = img_hw[..., 0].float()
    img_w = img_hw[..., 1].float()
    x_norm = (kpt_pixel[..., 0].float() / img_w) * 2 - 1
    y_norm = (kpt_pixel[..., 1].float() / img_h) * 2 - 1
    import torch
    return torch.stack([x_norm, y_norm], dim=-1)
