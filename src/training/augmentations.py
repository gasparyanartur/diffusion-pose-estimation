"""Training-time augmentations on pose samples and graphs."""

from __future__ import annotations

import torch
from torch import Tensor


def keypoint_jitter(keypoints: Tensor, std_px: float) -> Tensor:
    """Add Gaussian noise to keypoint coordinates (pixel space)."""
    return keypoints + torch.randn_like(keypoints) * std_px


def joint_dropout(node_pos: Tensor, can_vis: Tensor, p: float) -> Tensor:
    """Drop individual non-root joints (zero position + zero visibility)."""
    B, N, K = can_vis.shape
    drop_mask = torch.zeros_like(can_vis, dtype=torch.bool)
    drop_mask[:, :, 1:] = torch.bernoulli(
        torch.full((B, N, K - 1), p, device=node_pos.device)
    ).bool()
    node_pos[drop_mask.reshape(-1)] = 0.0
    return can_vis.masked_fill(drop_mask, 0.0)


def joint_randomize(node_pos: Tensor, can_vis: Tensor, p: float) -> Tensor:
    """Replace non-root joints with random canonical positions; visibility kept."""
    B, N, K = can_vis.shape
    rand_mask = torch.zeros_like(can_vis, dtype=torch.bool)
    rand_mask[:, :, 1:] = torch.bernoulli(
        torch.full((B, N, K - 1), p, device=node_pos.device)
    ).bool()
    flat_mask = rand_mask.reshape(-1)
    node_pos[flat_mask] = torch.rand_like(node_pos[flat_mask]) * 2.0 - 1.0
    return can_vis


def pose_dropout(node_pos: Tensor, can_vis: Tensor, p: float) -> Tensor:
    """Drop entire persons (all joints zeroed)."""
    B, N, K = can_vis.shape
    person_drop = torch.bernoulli(torch.full((B, N), p, device=node_pos.device)).bool()
    drop_mask = person_drop.unsqueeze(-1).expand(B, N, K)
    node_pos[drop_mask.reshape(-1)] = 0.0
    return can_vis.masked_fill(drop_mask, 0.0)


def image_dropout(image_cond: Tensor, p: float) -> None:
    """In-place: zero out a random fraction of node-conditioning rows."""
    drop = torch.bernoulli(
        torch.full((image_cond.shape[0],), p, device=image_cond.device)
    ).bool()
    image_cond[drop] = 0.0
