"""Loss functions for keypoint training."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_joint_loss(
    pred_pixel: Tensor,   # (B, N, K, 2)
    gt_pixel: Tensor,     # (B, N, K, 2)
    visibility: Tensor,   # (B, N, K)  0=absent, 1=occluded, 2=visible
) -> Tensor:
    """Visibility-weighted squared joint error in pixel space."""
    visibility_weight = visibility.float()
    per_joint_sq_err = ((pred_pixel - gt_pixel) ** 2).sum(dim=-1)
    total_weight = visibility_weight.sum().clamp(min=1.0)
    return (per_joint_sq_err * visibility_weight).sum() / total_weight


def compute_bone_loss(
    pred_pixel: Tensor,
    gt_pixel: Tensor,
    visibility: Tensor,
    edge_list: list[tuple[int, int]],
) -> Tensor:
    """Per-edge squared length error, weighted by endpoint visibility product."""
    visibility_weight = visibility.float()
    loss = gt_pixel.new_zeros(())
    for u, v in edge_list:
        edge_weight = visibility_weight[:, :, u] * visibility_weight[:, :, v]
        weight_sum = edge_weight.sum().clamp(min=1.0)
        pred_len = (pred_pixel[:, :, u] - pred_pixel[:, :, v]).norm(dim=-1)
        gt_len = (gt_pixel[:, :, u] - gt_pixel[:, :, v]).norm(dim=-1)
        loss = loss + ((pred_len - gt_len).pow(2) * edge_weight).sum() / weight_sum
    return loss / len(edge_list)
