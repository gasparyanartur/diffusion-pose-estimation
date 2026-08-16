"""Training-time augmentations on pose samples and graphs."""

from __future__ import annotations

import torch
import torchvision.transforms.v2 as TFv2
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


# ---------------------------------------------------------------------------
# Image augmentations (pixel space, no spatial transforms)
# ---------------------------------------------------------------------------


def add_gaussian_noise(image: Tensor, std: float) -> Tensor:
    """Add i.i.d. Gaussian noise to a float [0,1] image batch, clamped to [0,1].

    Args:
        image: float tensor [..., C, H, W] in [0, 1].
        std:   noise standard deviation.
    """
    return (image + torch.randn_like(image) * std).clamp_(0.0, 1.0)


def color_jitter(
    image: Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
) -> Tensor:
    """Apply random ColorJitter to a float [0,1] image batch [B, C, H, W].

    Uses torchvision.transforms.v2 which supports batched tensors.

    Args:
        image:      float [B, C, H, W] in [0, 1].
        brightness: max brightness jitter factor (symmetric around 1).
        contrast:   max contrast jitter factor.
        saturation: max saturation jitter factor.
    """
    jitter = TFv2.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
    )
    return jitter(image)


def random_hflip_image_and_boxes(
    image: Tensor,         # [B, C, H, W]
    boxes_canon: Tensor,   # [B, N, 4]  (cx, cy, w, h) in canonical space
    p: float,
) -> tuple[Tensor, Tensor]:
    """Horizontally flip images + canonical boxes with per-sample probability ``p``.

    Boxes are in padded-square canonical space where ``cx`` lives in
    ``[-1, 1]``. A horizontal flip negates ``cx`` and leaves ``cy, w, h``
    unchanged. Padded sentinel boxes ``(0, 0, 0, 0)`` are invariant under
    this transform.

    Returns the (possibly-flipped) image and boxes. When ``p == 0`` the inputs
    are returned unchanged.
    """
    if p <= 0.0:
        return image, boxes_canon
    B = image.shape[0]
    flip_mask = torch.bernoulli(
        torch.full((B,), float(p), device=image.device)
    ).bool()
    if not flip_mask.any():
        return image, boxes_canon

    image_out = image.clone()
    image_out[flip_mask] = torch.flip(image[flip_mask], dims=(-1,))

    boxes_out = boxes_canon.clone()
    boxes_out[flip_mask, :, 0] = -boxes_canon[flip_mask, :, 0]
    return image_out, boxes_out

