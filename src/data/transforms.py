"""Pure-function keypoint transforms shared across dataset classes."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def keypoints_from_coco_anns(
    anns: list[dict],
    num_keypoints: int,
) -> tuple[Tensor, Tensor]:
    """Parse raw COCO annotation dicts into keypoint and visibility tensors.

    Args:
        anns: List of COCO annotation dicts, each containing a flat
              ``"keypoints"`` list of length ``num_keypoints * 3``
              (x, y, v per joint).
        num_keypoints: Number of keypoints per person.

    Returns:
        kpts: Float tensor ``[N, K, 2]`` — pixel (x, y) coordinates.
        vis:  Float tensor ``[N, K]``    — raw visibility flags ``{0, 1, 2}``.
    """
    arr = np.array(
        [a["keypoints"] for a in anns], dtype=np.float32
    ).reshape(len(anns), num_keypoints, 3)
    return torch.from_numpy(arr[:, :, :2]), torch.from_numpy(arr[:, :, 2])


def pixel_to_canonical(
    kpts_pixel: Tensor,  # [..., 2]  (x, y) in pixels
    original_h: int | float,
    original_w: int | float,
) -> Tensor:
    """Convert pixel keypoints to canonical ``(-1, 1)`` space.

    ``canonical_x = x / W * 2 - 1``,  ``canonical_y = y / H * 2 - 1``.
    Preserves the input shape; operates on the last dimension.
    """
    out = kpts_pixel.clone().float()
    out[..., 0] = kpts_pixel[..., 0] / float(original_w) * 2.0 - 1.0
    out[..., 1] = kpts_pixel[..., 1] / float(original_h) * 2.0 - 1.0
    return out


def visibility_flags(vis_raw: Tensor) -> tuple[Tensor, Tensor]:
    """Split a raw COCO visibility tensor into labeled and visible masks.

    Args:
        vis_raw: Float or int tensor ``[..., K]`` with values in ``{0, 1, 2}``.
                 0 = absent, 1 = occluded (labeled but not visible), 2 = visible.

    Returns:
        labeled: Float ``{0, 1}`` — joint is annotated (vis > 0).
        visible: Float ``{0, 1}`` — joint is unoccluded (vis == 2).
    """
    return (vis_raw > 0).float(), (vis_raw == 2).float()
