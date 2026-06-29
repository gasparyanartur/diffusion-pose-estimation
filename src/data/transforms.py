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


def bboxes_xywh_from_coco_anns(anns: list[dict]) -> Tensor:
    """Parse the ``bbox`` field of COCO annotations into ``[P, 4]`` (x, y, w, h).

    Annotations missing a ``bbox`` field are emitted as ``(0, 0, 0, 0)``. The
    caller is expected to keep a separate validity mask (e.g. ``gt_box_labeled``)
    if downstream code needs to distinguish missing boxes from zero-sized ones.
    """
    if not anns:
        return torch.zeros(0, 4, dtype=torch.float32)
    rows = [a.get("bbox", [0.0, 0.0, 0.0, 0.0]) for a in anns]
    return torch.tensor(rows, dtype=torch.float32)


def boxes_xywh_to_padded_canonical(
    boxes_xywh: Tensor,        # [..., 4]  pixel (x, y, w, h) in original image
    original_h: int | float,
    original_w: int | float,
) -> Tensor:
    """Convert pixel xywh boxes to padded-square canonical ``(cx, cy, w, h)``.

    The padded canvas is built by ``resize_and_pad_to_square``: longest side
    scaled to ``S`` then padded on the right / bottom. After this remapping,
    coordinates lie in canonical space ``[-1, 1]`` for centres and ``[0, 2]``
    for widths / heights. The padding offset is zero (top-left aligned), so
    ``S`` cancels out and only ``max(original_h, original_w)`` matters.
    """
    out = boxes_xywh.clone().float()
    side = float(max(int(original_h), int(original_w)))
    x = boxes_xywh[..., 0]
    y = boxes_xywh[..., 1]
    w = boxes_xywh[..., 2]
    h = boxes_xywh[..., 3]
    out[..., 0] = (x + w / 2.0) / side * 2.0 - 1.0
    out[..., 1] = (y + h / 2.0) / side * 2.0 - 1.0
    out[..., 2] = w / side * 2.0
    out[..., 3] = h / side * 2.0
    return out


def boxes_padded_canonical_to_pixel_xywh(
    boxes_canon: Tensor,   # [..., 4]  (cx, cy, w, h) canonical
    img_hw: Tensor,        # [..., 2]  (H, W) original pixel
) -> Tensor:
    """Inverse of :func:`boxes_xywh_to_padded_canonical` (batched).

    The padded canvas side is ``max(H, W)``. We map canonical centres back to
    pixel centres on that canvas (origin at the top-left of the original
    image, so the canvas pixel coords coincide with the original-image pixel
    coords). Returns ``(x, y, w, h)`` in original-image pixel space.

    ``img_hw`` broadcasts against ``boxes_canon`` from the left: a typical
    call is ``boxes_canon=[B, N, 4]`` and ``img_hw=[B, 2]`` which is
    promoted to ``[B, 1, 2]``.
    """
    boxes_canon = boxes_canon.float()
    img_hw = img_hw.float()
    side = img_hw.amax(dim=-1).clamp_min(1.0)               # [...]
    while side.ndim < boxes_canon.ndim - 1:
        side = side.unsqueeze(-1)                            # broadcast
    cx = (boxes_canon[..., 0] + 1.0) / 2.0 * side
    cy = (boxes_canon[..., 1] + 1.0) / 2.0 * side
    w  = boxes_canon[..., 2] / 2.0 * side
    h  = boxes_canon[..., 3] / 2.0 * side
    x = cx - w / 2.0
    y = cy - h / 2.0
    return torch.stack([x, y, w, h], dim=-1)
