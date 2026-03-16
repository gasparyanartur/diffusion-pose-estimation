"""
Visualization utilities for pose-estimation predictions.

This module provides lightweight helpers for inspecting predictions produced
by :class:`~src.data.PredictionDataset`.  All functions rely on Matplotlib and
operate on plain NumPy arrays or PyTorch tensors so they integrate naturally
into both notebook exploration and offline analysis scripts.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def show_img_with_keypoints(
    img_tensor: Tensor,
    keypoints: Optional[Tensor] = None,
    bboxes: Optional[Tensor] = None,
    figsize: tuple[float, float] = (5, 8),
    visibility: Optional[Tensor] = None,
    include_occluded: bool = True,
    kpt_color: str = "lime",
    kpt_txt_color: str = "black",
    bbox_color: str = "red",
    kpt_size: float = 6.0,
    bbox_linewidth: float = 1.5,
    numbered: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Overlay keypoints and bounding-boxes on an image; return the axes."""
    img_np: np.ndarray = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)

    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(img_np)

    if bboxes is not None:
        bboxes_np = bboxes.numpy() if isinstance(bboxes, Tensor) else np.asarray(bboxes)
        for box in bboxes_np:
            x1, y1, x2, y2 = box[:4]
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                edgecolor=bbox_color,
                facecolor="none",
                linewidth=bbox_linewidth,
                alpha=0.8,
            )
            ax.add_patch(rect)

    if keypoints is not None:
        kpts_np = keypoints.numpy() if isinstance(keypoints, Tensor) else np.asarray(keypoints)
        for i, person_kpts in enumerate(kpts_np):
            if visibility is not None:
                keep_mask = (visibility[i] > 0) if include_occluded else (visibility[i] > 1)
                person_kpts = person_kpts[keep_mask]
            
            xs, ys = person_kpts[:, 0], person_kpts[:, 1]
            ax.plot(xs, ys, "o", color=kpt_color, markersize=kpt_size, alpha=0.9)
            if numbered:
                for j, (x, y) in enumerate(zip(xs, ys)):
                    ax.text(x, y, str(j), color=kpt_txt_color, fontsize=8, ha="center", va="center")

    ax.axis("off")
    plt.tight_layout()
    return ax
