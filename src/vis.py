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
    """Overlay predicted keypoints and bounding-boxes on an image.

    Parameters
    ----------
    img_tensor:
        RGB image as a ``uint8`` tensor of shape ``(C, H, W)``.
    keypoints:
        Keypoint coordinates of shape ``(N, K, 2)`` in ``(x, y)`` order,
        where *N* is the number of people and *K* is the number of joints.
        ``None`` skips keypoint drawing.
    bboxes:
        Bounding boxes of shape ``(N, 4)`` or ``(N, 5)`` in ``xyxy`` format.
        Only the first four values are used.  ``None`` skips bbox drawing.
    figsize:
        Figure size passed to :func:`matplotlib.pyplot.figure`.  Ignored when
        *ax* is provided.
    kpt_color:
        Matplotlib colour string for keypoint markers.
    kpt_txt_color:
        Matplotlib colour string for keypoint index text (when *numbered* is
        ``True``).
    bbox_color:
        Matplotlib colour string for bounding-box edges.
    kpt_size:
        Marker size for keypoints (in points).
    bbox_linewidth:
        Line width for bounding-box rectangles.
    ax:
        Existing :class:`~matplotlib.axes.Axes` to draw on.  A new figure is
        created when this is ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the visualisation was drawn.
    """
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
