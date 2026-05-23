"""
Visualization utilities for pose-estimation predictions.

This module provides lightweight helpers for inspecting predictions produced
by :class:`~src.data.PredictionDataset`.  All functions rely on Matplotlib and
operate on plain NumPy arrays or PyTorch tensors so they integrate naturally
into both notebook exploration and offline analysis scripts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe; must be set before pyplot is imported
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


def save_prediction_plot(
    out_path: str | Path,
    image_chw: Tensor,           # [3, H, W]  float in [0, 1]
    pred_kpts_pixel: Tensor,     # [N, K, 2]  pixel space (x, y)
    pred_scores: Tensor,         # [N]        objectness score in [0, 1]
    gt_kpts_pixel: Tensor,       # [P, K, 2]  pixel space (x, y)
    gt_labeled: Tensor,          # [P, K]     {0, 1}
    top_k: int = 5,
) -> None:
    """Save a side-by-side GT / top-k-predictions PNG to *out_path*.

    Left panel: GT keypoints (green).
    Right panel: top-``top_k`` predicted slots ranked by *pred_scores*,
    coloured from yellow (highest) to red (lowest) so rank is visible at a
    glance.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    img_np = (image_chw.detach().cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ---- GT ----
    axes[0].imshow(img_np)
    axes[0].set_title("GT", fontsize=10)
    P = gt_kpts_pixel.shape[0]
    gt_kpts_np = gt_kpts_pixel.detach().cpu().float().numpy()
    gt_labeled_np = gt_labeled.detach().cpu().float().numpy()
    for p in range(P):
        mask = gt_labeled_np[p] > 0
        xs, ys = gt_kpts_np[p, mask, 0], gt_kpts_np[p, mask, 1]
        axes[0].plot(xs, ys, "o", color="lime", markersize=4, alpha=0.9)
    axes[0].axis("off")

    # ---- Predictions ----
    axes[1].imshow(img_np)
    axes[1].set_title(f"Predictions (top {top_k})", fontsize=10)

    scores_np = pred_scores.detach().cpu().float().numpy()
    order = np.argsort(scores_np)[::-1][:top_k]
    pred_kpts_np = pred_kpts_pixel.detach().cpu().float().numpy()

    cmap = matplotlib.colormaps["autumn"]  # yellow -> red for rank 0 -> top_k-1
    n_show = len(order)
    for rank, slot_idx in enumerate(order):
        color = cmap(rank / max(n_show - 1, 1))
        xs = pred_kpts_np[slot_idx, :, 0]
        ys = pred_kpts_np[slot_idx, :, 1]
        axes[1].plot(xs, ys, "o", color=color, markersize=4, alpha=0.85,
                     label=f"s={scores_np[slot_idx]:.2f}")
    if n_show > 0:
        axes[1].legend(fontsize=6, loc="upper right", framealpha=0.5)
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
