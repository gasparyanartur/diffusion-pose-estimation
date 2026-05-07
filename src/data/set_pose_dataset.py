"""Set-prediction dataset wrapper and collate.

Wraps a :class:`PoseDataset` to:
- resize the image to a fixed (H, W);
- map keypoints into canonical $(-1, 1)$ space (see ``docs/assumptions.md``);
- pack per-image GT into fixed-shape tensors (P_max-padded by the collate);
- compute a per-person scale from the labeled-keypoint extent.

Output keys (one sample):
    image           [3, H, W]   float in [0, 1]
    gt_keypoints    [P, K, 2]   canonical (-1, 1)
    gt_labeled      [P, K]      {0, 1}
    gt_visible      [P, K]      {0, 1}
    person_scale    [P]         canonical-space extent
    image_id        int
    img_path        str
    img_hw          [2]         original (H, W) before resize, in pixels

Output keys (after ``set_pose_collate``):
    image, image_id, img_path, img_hw       stacked / list
    gt_keypoints    [B, P_max, K, 2]
    gt_labeled      [B, P_max, K]
    gt_visible      [B, P_max, K]
    person_scale    [B, P_max]
    person_mask     [B, P_max]              {0, 1}
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.datasets import PoseDataset
from src.data.io import read_image_pil


def _load_and_resize_image(
    img_path: str, target_h: int, target_w: int,
) -> tuple[Tensor, int, int]:
    pil = read_image_pil(img_path)
    original_w, original_h = pil.size
    pil = pil.resize((target_w, target_h))
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return image, original_h, original_w


def compute_person_scale(
    keypoints_canonical: Tensor,  # [P, K, 2]   canonical
    labeled: Tensor,              # [P, K]      {0, 1}
    floor: float = 0.1,
) -> Tensor:
    """Per-person scale [P] from labeled-keypoint extent in canonical space.

    Falls back to ``floor`` when fewer than two joints are labeled.
    """
    P = keypoints_canonical.shape[0]
    scale = torch.full((P,), floor, dtype=torch.float32)
    for p in range(P):
        mask = labeled[p].bool()
        if mask.sum() < 2:
            continue
        kp = keypoints_canonical[p][mask]
        extent = (kp.max(dim=0).values - kp.min(dim=0).values).max().item()
        scale[p] = max(extent, floor / 2.0)
    return scale


class SetPoseDataset(Dataset):
    """Set-format wrapper around :class:`PoseDataset`."""

    def __init__(
        self,
        base: PoseDataset,
        image_h: int,
        image_w: int,
    ) -> None:
        self.base = base
        self.image_h = image_h
        self.image_w = image_w

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base.samples[idx]
        image, original_h, original_w = _load_and_resize_image(
            sample["img_path"], self.image_h, self.image_w,
        )

        keypoints_pixel = sample["keypoints"].float().clone()       # [P, K, 2]
        visibility_raw = sample["visibility"].float().clone()        # [P, K] {0,1,2}

        # canonical = pixel / [W, H] * 2 - 1
        keypoints_canon = keypoints_pixel.clone()
        keypoints_canon[..., 0] = keypoints_pixel[..., 0] / float(original_w) * 2.0 - 1.0
        keypoints_canon[..., 1] = keypoints_pixel[..., 1] / float(original_h) * 2.0 - 1.0

        labeled = (visibility_raw > 0).float()
        visible = (visibility_raw == 2).float()
        person_scale = compute_person_scale(keypoints_canon, labeled)

        return {
            "image": image,
            "gt_keypoints": keypoints_canon,
            "gt_labeled": labeled,
            "gt_visible": visible,
            "person_scale": person_scale,
            "image_id": int(sample["image_id"]),
            "img_path": sample["img_path"],
            "img_hw": torch.tensor([original_h, original_w], dtype=torch.float32),
        }


def set_pose_collate(batch: list[dict]) -> dict:
    """Stack images, pad people to per-batch P_max, emit person_mask."""
    images = torch.stack([s["image"] for s in batch], dim=0)
    P_max = max(s["gt_keypoints"].shape[0] for s in batch)
    P_max = max(P_max, 1)
    K = batch[0]["gt_keypoints"].shape[1]
    B = len(batch)

    gt_keypoints = torch.zeros(B, P_max, K, 2)
    gt_labeled = torch.zeros(B, P_max, K)
    gt_visible = torch.zeros(B, P_max, K)
    person_scale = torch.zeros(B, P_max)
    person_mask = torch.zeros(B, P_max)
    img_hw = torch.zeros(B, 2)

    for i, s in enumerate(batch):
        P = s["gt_keypoints"].shape[0]
        if P > 0:
            gt_keypoints[i, :P] = s["gt_keypoints"]
            gt_labeled[i, :P] = s["gt_labeled"]
            gt_visible[i, :P] = s["gt_visible"]
            person_scale[i, :P] = s["person_scale"]
            person_mask[i, :P] = 1.0
        img_hw[i] = s["img_hw"]

    return {
        "image": images,
        "gt_keypoints": gt_keypoints,
        "gt_labeled": gt_labeled,
        "gt_visible": gt_visible,
        "person_scale": person_scale,
        "person_mask": person_mask,
        "img_hw": img_hw,
        "image_id": torch.tensor([s["image_id"] for s in batch], dtype=torch.long),
        "img_path": [s["img_path"] for s in batch],
    }
