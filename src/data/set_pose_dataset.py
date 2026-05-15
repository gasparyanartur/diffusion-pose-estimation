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

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.datasets import PoseDataset
from src.data.io import read_image_size
from src.data.transforms import pixel_to_canonical, visibility_flags


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

        # Get original image size from COCO metadata (no I/O).
        # Fall back to reading the file header (cached) if metadata is absent.
        stored_hw = sample.get("img_hw")
        if stored_hw is not None:
            original_h, original_w = stored_hw
        else:
            original_h, original_w = read_image_size(sample["img_path"])

        # Read raw JPEG bytes — decode happens on GPU in on_after_batch_transfer.
        with open(sample["img_path"], "rb") as fh:
            raw_jpeg = fh.read()  # Python bytes; not a tensor; won't be device-transferred

        keypoints_canon = pixel_to_canonical(
            sample["keypoints"], original_h, original_w,
        )  # [P, K, 2]
        labeled, visible = visibility_flags(sample["visibility"].float())
        person_scale = compute_person_scale(keypoints_canon, labeled)

        return {
            "raw_jpeg": raw_jpeg,
            "target_hw": torch.tensor([self.image_h, self.image_w], dtype=torch.long),
            "gt_keypoints": keypoints_canon,
            "gt_labeled": labeled,
            "gt_visible": visible,
            "person_scale": person_scale,
            "image_id": int(sample["image_id"]),
            "img_path": sample["img_path"],
            "img_hw": torch.tensor([original_h, original_w], dtype=torch.float32),
        }


def set_pose_collate(batch: list[dict]) -> dict:
    """Collect raw JPEG bytes, pad people to per-batch P_max, emit person_mask.

    Images are NOT decoded here; decoding happens on GPU in
    ``CrowdPoseDataModule.on_after_batch_transfer``.
    """
    raw_jpegs = [s["raw_jpeg"] for s in batch]   # list[bytes] — not moved to device
    target_hw = batch[0]["target_hw"]             # Tensor([H, W])
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
        "raw_jpeg": raw_jpegs,
        "target_hw": target_hw,
        "gt_keypoints": gt_keypoints,
        "gt_labeled": gt_labeled,
        "gt_visible": gt_visible,
        "person_scale": person_scale,
        "person_mask": person_mask,
        "img_hw": img_hw,
        "image_id": torch.tensor([s["image_id"] for s in batch], dtype=torch.long),
        "img_path": [s["img_path"] for s in batch],
    }
