"""Pose datasets and their config wrappers.

PoseDataset
    GT-only; one sample per image; N people; K keypoints.
PredictionDataset
    Wraps detector predictions with optional GT side-table.

Both pair with :class:`SceneGraphSampler` for uniform-N batches.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import torch
import torchvision.io as tvio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.base_config import BaseConfig
from src.data.io import load_gt_annotations, read_image
from src.data.sampler import SceneGraphSampler
from src.data.transforms import keypoints_from_coco_anns, pixel_to_canonical, visibility_flags


def _collate_with_images(batch: list[dict]) -> dict:
    """Like ``default_collate`` but keeps ``image`` as a list (variable spatial size)."""
    images = [s["image"] for s in batch] if "image" in batch[0] else None
    other = [{k: v for k, v in s.items() if k != "image"} for s in batch]
    result = torch.utils.data.default_collate(other)
    if images is not None:
        result["image"] = images
    return result


class PoseDataset(Dataset):
    """One sample per image; N people, K keypoints each."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        num_keypoints: int = 14,
        load_images: bool = False,
    ) -> None:
        self.samples = samples
        self.num_keypoints = num_keypoints
        self._load_images = load_images

    @classmethod
    def from_path(
        cls,
        ann_path: Path | str,
        img_root: Path | str | None = None,
        num_keypoints: int = 14,
    ) -> "PoseDataset":
        """Load from a COCO-style annotation JSON."""
        with open(ann_path) as fh:
            data = json.load(fh)

        img_root_path = Path(img_root) if img_root is not None else None
        image_paths: dict[int, str] = {
            img["id"]: (
                str(img_root_path / img["file_name"])
                if img_root_path is not None
                else img.get("file_name", "")
            )
            for img in data.get("images", [])
        }
        # Original image sizes from COCO metadata — avoids opening the file at runtime.
        image_sizes: dict[int, tuple[int, int]] = {
            img["id"]: (int(img["height"]), int(img["width"]))
            for img in data.get("images", [])
            if "height" in img and "width" in img
        }

        by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            if ann.get("num_keypoints", 0) > 0:
                by_image[ann["image_id"]].append(ann)

        samples = []
        for image_id, anns in by_image.items():
            kpts, vis = keypoints_from_coco_anns(anns, num_keypoints)
            samples.append({
                "image_id":   image_id,
                "img_path":   image_paths.get(image_id, ""),
                "img_hw":     image_sizes.get(image_id),  # (H, W) or None if missing
                "keypoints":  kpts,
                "visibility": vis,
            })

        return cls(samples=samples, num_keypoints=num_keypoints)

    def create_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        load_images: bool = False,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> DataLoader:
        self._load_images = load_images
        sampler = SceneGraphSampler(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
        return DataLoader(
            self,
            batch_sampler=sampler,
            collate_fn=_collate_with_images,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        if self._load_images:
            sample = dict(sample)
            sample["image"] = read_image(sample["img_path"]).float().div(255.0)
        return sample


class PredictionDataset(PoseDataset):
    """:class:`PoseDataset` extended with model predictions and optional GT."""

    def __init__(
        self,
        model_name: str,
        architecture_type: Literal["top-down", "bottom-up"],
        num_keypoints: int,
        predictions: list[dict[str, Any]],
        gt_by_image_id: dict[int, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.architecture_type = architecture_type
        self.num_keypoints = num_keypoints
        self.predictions = predictions
        self.has_bboxes: bool = bool(predictions) and "bboxes" in predictions[0]
        self._gt_by_image_id = gt_by_image_id
        self.has_gt: bool = gt_by_image_id is not None

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        ann_path: Path | str | None = None,
        **kwargs: Any,
    ) -> "PredictionDataset":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        kwargs.update({
            "model_name": data["model_name"],
            "architecture_type": data["architecture_type"],
            "num_keypoints": data["num_keypoints"],
            "predictions": data["predictions"],
        })
        if ann_path is not None:
            kwargs["gt_by_image_id"] = load_gt_annotations(ann_path)
        return cls(**kwargs)

    def create_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> DataLoader:
        sampler = SceneGraphSampler(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            group_key="pred_keypoints",
        )
        return DataLoader(
            self,
            batch_sampler=sampler,
            collate_fn=torch.utils.data.default_collate,
            num_workers=num_workers,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.predictions)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pred = self.predictions[idx]
        sample: dict[str, Any] = {
            "image_id":             pred["image_id"],
            "img_path":             pred.get("img_path", ""),
            "pred_keypoints":       torch.tensor(pred["keypoints"],       dtype=torch.float32),
            "pred_keypoint_scores": torch.tensor(pred["keypoint_scores"], dtype=torch.float32),
        }
        if self.has_bboxes:
            sample["bboxes"]      = torch.tensor(pred["bboxes"],      dtype=torch.float32)
            sample["bbox_scores"] = torch.tensor(pred["bbox_scores"], dtype=torch.float32)
        return sample

    def get_gt(self, image_id: int) -> dict[str, Any] | None:
        """Return GT keypoints / visibility for an image, or ``None`` if no GT loaded."""
        if not self.has_gt or self._gt_by_image_id is None:
            return None
        gt_anns = self._gt_by_image_id.get(image_id, [])
        if not gt_anns:
            return {
                "keypoints":  torch.zeros(0, self.num_keypoints, 2, dtype=torch.float32),
                "visibility": torch.zeros(0, self.num_keypoints,    dtype=torch.float32),
            }
        kpts, vis = keypoints_from_coco_anns(gt_anns, self.num_keypoints)
        return {"keypoints": kpts, "visibility": vis}


# ---------------------------------------------------------------------------
# Set-prediction dataset
# ---------------------------------------------------------------------------


def compute_person_scale(
    keypoints_canonical: Tensor,  # [P, K, 2]
    labeled: Tensor,              # [P, K]
    floor: float = 0.1,
) -> Tensor:
    """Per-person scale [P] from labeled-keypoint extent in canonical space."""
    mask = labeled.bool()                          # [P, K]
    enough = mask.sum(dim=1) >= 2                  # [P]
    kp_max = keypoints_canonical.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)  # [P, 2]
    kp_min = keypoints_canonical.masked_fill(~mask.unsqueeze(-1), float("inf")).amin(dim=1)   # [P, 2]
    extent = (kp_max - kp_min).amax(dim=1)         # [P]
    return torch.where(enough, extent.clamp(min=floor / 2.0), torch.full_like(extent, floor))


class SetPoseDataset(Dataset):
    """Set-format wrapper around :class:`PoseDataset`.

    Reads images via workers; maps keypoints to canonical space.
    Resize and normalize happen on the GPU in ``on_after_batch_transfer``.
    """

    def __init__(self, base: PoseDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base.samples[idx]

        image = tvio.read_image(sample["img_path"])  # uint8 [C, H, W]
        _, original_h, original_w = image.shape

        keypoints_canon = pixel_to_canonical(
            sample["keypoints"], original_h, original_w,
        )  # [P, K, 2]
        labeled, visible = visibility_flags(sample["visibility"].float())
        person_scale = compute_person_scale(keypoints_canon, labeled)

        return {
            "image":        image,
            "gt_keypoints": keypoints_canon,
            "gt_labeled":   labeled,
            "gt_visible":   visible,
            "person_scale": person_scale,
            "image_id":     int(sample["image_id"]),
            "img_path":     sample["img_path"],
            "img_hw":       torch.tensor([original_h, original_w], dtype=torch.float32),
        }


def set_pose_collate(batch: list[dict]) -> dict:
    """Pad people to per-batch P_max, emit person_mask.

    Images are kept as a list of uint8 tensors (variable spatial size).
    Resize and normalize happen on the GPU in ``on_after_batch_transfer``.
    """
    images = [s["image"] for s in batch]  # list[Tensor] uint8 [C, H, W] variable size
    P_max = max(s["gt_keypoints"].shape[0] for s in batch)
    P_max = max(P_max, 1)
    K = batch[0]["gt_keypoints"].shape[1]
    B = len(batch)

    gt_keypoints = torch.zeros(B, P_max, K, 2)
    gt_labeled   = torch.zeros(B, P_max, K)
    gt_visible   = torch.zeros(B, P_max, K)
    person_scale = torch.zeros(B, P_max)
    person_mask  = torch.zeros(B, P_max)
    img_hw       = torch.zeros(B, 2)

    for i, s in enumerate(batch):
        P = s["gt_keypoints"].shape[0]
        if P > 0:
            gt_keypoints[i, :P] = s["gt_keypoints"]
            gt_labeled[i, :P]   = s["gt_labeled"]
            gt_visible[i, :P]   = s["gt_visible"]
            person_scale[i, :P] = s["person_scale"]
            person_mask[i, :P]  = 1.0
        img_hw[i] = s["img_hw"]

    return {
        "image":        images,
        "gt_keypoints": gt_keypoints,
        "gt_labeled":   gt_labeled,
        "gt_visible":   gt_visible,
        "person_scale": person_scale,
        "person_mask":  person_mask,
        "img_hw":       img_hw,
        "image_id":     torch.tensor([s["image_id"] for s in batch], dtype=torch.long),
        "img_path":     [s["img_path"] for s in batch],
    }


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


class BaseDatasetConfig(BaseConfig):
    """Base config for any dataset. Override ``build()`` in subclasses."""

    num_keypoints: int = 14

    def build(self) -> Dataset:
        raise NotImplementedError


class PoseDatasetConfig(BaseDatasetConfig):
    """Config for :class:`PoseDataset` loaded from a COCO annotation file."""

    ann_path: str
    img_root: str | None = None

    def build(self) -> PoseDataset:
        return PoseDataset.from_path(
            ann_path=self.ann_path,
            img_root=self.img_root,
            num_keypoints=self.num_keypoints,
        )


class PredictionDatasetConfig(BaseDatasetConfig):
    """Config for :class:`PredictionDataset` loaded from a pickle cache."""

    cache_path: str
    ann_path: str | None = None

    def build(self) -> PredictionDataset:
        return PredictionDataset.from_path(
            path=self.cache_path,
            ann_path=self.ann_path,
        )
