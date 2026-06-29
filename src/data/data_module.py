"""Lightning DataModules for pose estimation.

The DataConfig.build() factory returns a concrete PoseDataModule. The training
module reads the resolved annotation paths and prediction-cache paths from the
data module so the lightning module never has to know the on-disk layout.
"""

from __future__ import annotations

import logging
from abc import ABC
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

from src.data.datasets import PoseDataset, SetPoseDataset, set_pose_collate

log = logging.getLogger(__name__)


def resize_and_pad_to_square(img: Tensor, size: int, mask_value: float = 0.0) -> tuple[Tensor, Tensor]:
    """Resize longest side to `size`, pad the shorter side to make a square.

    Args:
        img:  float [3, H, W]
        size: target side length
        mask_value: value to fill the padding area

    Returns:
        image:        [3, size, size]  float, zero-padded
        padding_mask: [size, size]    1 = real pixel, 0 = padding
    """
    _, h, w = img.shape
    scale = size / max(h, w)
    new_h = round(h * scale)
    new_w = round(w * scale)
    img = TF.resize(img, [new_h, new_w], interpolation=InterpolationMode.BICUBIC, antialias=True)
    # pad right and bottom to reach size × size
    padded = TF.pad(img, [0, 0, size - new_w, size - new_h], fill=mask_value)
    mask = torch.zeros(size, size, dtype=torch.float32, device=img.device)
    mask[:new_h, :new_w] = 1.0
    return padded, mask


def _boxes_to_padded_canonical_batch(
    boxes_xywh_pixel: Tensor,  # [B, P, 4]
    img_hw: Tensor,            # [B, 2]   (H, W) original
) -> Tensor:
    """Vectorised batched form of :func:`boxes_xywh_to_padded_canonical`.

    Each row uses its own original ``max(H, W)``. Padding entries (all zeros)
    stay all zeros after this transform (they get mapped to ``(-1, -1, 0, 0)``
    which is still a sentinel-only region; the ``gt_box_labeled`` mask is the
    source of truth for which slots are real).
    """
    side = img_hw.amax(dim=-1).clamp_min(1.0).unsqueeze(-1)  # [B, 1]
    x = boxes_xywh_pixel[..., 0]
    y = boxes_xywh_pixel[..., 1]
    w = boxes_xywh_pixel[..., 2]
    h = boxes_xywh_pixel[..., 3]
    cx = (x + w / 2.0) / side * 2.0 - 1.0
    cy = (y + h / 2.0) / side * 2.0 - 1.0
    cw = w / side * 2.0
    ch = h / side * 2.0
    return torch.stack([cx, cy, cw, ch], dim=-1)


class PoseDataModule(pl.LightningDataModule, ABC):
    """Contract: training module reads these attributes after construction."""

    val_ann_path: Path
    test_ann_path: Path
    val_pred_cache_resolved: str | None
    test_pred_cache_resolved: str | None
    num_keypoints: int
    batch_size: int
    num_workers: int
    ap_batch_size: int


class CrowdPoseDataModule(PoseDataModule):
    """CrowdPose train/val/test loaders backed by :class:`PoseDataset`."""

    def __init__(
        self,
        train_ann: Path,
        val_ann: Path,
        test_ann: Path,
        img_dir: Path,
        val_pred_cache: str | None,
        test_pred_cache: str | None,
        batch_size: int,
        num_workers: int,
        ap_batch_size: int,
        num_keypoints: int = 14,
        subset_frac: float = 1.0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        target_format: Literal["refinement", "set"] = "refinement",
        image_h: int | None = 256,
        image_w: int | None = 192,
        max_persons: int | None = None,
        max_size: int | None = 512,
    ) -> None:
        super().__init__()
        self.train_ann_path = Path(train_ann)
        self.val_ann_path = Path(val_ann)
        self.test_ann_path = Path(test_ann)
        self.img_dir = Path(img_dir)
        self.val_pred_cache_resolved = self._resolve_cache(val_pred_cache, "val_pred_cache")
        self.test_pred_cache_resolved = self._resolve_cache(test_pred_cache, "test_pred_cache")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ap_batch_size = ap_batch_size
        self.num_keypoints = num_keypoints
        self.subset_frac = subset_frac
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.target_format = target_format
        self.image_h = image_h
        self.image_w = image_w
        self.max_persons = max_persons
        self.max_size = max_size

        self._train_ds: PoseDataset | None = None
        self._val_ds: PoseDataset | None = None
        self._test_ds: PoseDataset | None = None

    @staticmethod
    def _resolve_cache(raw: str | None, label: str) -> str | None:
        if not raw:
            return None
        path = Path(raw)
        if not path.exists():
            log.warning(
                "%s not found at %s — AP eval will be skipped for this split.",
                label, path,
            )
            return None
        return str(path)

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is None:
            self._train_ds = PoseDataset.from_path(
                self.train_ann_path, img_root=self.img_dir, num_keypoints=self.num_keypoints,
            )
            self._maybe_subset(self._train_ds, "train")
            self._maybe_filter_persons(self._train_ds, "train")
        if self._val_ds is None:
            self._val_ds = PoseDataset.from_path(
                self.val_ann_path, img_root=self.img_dir, num_keypoints=self.num_keypoints,
            )
            self._maybe_subset(self._val_ds, "val")
            self._maybe_filter_persons(self._val_ds, "val")
        if self._test_ds is None:
            self._test_ds = PoseDataset.from_path(
                self.test_ann_path, img_root=self.img_dir, num_keypoints=self.num_keypoints,
            )
            self._maybe_subset(self._test_ds, "test")
            self._maybe_filter_persons(self._test_ds, "test")

    def _maybe_subset(self, ds: PoseDataset, split: str) -> None:
        if self.subset_frac >= 1.0:
            return
        n_full = len(ds.samples)
        n_keep = max(1, round(n_full * self.subset_frac))
        ds.samples = ds.samples[:n_keep]
        log.warning(
            "subset_frac=%.4g: %s split reduced %d -> %d samples.",
            self.subset_frac, split, n_full, n_keep,
        )

    def _maybe_filter_persons(self, ds: PoseDataset, split: str) -> None:
        if self.max_persons is None:
            return
        n_full = len(ds.samples)
        ds.samples = [
            s for s in ds.samples
            if s["keypoints"].shape[0] <= self.max_persons
        ]
        n_keep = len(ds.samples)
        log.warning(
            "max_persons=%d: %s split reduced %d -> %d samples.",
            self.max_persons, split, n_full, n_keep,
        )

    def _resolve_target_size(self) -> tuple[int, int] | None:
        h, w = self.image_h, self.image_w
        if h is None and w is None:
            return None
        if h is None:
            h = w
        if w is None:
            w = h
        return (h, w)  # type: ignore[return-value]

    def _worker_kwargs(self) -> dict:
        """Keyword arguments shared by all DataLoader constructions."""
        kwargs: dict = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def _make_loader(self, ds: PoseDataset, shuffle: bool, drop_last: bool) -> DataLoader:
        if self.target_format == "set":
            wrapped = SetPoseDataset(ds)
            return DataLoader(
                wrapped,
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=set_pose_collate,
                **self._worker_kwargs(),
            )
        return ds.create_dataloader(
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            load_images=False,
            pin_memory=self.pin_memory,
        )

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        """Resize and normalize the per-sample uint8 images as a GPU batch."""
        if "image" not in batch or not isinstance(batch["image"], list):
            return batch

        imgs: list[Tensor] = batch.pop("image")  # list of uint8 [C, H, W] on device
        target_size = self._resolve_target_size()
        normalized = []
        for img in imgs:
            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)       # grayscale → 3-channel
            normalized.append(img.float().div_(255.0))

        if target_size is not None:
            batch["image"] = torch.stack([
                TF.resize(img, list(target_size), interpolation=InterpolationMode.BILINEAR, antialias=True)
                for img in normalized
            ])  # [B, 3, H, W]
        elif self.max_size is not None:
            pairs = [resize_and_pad_to_square(img, self.max_size) for img in normalized]
            batch["image"]        = torch.stack([p for p, _ in pairs])   # [B, 3, S, S]
            batch["padding_mask"] = torch.stack([m for _, m in pairs])   # [B, S, S]
            if "gt_boxes_pixel" in batch:
                batch["gt_boxes_canon"] = _boxes_to_padded_canonical_batch(
                    batch["gt_boxes_pixel"], batch["img_hw"],
                )
        else:
            batch["image"] = normalized
        return batch

    def train_dataloader(self) -> DataLoader:
        self.setup()
        assert self._train_ds is not None
        return self._make_loader(self._train_ds, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        self.setup()
        assert self._val_ds is not None
        return self._make_loader(self._val_ds, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        self.setup()
        assert self._test_ds is not None
        return self._make_loader(self._test_ds, shuffle=False, drop_last=False)
