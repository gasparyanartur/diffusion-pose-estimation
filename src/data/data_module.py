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
from torch.utils.data import DataLoader

from src.data.datasets import PoseDataset
from src.data.set_pose_dataset import SetPoseDataset, set_pose_collate

log = logging.getLogger(__name__)


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
        target_format: Literal["refinement", "set"] = "refinement",
        image_h: int = 256,
        image_w: int = 192,
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
        self.target_format = target_format
        self.image_h = image_h
        self.image_w = image_w

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
        if self._val_ds is None:
            self._val_ds = PoseDataset.from_path(
                self.val_ann_path, img_root=self.img_dir, num_keypoints=self.num_keypoints,
            )
            self._maybe_subset(self._val_ds, "val")
        if self._test_ds is None:
            self._test_ds = PoseDataset.from_path(
                self.test_ann_path, img_root=self.img_dir, num_keypoints=self.num_keypoints,
            )
            self._maybe_subset(self._test_ds, "test")

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

    def _make_loader(self, ds: PoseDataset, shuffle: bool, drop_last: bool) -> DataLoader:
        if self.target_format == "set":
            wrapped = SetPoseDataset(ds, image_h=self.image_h, image_w=self.image_w)
            return DataLoader(
                wrapped,
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=self.num_workers,
                collate_fn=set_pose_collate,
                pin_memory=self.pin_memory,
                persistent_workers=(self.num_workers > 0),
            )
        return ds.create_dataloader(
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            load_images=False,
            pin_memory=self.pin_memory,
        )

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
