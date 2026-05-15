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
import torchvision.io as tvio
import torchvision.transforms.functional as TF
from torch import Tensor
from torchvision.transforms import InterpolationMode
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
        prefetch_factor: int = 2,
        target_format: Literal["refinement", "set"] = "refinement",
        image_h: int = 256,
        image_w: int = 192,
        max_persons: int | None = None,
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
            wrapped = SetPoseDataset(ds, image_h=self.image_h, image_w=self.image_w)
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

    def transfer_batch_to_device(self, batch: dict, device, dataloader_idx: int) -> dict:
        """Move batch to device, excluding raw_jpeg bytes to avoid byte-by-byte iteration.

        Lightning's apply_to_collection treats bytes as a Sequence and iterates
        byte-by-byte (~3.2M calls per batch at bs=32), causing ~5s overhead.
        Pop raw_jpeg before the transfer and restore it afterwards.
        """
        raw_jpeg = batch.pop("raw_jpeg", None)
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        if raw_jpeg is not None:
            batch["raw_jpeg"] = raw_jpeg
        return batch

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        """Decode raw JPEG bytes on GPU, resize, and normalize in one batched step."""
        if "raw_jpeg" not in batch:
            return batch

        raw_list: list[bytes] = batch.pop("raw_jpeg")
        target_h, target_w = batch.pop("target_hw").tolist()
        # Infer device from a tensor that was already transferred.
        device = batch["img_hw"].device

        decoded: list[Tensor] = []
        for raw in raw_list:
            # bytearray gives a writable buffer — avoids a PyTorch read-only warning.
            t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            img = tvio.decode_jpeg(t, device=device)  # uint8 [C, H, W] on device
            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)           # grayscale → 3-channel
            img = TF.resize(
                img, [target_h, target_w],
                interpolation=InterpolationMode.BILINEAR,
                antialias=False,
            )
            decoded.append(img)

        batch["image"] = torch.stack(decoded).float().div_(255.0)  # [B, 3, H, W]
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
