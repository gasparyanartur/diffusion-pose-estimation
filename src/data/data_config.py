"""Polymorphic :class:`DataConfig` that builds a :class:`PoseDataModule`."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Literal

from hydra.utils import to_absolute_path
from pydantic import field_validator

from src.base_config import BaseConfig
from src.data.data_module import CrowdPoseDataModule, PoseDataModule


class DataConfig(BaseConfig):
    """Base config for any pose dataset. Override ``build()`` in subclasses."""

    num_workers: int = 4
    ap_batch_size: int = 256
    batch_size: int = 64
    num_keypoints: int = 14
    subset_frac: float = 1.0
    """Fraction of each split's samples to keep (0 < x <= 1). 1.0 = full dataset."""

    @field_validator("subset_frac")
    @classmethod
    def _validate_subset_frac(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"subset_frac must be in (0, 1], got {v}")
        return v

    @abstractmethod
    def build(self, pin_memory: bool = False) -> PoseDataModule:
        ...


class CrowdPoseDataConfig(DataConfig):
    """CrowdPose paths + loader settings."""

    root: str = "data"
    train_ann: str
    val_ann: str
    test_ann: str
    img_dir: str
    val_pred_cache: str | None = None
    test_pred_cache: str | None = None

    target_format: Literal["refinement", "set"] = "refinement"
    image_h: int = 256
    image_w: int = 192

    def build(self, pin_memory: bool = False) -> CrowdPoseDataModule:
        root = Path(to_absolute_path(self.root))
        return CrowdPoseDataModule(
            train_ann=root / self.train_ann,
            val_ann=root / self.val_ann,
            test_ann=root / self.test_ann,
            img_dir=root / self.img_dir,
            val_pred_cache=to_absolute_path(self.val_pred_cache) if self.val_pred_cache else None,
            test_pred_cache=to_absolute_path(self.test_pred_cache) if self.test_pred_cache else None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            ap_batch_size=self.ap_batch_size,
            num_keypoints=self.num_keypoints,
            subset_frac=self.subset_frac,
            pin_memory=pin_memory,
            target_format=self.target_format,
            image_h=self.image_h,
            image_w=self.image_w,
        )
