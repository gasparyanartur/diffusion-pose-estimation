from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor

from src.base_config import BaseConfig


class BoxDetector(ABC):
    """Predict person bounding boxes for a batch of images.

    Inputs are file paths so that each backend can handle its own loading
    format (e.g. Detectron2 expects BGR numpy arrays, our model wants RGB tensors).

    Returns:
        boxes_xywh: (B, N, 4)  pixel-space [x, y, w, h], zero-padded to max N.
        scores:     (B, N)     confidence scores in [0, 1], zero-padded.
    """

    @abstractmethod
    def predict(
        self,
        image_paths: list[Path],
        image_ids: list[int],
    ) -> dict[str, Tensor]:
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...


class BoxDetectorConfig(BaseConfig, ABC):
    @abstractmethod
    def build(self) -> BoxDetector:
        ...
