from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor, nn

from src.configs.base_config import BaseConfig


class BoxDetectorConfig(BaseConfig, ABC):
    @abstractmethod
    def build(self) -> BoxDetector:
        raise NotImplementedError


class BoxDetector(nn.Module, ABC):
    """Path-based interface retained only for external detector adapters."""

    @abstractmethod
    def predict(self, image_paths: list[Path], image_ids: list[int]) -> dict[str, Tensor]:
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...
