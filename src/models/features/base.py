from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
from torch import nn

from src.configs.base_config import BaseConfig


class BaseFeatureExtractor(nn.Module):
    @abstractmethod
    def forward(
        self,
        image: torch.Tensor | None = None,
        image_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    def get_output_size(self, input_size: int) -> int:
        ...


class BaseFeatureExtractorConfig(BaseConfig):
    def build(self) -> BaseFeatureExtractor:
        raise NotImplementedError
