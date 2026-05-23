"""Feature extractors: image-level → raw feature dict.

A feature extractor produces image-level features (e.g. multi-scale CNN feature
maps, transformer patch tokens, pooled global vectors). It does **not** know
about keypoints; that work happens in :mod:`src.embedders`.

Output is always a ``dict[str, Any]`` so different extractors can carry
different keys (e.g. ``"spatial"``, ``"semantic"``, ``"global"``). When
spatial dimensions vary across images, return a ``list[Tensor]``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
from torch import nn

from src.base_config import BaseConfig


class BaseFeatureExtractor(nn.Module):
    """Image-level feature extractor.

    Subclasses must implement :meth:`forward` returning a dict whose keys are
    documented per-extractor. The dict is the input to a
    :class:`~src.embedders.BaseFeatureEmbedder`.
    """

    @abstractmethod
    def forward(self, image_ids: list[int]) -> dict[str, Any]:
        ...


class BaseFeatureExtractorConfig(BaseConfig):
    """Polymorphic config; subclasses build their concrete extractor."""

    def build(self) -> BaseFeatureExtractor:
        raise NotImplementedError
