"""Cached DINO feature extractor.

Loads precomputed DINOv3 ConvNeXt features from per-image ``.pt`` files
produced by ``scripts/precompute_dino.py --backbone-type convnext``.

Cache contents per image:
    spatial:   stage1 features  — (96,  H/4,  W/4)
    semantic:  stage4 features  — (768, H/32, W/32)
    global:    pooled vector    — (768,)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig

# Architectural constants for facebook/dinov3-convnext-small.
_SPATIAL_DIM: int = 96
_HIDDEN_DIM: int = 768


class CachedDinoFeatureExtractor(BaseFeatureExtractor):
    """Load precomputed DINO features by image id.

    Spatial dimensions vary per image (aspect-ratio-preserving resize), so
    ``"spatial"`` and ``"semantic"`` are emitted as ``list[Tensor]``.
    ``"global"`` is stacked into a single ``(B, 768)`` tensor.
    """

    SPATIAL_DIM: int = _SPATIAL_DIM
    HIDDEN_DIM: int = _HIDDEN_DIM

    def __init__(
        self,
        model_name: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        cache_paths: list[str | Path] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.spatial_dim = _SPATIAL_DIM
        self.hidden_dim = _HIDDEN_DIM

        self._cache: dict[int, dict] = {}
        self._cache_dirs: list[Path] = []
        if cache_paths:
            self.load_cache(*cache_paths)

    def load_cache(self, *paths: str | Path) -> None:
        """Register one or more cache directories to load images from on demand."""
        for path in paths:
            p = Path(path)
            if not p.exists():
                if p.suffix == ".pt":
                    candidate = p.with_suffix("")
                    if candidate.is_dir():
                        logging.info(
                            "CachedDinoFeatureExtractor: '%s' not found; using directory '%s'",
                            p, candidate,
                        )
                        p = candidate
                    else:
                        logging.warning(
                            "CachedDinoFeatureExtractor: cache path not found: %s", p
                        )
                        continue
                else:
                    logging.warning(
                        "CachedDinoFeatureExtractor: cache directory not found: %s", p
                    )
                    continue
            if not p.is_dir():
                raise RuntimeError(
                    f"Cache path {p} is a file, not a directory."
                )
            self._cache_dirs.append(p)

    def _load_one(self, image_id: int) -> dict:
        if image_id not in self._cache:
            if not self._cache_dirs:
                raise RuntimeError(
                    "Feature cache not loaded; call load_cache() first."
                )
            for cache_dir in self._cache_dirs:
                pt_path = cache_dir / f"{image_id}.pt"
                if pt_path.exists():
                    self._cache[image_id] = torch.load(
                        pt_path, map_location="cpu", weights_only=False
                    )
                    break
            else:
                raise RuntimeError(
                    f"Image id {image_id} not in any cache directory: "
                    + ", ".join(str(d) for d in self._cache_dirs)
                )
        return self._cache[image_id]

    def forward(self, image_ids: list[int]) -> dict[str, Any]:
        device = self._infer_device()
        spatial: list[Tensor] = []
        semantic: list[Tensor] = []
        global_vecs: list[Tensor] = []
        for iid in image_ids:
            entry = self._load_one(iid)
            spatial.append(entry["spatial"].to(device))
            semantic.append(entry["semantic"].to(device))
            global_vecs.append(entry["global"].to(device))
        return {
            "spatial": spatial,
            "semantic": semantic,
            "global": torch.stack(global_vecs),
        }

    def _infer_device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cpu")


class CachedDinoFeatureExtractorConfig(BaseFeatureExtractorConfig):
    """Polymorphic config that builds :class:`CachedDinoFeatureExtractor`."""

    model_name: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
    cache_paths: list[str] = []

    def build(self) -> CachedDinoFeatureExtractor:
        return CachedDinoFeatureExtractor(
            model_name=self.model_name,
            cache_paths=list(self.cache_paths) if self.cache_paths else None,
        )
