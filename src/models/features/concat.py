"""ConcatFeatureExtractor: fuses multiple backbone feature maps into a token sequence.

For each child backbone:
    spatial [B, C, H, W]
      → add 2D sinusoidal PE
      → permute → [B, H*W, C]
      → LayerNorm(C)
      → Linear(C → dim)
      → add learned modality embedding [1, 1, dim]

All modality token sequences are concatenated along the sequence axis:
    output["spatial"]  →  [B, T, dim],  T = sum_i(H_i * W_i)

The extractor is recursive: each child is a full BaseFeatureExtractor, so
ConcatFeatureExtractors can be nested.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.models.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig


class ConcatFeatureExtractor(BaseFeatureExtractor):
    """Fuses multiple backbone feature maps into one token sequence.

    Args:
        backbones:   Child extractors. Each must expose ``out_channels``.
        dim:         Shared projection dimension for all modalities.
        feature_key: Key to read from each child's output dict.
    """

    def __init__(
        self,
        backbones: list[BaseFeatureExtractor],
        dim: int = 256,
        feature_key: str = "spatial",
    ) -> None:
        super().__init__()
        if not backbones:
            raise ValueError("ConcatFeatureExtractor requires at least one backbone")

        self.backbones = nn.ModuleList(backbones)
        self.dim = dim
        self.out_channels = dim
        self.feature_key = feature_key

        in_channels_list = [b.out_channels for b in backbones]
        self.norms = nn.ModuleList([nn.LayerNorm(c) for c in in_channels_list])
        self.projs = nn.ModuleList([nn.Linear(c, dim) for c in in_channels_list])
        self.modality_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, dim)) for _ in backbones
        ])

    def get_output_size(self, input_size: int) -> int:
        return sum(b.get_output_size(input_size) for b in self.backbones)

    def forward(
        self,
        image: Tensor | None = None,
        image_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        all_tokens: list[Tensor] = []
        all_centers: list[Tensor] = []
        for i, backbone in enumerate(self.backbones):
            feat_dict = backbone(image=image, image_ids=image_ids, **kwargs)
            feat = feat_dict[self.feature_key]
            if not isinstance(feat, Tensor):
                raise TypeError(
                    f"ConcatFeatureExtractor: backbone {i} returned a non-Tensor for "
                    f"key '{self.feature_key}'. list[Tensor] is not supported."
                )
            B, C, H, W = feat.shape
            tokens = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            tokens = self.norms[i](tokens)              # [B, H*W, C]
            tokens = self.projs[i](tokens)              # [B, H*W, dim]
            tokens = tokens + self.modality_embeds[i]   # [B, H*W, dim]
            all_tokens.append(tokens)
            all_centers.append(feat_dict["patch_centers"].reshape(-1, 2))

        return {
            "spatial": torch.cat(all_tokens, dim=1),       # [B, T, dim]
            "patch_centers": torch.cat(all_centers, dim=0), # [T, 2]
        }


class ConcatFeatureExtractorConfig(BaseFeatureExtractorConfig):
    """Config for :class:`ConcatFeatureExtractor`.

    ``backbones`` holds a list of :class:`~src.models.features.base.BaseFeatureExtractorConfig`
    subclass instances.  When loaded via Hydra, nested ``_target_`` nodes are
    resolved by :func:`~src.base_config.BaseConfig.from_hydra_config` before
    this config is constructed, so each element will already be an instantiated
    config object.
    """

    backbones: list[Any]
    dim: int = 256
    feature_key: str = "spatial"

    def build(self) -> ConcatFeatureExtractor:
        built: list[BaseFeatureExtractor] = []
        for i, cfg in enumerate(self.backbones):
            if not isinstance(cfg, BaseFeatureExtractorConfig):
                raise TypeError(
                    f"ConcatFeatureExtractorConfig.backbones[{i}] must be a "
                    f"BaseFeatureExtractorConfig; got {type(cfg).__name__}"
                )
            built.append(cfg.build())
        return ConcatFeatureExtractor(
            backbones=built,
            dim=self.dim,
            feature_key=self.feature_key,
        )
