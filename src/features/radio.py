"""NVIDIA C-RADIO feature extractors.

C-RADIO models (nvidia/C-RADIOv3-B, nvidia/C-RADIOv4-H, …) return a tuple of
``(summary, spatial_features)``:

    summary          [B, C]     — global CLS-like embedding (not used here)
    spatial_features [B, T, D]  — flat patch tokens (T = G_H * G_W)

The extractor reshapes the flat tokens back to a 2-D spatial grid
``[B, D, G_H, G_W]`` (G = input_size // patch_size) and returns it under the
``"spatial"`` key, matching the contract expected by the rest of the codebase.

Normalisation is handled internally by the model's ``input_conditioner``;
upstream code should feed raw ``[0, 1]`` tensors (same contract as all other
extractors).  ``trust_remote_code=True`` is required by NVIDIA's custom model
class.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.embedders.spatial_attention import make_patch_centers
from src.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig


_PATCH_SIZE = 16  # all current C-RADIO models use patch size 16

_C_RADIO_V3_B_MODEL = "nvidia/C-RADIOv3-B"
_C_RADIO_V4_H_MODEL = "nvidia/C-RADIOv4-H"


class RadioFeatureExtractor(BaseFeatureExtractor):
    """Generic C-RADIO feature extractor.

    Loads any RADIO-family model from HuggingFace, applies CLIP normalisation
    internally, and returns spatial patch features as ``[B, D, H/patch, W/patch]``.

    Args:
        model_name:  HuggingFace model ID.
        patch_size:  Patch size in pixels (16 for all current C-RADIO variants).
        frozen:      If True, all backbone parameters are frozen.
    """

    def __init__(
        self,
        model_name: str,
        patch_size: int = _PATCH_SIZE,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self._patch_size = patch_size
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        if frozen:
            for p in self._model.parameters():
                p.requires_grad_(False)

        # Introspect spatial feature channels via a single CPU dummy forward.
        # Pass raw zeros — the model's input_conditioner handles normalisation.
        self._model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            _, spatial = self._model(dummy)  # spatial: [1, T, D]
            self._out_channels: int = spatial.shape[-1]

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def get_output_size(self, input_size: int) -> int:
        return input_size // self._patch_size

    def forward(
        self,
        image: Tensor | None = None,
        image_ids: list[int] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        assert image is not None, "RadioFeatureExtractor requires image tensor"
        B, _, H, W = image.shape
        G_H = H // self._patch_size
        G_W = W // self._patch_size

        _, spatial = self._model(image)                         # [B, T, D]
        spatial = spatial.permute(0, 2, 1).reshape(B, self._out_channels, G_H, G_W)
        return {
            "spatial": spatial,
            "patch_centers": make_patch_centers(G_H, G_W, image.device, image.dtype),
        }


class RadioFeatureExtractorConfig(BaseFeatureExtractorConfig):
    model_name: str
    patch_size: int = _PATCH_SIZE
    frozen: bool = True

    def build(self) -> RadioFeatureExtractor:
        return RadioFeatureExtractor(
            model_name=self.model_name,
            patch_size=self.patch_size,
            frozen=self.frozen,
        )


class CRadioV3BFeatureExtractorConfig(BaseFeatureExtractorConfig):
    """C-RADIOv3-B (90 M params, ViT-B/16).  Suitable for most GPU budgets."""

    frozen: bool = True

    def build(self) -> RadioFeatureExtractor:
        return RadioFeatureExtractor(
            model_name=_C_RADIO_V3_B_MODEL,
            patch_size=_PATCH_SIZE,
            frozen=self.frozen,
        )


class CRadioV4HFeatureExtractorConfig(BaseFeatureExtractorConfig):
    """C-RADIOv4-H (653 M params, ViT-H/16).  May not fit on smaller GPUs."""

    frozen: bool = True

    def build(self) -> RadioFeatureExtractor:
        return RadioFeatureExtractor(
            model_name=_C_RADIO_V4_H_MODEL,
            patch_size=_PATCH_SIZE,
            frozen=self.frozen,
        )
