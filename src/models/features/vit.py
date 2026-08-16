"""ViT-B/16 feature extractor (trainable or pretrained).

Wraps ``torchvision.models.vit_b_16``, strips the classification head, and
returns the spatial patch tokens reshaped to a 2-D feature map under the
``"spatial"`` key — matching the contract of :class:`ResNetFeatureExtractor`.

Output:  ``[B, 768, H//16, W//16]``
Stride:  16  (patch size)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.common.positional import make_patch_centers
from src.models.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_OUT_CHANNELS = 768   # ViT-B hidden dimension
_PATCH_SIZE   = 16
_VIT_SIZE     = 224   # model's fixed input resolution


class ViTFeatureExtractor(BaseFeatureExtractor):
    """Headless ViT-B/16 that returns a spatial feature map.

    Applies ImageNet normalisation internally so upstream code feeds raw
    ``[0, 1]`` tensors (same contract as :class:`ResNetFeatureExtractor`).

    Args:
        pretrained: Load ``ViT_B_16_Weights.DEFAULT`` when ``True``; random
                    init when ``False``.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        import torchvision.models as tvm

        weights = tvm.ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = tvm.vit_b_16(weights=weights)
        # Keep all parts needed for the encoder; discard the head.
        self._conv_proj   = vit.conv_proj
        self._class_token = vit.class_token
        self._encoder     = vit.encoder
        self._patch_size  = _PATCH_SIZE

        self.register_buffer("_imagenet_mean", _IMAGENET_MEAN)
        self.register_buffer("_imagenet_std",  _IMAGENET_STD)

    @property
    def out_channels(self) -> int:
        return _OUT_CHANNELS

    def get_output_size(self, input_size: int) -> int:  # noqa: ARG002
        return _VIT_SIZE // _PATCH_SIZE  # always 14

    def forward(
        self,
        image: Tensor | None = None,
        image_ids: list[int] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        if image is None:
            raise ValueError("ViTFeatureExtractor requires `image`; got None")

        B, _, H, W = image.shape

        # ViT-B/16 has fixed 224×224 positional embeddings; resize if needed.
        if H != _VIT_SIZE or W != _VIT_SIZE:
            image = F.interpolate(image, size=(_VIT_SIZE, _VIT_SIZE), mode="bilinear", align_corners=False)

        x = (image - self._imagenet_mean) / self._imagenet_std

        # Patch embedding → [B, T, hidden]  (T = 14*14 = 196)
        G = _VIT_SIZE // _PATCH_SIZE  # 14
        x = self._conv_proj(x)
        x = x.reshape(B, _OUT_CHANNELS, G * G).permute(0, 2, 1)

        # Prepend class token, run encoder.
        cls = self._class_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self._encoder(x)          # [B, 1+T, hidden]

        # Drop class token, reshape to spatial grid.
        x = x[:, 1:]                  # [B, T, hidden]
        x = x.permute(0, 2, 1).reshape(B, _OUT_CHANNELS, G, G)
        return {
            "spatial": x,
            "patch_centers": make_patch_centers(G, G, x.device, x.dtype),
        }


class ViTFeatureExtractorConfig(BaseFeatureExtractorConfig):
    pretrained: bool = True

    def build(self) -> ViTFeatureExtractor:
        return ViTFeatureExtractor(pretrained=self.pretrained)
