"""ResNet family feature extractors (headless, trainable or pretrained).

Wraps torchvision ResNets, strips the global pool and FC head, and returns
the spatial feature map from the chosen layer under key ``"spatial"``.

Supported variants: ``"resnet18"``, ``"resnet34"``, ``"resnet50"``.
Output layer: ``"layer3"`` (stride 16) or ``"layer4"`` (stride 32).

Output channels by variant/layer:
    resnet18 / resnet34 : layer3 → 256,  layer4 → 512
    resnet50            : layer3 → 1024, layer4 → 2048
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor, nn

from src.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_VARIANT = Literal["resnet18", "resnet34", "resnet50"]
_LAYER   = Literal["layer3", "layer4"]

_OUT_CHANNELS: dict[str, dict[str, int]] = {
    "resnet18": {"layer3": 256, "layer4": 512},
    "resnet34": {"layer3": 256, "layer4": 512},
    "resnet50": {"layer3": 1024, "layer4": 2048},
}


class ResNetFeatureExtractor(BaseFeatureExtractor):
    """Headless ResNet that returns a spatial feature map.

    The module normalises images with ImageNet statistics internally so
    upstream code feeds raw ``[0, 1]`` tensors (same contract as
    :class:`~src.features.cnn.CNNFeatureExtractor`).

    Args:
        variant:      Which ResNet depth to use.
        output_layer: Which stage to read features from (``"layer3"`` or
                      ``"layer4"``).  ``"layer3"`` gives stride 16 and is
                      recommended for the backbone probe benchmark on 256×192
                      images (produces a 16×12 feature map).
        pretrained:   When ``True`` loads ImageNet weights via torchvision's
                      ``DEFAULT`` preset.  When ``False`` initialises randomly.
    """

    def __init__(
        self,
        variant: _VARIANT = "resnet18",
        output_layer: _LAYER = "layer3",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self._output_layer = output_layer
        self._out_channels = _OUT_CHANNELS[variant][output_layer]

        backbone = self._build_torchvision(variant, pretrained)
        # Keep only the stages up to and including output_layer.
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        if output_layer == "layer4":
            self.layer4 = backbone.layer4

        self.register_buffer("_imagenet_mean", _IMAGENET_MEAN)
        self.register_buffer("_imagenet_std",  _IMAGENET_STD)

    @staticmethod
    def _build_torchvision(variant: str, pretrained: bool):
        import torchvision.models as tvm
        weights_map = {
            "resnet18": tvm.ResNet18_Weights.DEFAULT,
            "resnet34": tvm.ResNet34_Weights.DEFAULT,
            "resnet50": tvm.ResNet50_Weights.DEFAULT,
        }
        factory = getattr(tvm, variant)
        weights = weights_map[variant] if pretrained else None
        return factory(weights=weights)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(
        self,
        image: Tensor | None = None,
        image_ids: list[int] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        if image is None:
            raise ValueError("ResNetFeatureExtractor requires `image`; got None")
        x = (image - self._imagenet_mean) / self._imagenet_std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self._output_layer == "layer4":
            x = self.layer4(x)
        return {"spatial": x}


class ResNetFeatureExtractorConfig(BaseFeatureExtractorConfig):
    variant: _VARIANT = "resnet18"
    output_layer: _LAYER = "layer3"
    pretrained: bool = True

    def build(self) -> ResNetFeatureExtractor:
        return ResNetFeatureExtractor(
            variant=self.variant,
            output_layer=self.output_layer,
            pretrained=self.pretrained,
        )
