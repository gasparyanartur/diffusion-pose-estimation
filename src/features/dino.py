"""Online DINOv3-ConvNeXt feature extractor.

Loads the HuggingFace model and runs it on raw images at inference time.
Returns the last ConvNeXt stage (stage 4, stride 32, 768 channels) as
``"spatial"``.

Use ``frozen=True`` (the default) to treat the backbone as a fixed feature
extractor and only train the probe head.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig

_MODEL_NAME = "facebook/dinov3-convnext-small-pretrain-lvd1689m"

# ConvNeXt-small stage → (out_channels, cumulative_stride)
_STAGE_OUT_CHANNELS = {1: 96, 2: 192, 3: 384, 4: 768}
_STAGE_STRIDE       = {1: 4,  2: 8,   3: 16,  4: 32}

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoFeatureExtractor(BaseFeatureExtractor):
    """Online DINOv3-ConvNeXt backbone; returns stage-4 spatial features.

    Args:
        model_name: HuggingFace model ID.
        frozen:     If True, all backbone parameters are frozen (no gradients).
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        stage: int = 4,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        if stage not in _STAGE_OUT_CHANNELS:
            raise ValueError(f"stage must be 1-4; got {stage}")
        from transformers import AutoModel

        self._stage = stage
        self._model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        if frozen:
            for p in self._model.parameters():
                p.requires_grad_(False)

        self.register_buffer("_mean", _IMAGENET_MEAN)
        self.register_buffer("_std",  _IMAGENET_STD)

    @property
    def out_channels(self) -> int:
        return _STAGE_OUT_CHANNELS[self._stage]

    def get_output_size(self, input_size: int) -> int:
        return input_size // _STAGE_STRIDE[self._stage]

    def forward(
        self,
        image: Tensor | None = None,
        image_ids: list[int] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        assert image is not None, "DinoFeatureExtractor requires image tensor"
        x = (image - self._mean.to(image)) / self._std.to(image)
        outputs = self._model(x)
        # hidden_states: 0=input, 1=stage1, 2=stage2, 3=stage3, 4=stage4
        spatial = outputs.hidden_states[self._stage]  # [B, C, H/stride, W/stride]
        return {"spatial": spatial}


class DinoFeatureExtractorConfig(BaseFeatureExtractorConfig):
    model_name: str = _MODEL_NAME
    stage: int = 4
    frozen: bool = True

    def build(self) -> DinoFeatureExtractor:
        return DinoFeatureExtractor(
            model_name=self.model_name,
            stage=self.stage,
            frozen=self.frozen,
        )
