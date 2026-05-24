"""BoxDiffusion adapter for the BoxDetector interface.

Wraps a trained BoxSetDiffusionModule loaded from a Lightning checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

from src.box_detectors.base import BoxDetector, BoxDetectorConfig
from src.data.transforms import boxes_padded_canonical_to_pixel_xywh


def _load_pil_rgb(path: Path) -> Tensor:
    """Load an image as a float RGB tensor in [0, 1], shape [3, H, W]."""
    from PIL import Image
    import torchvision.transforms.functional as TF
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img)


class BoxDiffusionDetector(BoxDetector):
    """Run our trained box-set diffusion model as a BoxDetector.

    The Lightning checkpoint is loaded once at construction time.
    The backbone and decoder are taken from the checkpoint.
    """

    def __init__(
        self,
        checkpoint: Path,
        num_inference_steps: int = 5,
        score_threshold: float = 0.3,
        device: str = "cuda",
    ) -> None:
        from src.training.box_diffusion_module import BoxSetDiffusionModule

        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._module = BoxSetDiffusionModule.load_from_checkpoint(
            str(checkpoint), map_location=self._device
        )
        self._module.eval().to(self._device)
        self._num_inference_steps = num_inference_steps
        self._score_threshold = score_threshold

    @property
    def device(self) -> torch.device:
        return self._device

    def predict(
        self,
        image_paths: list[Path],
        image_ids: list[int],
    ) -> dict[str, Tensor]:
        images = torch.stack([_load_pil_rgb(p) for p in image_paths]).to(self._device)
        # B, C, H, W
        B, _, H, W = images.shape

        with torch.no_grad():
            features = self._module.backbone(image=images)
            from src.training.box_diffusion_module import _image_tokens_and_centers
            image_tokens, patch_centers = _image_tokens_and_centers(
                features, self._module.hparams.get("feature_key", "spatial")
            )
            sample_out = self._module.decoder.sample(
                image_tokens=image_tokens,
                patch_centers=patch_centers,
                num_inference_steps=self._num_inference_steps,
            )

        pred_boxes_canon = sample_out["pred_boxes"]                      # [B, N, 4] canonical
        obj_logits = sample_out["obj_logits"]                            # [B, N]
        scores = torch.sigmoid(obj_logits)                               # [B, N]

        img_hw = images.new_tensor([[H, W]]).expand(B, 2)               # [B, 2]
        boxes_xywh = boxes_padded_canonical_to_pixel_xywh(pred_boxes_canon, img_hw)

        # Mask out low-confidence slots.
        low_conf = scores < self._score_threshold
        boxes_xywh[low_conf] = 0.0
        scores[low_conf] = 0.0

        return {"boxes_xywh": boxes_xywh, "scores": scores}


class BoxDiffusionDetectorConfig(BoxDetectorConfig):
    """Config for BoxDiffusionDetector."""

    checkpoint: str
    num_inference_steps: int = 5
    score_threshold: float = 0.3
    device: str = "cuda"

    def build(self) -> BoxDiffusionDetector:
        return BoxDiffusionDetector(
            checkpoint=Path(self.checkpoint),
            num_inference_steps=self.num_inference_steps,
            score_threshold=self.score_threshold,
            device=self.device,
        )
