"""Thin DiffusionDet adapter for path-based external inference."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.models.pipelines.object_detectors.base import BoxDetector, BoxDetectorConfig

_COCO_PERSON_CLASS_ID = 0
_DETECTRON2_INSTALL_MSG = (
    "Detectron2 is required to run DiffusionDet but is not installed.\n"
    "See external/diffusiondet/GETTING_STARTED.md for instructions.\n"
    "Quick install (example):\n"
    "  pip install 'git+https://github.com/facebookresearch/detectron2.git'"
)


def _require_detectron2() -> Any:
    try:
        import detectron2  # noqa: F401
    except ImportError:
        raise ImportError(_DETECTRON2_INSTALL_MSG) from None


def _build_cfg(config_file: Path, checkpoint: Path, score_threshold: float, device: str):
    _require_detectron2()
    from detectron2.config import get_cfg

    diffusiondet_root = Path(__file__).parents[4] / "external" / "diffusiondet"
    sys.path.insert(0, str(diffusiondet_root))
    from diffusiondet import add_diffusiondet_config

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    cfg.merge_from_file(str(config_file))
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


def _pad_to_n(boxes: np.ndarray, scores: np.ndarray, n: int):
    k = boxes.shape[0]
    if k >= n:
        return boxes[:n], scores[:n]
    return (
        np.concatenate([boxes, np.zeros((n - k, 4), dtype=np.float32)], axis=0),
        np.concatenate([scores, np.zeros(n - k, dtype=np.float32)], axis=0),
    )


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    result = boxes.copy()
    result[:, 2] = boxes[:, 2] - boxes[:, 0]
    result[:, 3] = boxes[:, 3] - boxes[:, 1]
    return result


class DiffusionDetDetector(BoxDetector):
    def __init__(self, config_file: Path, checkpoint: Path, score_threshold: float = 0.3, max_detections: int = 100, device: str = "cuda") -> None:
        _require_detectron2()
        import cv2
        from detectron2.engine.defaults import DefaultPredictor

        self._cv2 = cv2
        self._predictor = DefaultPredictor(_build_cfg(config_file, checkpoint, score_threshold, device))
        self._max_detections = max_detections
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def predict(self, image_paths: list[Path], image_ids: list[int]) -> dict[str, Tensor]:
        del image_ids
        all_boxes, all_scores = [], []
        for path in image_paths:
            instances = self._predictor(self._cv2.imread(str(path)))["instances"].to("cpu")
            person_mask = instances.pred_classes == _COCO_PERSON_CLASS_ID
            boxes = _xyxy_to_xywh(instances.pred_boxes.tensor[person_mask].numpy())
            scores = instances.scores[person_mask].numpy()
            boxes, scores = _pad_to_n(boxes, scores, self._max_detections)
            all_boxes.append(boxes)
            all_scores.append(scores)
        return {
            "boxes_xywh": torch.from_numpy(np.stack(all_boxes)).to(self._device),
            "scores": torch.from_numpy(np.stack(all_scores)).to(self._device),
        }


class DiffusionDetConfig(BoxDetectorConfig):
    config_file: str
    checkpoint: str
    score_threshold: float = 0.3
    max_detections: int = 100
    device: str = "cuda"

    def build(self) -> DiffusionDetDetector:
        return DiffusionDetDetector(Path(self.config_file), Path(self.checkpoint), self.score_threshold, self.max_detections, self.device)
