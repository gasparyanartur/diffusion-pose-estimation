"""DiffusionDet adapter for the BoxDetector interface.

Detectron2 is NOT in this project's venv. The import is guarded so that
other detectors (e.g. BoxDiffusion) can be used without Detectron2 installed.
The error message points to the setup instructions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.box_detectors.base import BoxDetector, BoxDetectorConfig

# COCO thing_classes list produced by Detectron2's COCO metadata.
# Person is the first class (index 0).
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
    # DiffusionDet extends Detectron2's cfg with custom keys.
    diffusiondet_root = Path(__file__).parents[2] / "external" / "diffusiondet"
    sys.path.insert(0, str(diffusiondet_root))
    from diffusiondet import add_diffusiondet_config  # type: ignore[import]

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    cfg.merge_from_file(str(config_file))
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


def _pad_to_n(boxes: np.ndarray, scores: np.ndarray, n: int):
    """Zero-pad boxes [K,4] and scores [K] to [n,4] and [n]."""
    k = boxes.shape[0]
    if k >= n:
        return boxes[:n], scores[:n]
    pad_b = np.zeros((n - k, 4), dtype=np.float32)
    pad_s = np.zeros(n - k, dtype=np.float32)
    return np.concatenate([boxes, pad_b], axis=0), np.concatenate([scores, pad_s], axis=0)


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert [x1,y1,x2,y2] to [x,y,w,h]."""
    result = boxes.copy()
    result[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
    result[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
    return result


class DiffusionDetDetector(BoxDetector):
    """Runs DiffusionDet inference and filters to person detections.

    Images are loaded as BGR numpy arrays (OpenCV) because that is what
    Detectron2's DefaultPredictor expects.
    """

    def __init__(
        self,
        config_file: Path,
        checkpoint: Path,
        score_threshold: float = 0.3,
        max_detections: int = 100,
        device: str = "cuda",
    ) -> None:
        _require_detectron2()
        import cv2  # noqa: F401 — needed for image loading
        self._cv2 = cv2
        from detectron2.engine.defaults import DefaultPredictor
        cfg = _build_cfg(config_file, checkpoint, score_threshold, device)
        self._predictor = DefaultPredictor(cfg)
        self._max_detections = max_detections
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def predict(
        self,
        image_paths: list[Path],
        image_ids: list[int],  # noqa: ARG002 — unused here; part of interface
    ) -> dict[str, Tensor]:
        all_boxes = []
        all_scores = []

        for path in image_paths:
            bgr = self._cv2.imread(str(path))
            instances = self._predictor(bgr)["instances"].to("cpu")

            # Filter to person class.
            person_mask = instances.pred_classes == _COCO_PERSON_CLASS_ID
            person_boxes = instances.pred_boxes.tensor[person_mask].numpy()   # [K, 4] xyxy
            person_scores = instances.scores[person_mask].numpy()             # [K]

            person_boxes = _xyxy_to_xywh(person_boxes)
            boxes_padded, scores_padded = _pad_to_n(
                person_boxes, person_scores, self._max_detections
            )
            all_boxes.append(boxes_padded)
            all_scores.append(scores_padded)

        # B x max_detections x 4
        boxes_tensor = torch.from_numpy(np.stack(all_boxes)).to(self._device)
        scores_tensor = torch.from_numpy(np.stack(all_scores)).to(self._device)
        return {"boxes_xywh": boxes_tensor, "scores": scores_tensor}


class DiffusionDetConfig(BoxDetectorConfig):
    """Hydra-compatible config for DiffusionDetDetector.

    config_file and checkpoint paths should be absolute or relative to the
    workspace root.
    """

    config_file: str
    checkpoint: str
    score_threshold: float = 0.3
    max_detections: int = 100
    device: str = "cuda"

    def build(self) -> DiffusionDetDetector:
        return DiffusionDetDetector(
            config_file=Path(self.config_file),
            checkpoint=Path(self.checkpoint),
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            device=self.device,
        )
