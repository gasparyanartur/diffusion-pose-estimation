"""Tests for the BoxDetector interface and adapters."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.box_detectors.base import BoxDetector, BoxDetectorConfig
from src.box_detectors.box_diffusion import BoxDiffusionDetectorConfig
from src.box_detectors.diffusiondet import (
    DiffusionDetConfig,
    _DETECTRON2_INSTALL_MSG,
    _xyxy_to_xywh,
    _pad_to_n,
)


# ---------------------------------------------------------------------------
# Helpers / utilities
# ---------------------------------------------------------------------------


class _DummyDetector(BoxDetector):
    """Minimal concrete implementation for interface tests."""

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def predict(self, image_paths, image_ids):
        B = len(image_paths)
        return {
            "boxes_xywh": torch.zeros(B, 10, 4),
            "scores": torch.zeros(B, 10),
        }


class _DummyDetectorConfig(BoxDetectorConfig):
    def build(self) -> BoxDetector:
        return _DummyDetector()


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------


def test_dummy_detector_satisfies_interface():
    cfg = _DummyDetectorConfig()
    det = cfg.build()
    result = det.predict(image_paths=[Path("a.jpg")], image_ids=[1])
    assert result["boxes_xywh"].shape == (1, 10, 4)
    assert result["scores"].shape == (1, 10)


def test_dummy_detector_device():
    det = _DummyDetector()
    assert det.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# DiffusionDet adapter utilities
# ---------------------------------------------------------------------------


def test_xyxy_to_xywh_values():
    boxes = np.array([[10.0, 20.0, 50.0, 80.0]])
    result = _xyxy_to_xywh(boxes)
    np.testing.assert_allclose(result, [[10.0, 20.0, 40.0, 60.0]])


def test_xyxy_to_xywh_does_not_modify_input():
    boxes = np.array([[0.0, 0.0, 100.0, 200.0]])
    original = boxes.copy()
    _xyxy_to_xywh(boxes)
    np.testing.assert_array_equal(boxes, original)


def test_pad_to_n_extends():
    boxes = np.ones((3, 4), dtype=np.float32)
    scores = np.ones(3, dtype=np.float32)
    padded_b, padded_s = _pad_to_n(boxes, scores, n=5)
    assert padded_b.shape == (5, 4)
    assert padded_s.shape == (5,)
    np.testing.assert_array_equal(padded_b[3:], 0.0)
    np.testing.assert_array_equal(padded_s[3:], 0.0)


def test_pad_to_n_clips_excess():
    boxes = np.ones((10, 4), dtype=np.float32)
    scores = np.ones(10, dtype=np.float32)
    padded_b, padded_s = _pad_to_n(boxes, scores, n=5)
    assert padded_b.shape == (5, 4)
    assert padded_s.shape == (5,)


# ---------------------------------------------------------------------------
# DiffusionDet import guard
# ---------------------------------------------------------------------------


def test_diffusiondet_build_raises_without_detectron2():
    cfg = DiffusionDetConfig(
        config_file="external/diffusiondet/configs/diffdet.coco.res50.yaml",
        checkpoint="checkpoints/diffdet_coco_res50.pth",
    )
    with patch.dict("sys.modules", {"detectron2": None}):
        with pytest.raises(ImportError, match="Detectron2"):
            cfg.build()


# ---------------------------------------------------------------------------
# BoxDiffusionDetectorConfig construction
# ---------------------------------------------------------------------------


def test_box_diffusion_config_construction():
    cfg = BoxDiffusionDetectorConfig(
        checkpoint="results/my_run/checkpoints/last.ckpt",
        num_inference_steps=3,
        score_threshold=0.5,
        device="cpu",
    )
    assert cfg.num_inference_steps == 3
    assert cfg.score_threshold == 0.5
