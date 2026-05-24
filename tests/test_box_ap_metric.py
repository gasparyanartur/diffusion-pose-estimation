"""Tests for BoxAPMetric — pycocotools-backed bbox AP (I5)."""

from __future__ import annotations

import torch
from pycocotools.coco import COCO

import pytest

from src.metrics import BoxAPMetric


def _build_synthetic_coco() -> COCO:
    """Build a tiny in-memory COCO dataset with 2 images, 3 GT boxes total.

    Image 1: 2 boxes. Image 2: 1 box. All boxes are "medium" so that the
    ``APm`` slot in COCOeval's stats is populated and ``APs`` / ``APl`` are -1.
    """
    coco_dict = {
        "images": [
            {"id": 1, "width": 200, "height": 200, "file_name": "a.jpg"},
            {"id": 2, "width": 200, "height": 200, "file_name": "b.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [10.0, 10.0, 50.0, 50.0], "area": 50 * 50, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1,
             "bbox": [80.0, 80.0, 60.0, 60.0], "area": 60 * 60, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1,
             "bbox": [40.0, 40.0, 70.0, 70.0], "area": 70 * 70, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    coco = COCO()
    coco.dataset = coco_dict
    coco.createIndex()
    return coco


def test_init_requires_exactly_one_source():
    try:
        BoxAPMetric()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when neither source is given")

    coco = _build_synthetic_coco()
    try:
        BoxAPMetric(ann_file="foo.json", coco=coco)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when both sources are given")


def test_perfect_predictions_yield_ap_one():
    coco = _build_synthetic_coco()
    metric = BoxAPMetric(coco=coco)

    # Pad slot dimension to N=3 with zero-area boxes; metric filters those out.
    pred_boxes_img1 = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],
        [80.0, 80.0, 60.0, 60.0],
        [0.0, 0.0, 0.0, 0.0],            # filtered (w/h = 0)
    ])
    pred_boxes_img2 = torch.tensor([
        [40.0, 40.0, 70.0, 70.0],
        [0.0, 0.0, 0.0, 0.0],            # filtered
        [0.0, 0.0, 0.0, 0.0],            # filtered
    ])
    pred_boxes = torch.stack([pred_boxes_img1, pred_boxes_img2], dim=0)  # [2, 3, 4]
    scores = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    image_ids = torch.tensor([1, 2])

    metric.update(pred_boxes, scores, image_ids)
    out = metric.compute()

    assert out["bbox/AP"] > 0.99
    assert out["bbox/AP50"] > 0.99
    assert out["bbox/AP75"] > 0.99


def test_disjoint_predictions_yield_zero_ap():
    coco = _build_synthetic_coco()
    metric = BoxAPMetric(coco=coco)

    # Predict far-away tiny boxes that do not overlap any GT.
    pred_boxes = torch.tensor([
        [[180.0, 180.0, 5.0, 5.0]],
        [[180.0, 180.0, 5.0, 5.0]],
    ])  # [2, 1, 4]
    scores = torch.tensor([[1.0], [1.0]])
    image_ids = torch.tensor([1, 2])

    metric.update(pred_boxes, scores, image_ids)
    out = metric.compute()

    assert out["bbox/AP"] == 0.0
    assert out["bbox/AP50"] == 0.0


def test_compute_empty_returns_empty_dict():
    coco = _build_synthetic_coco()
    metric = BoxAPMetric(coco=coco)
    out = metric.compute()
    assert out == {}


def test_reset_clears_state():
    coco = _build_synthetic_coco()
    metric = BoxAPMetric(coco=coco)
    pred_boxes = torch.tensor([[[10.0, 10.0, 50.0, 50.0]]])
    scores = torch.tensor([[1.0]])
    metric.update(pred_boxes, scores, torch.tensor([1]))
    assert len(metric._results) == 1
    metric.reset()
    assert len(metric._results) == 0
    assert metric.compute() == {}


def test_update_filters_unknown_image_ids():
    coco = _build_synthetic_coco()
    metric = BoxAPMetric(coco=coco)
    pred_boxes = torch.tensor([[[10.0, 10.0, 50.0, 50.0]]])
    scores = torch.tensor([[1.0]])
    # 999 is not in the synthetic COCO.
    metric.update(pred_boxes, scores, torch.tensor([999]))
    assert len(metric._results) == 0


def test_score_threshold_filters_low_confidence():
    coco = _build_synthetic_coco()
    metric = BoxAPMetric(coco=coco, score_threshold=0.5)
    pred_boxes = torch.tensor([
        [[10.0, 10.0, 50.0, 50.0], [80.0, 80.0, 60.0, 60.0]],
    ])
    scores = torch.tensor([[0.9, 0.1]])  # second one filtered out
    metric.update(pred_boxes, scores, torch.tensor([1]))
    assert len(metric._results) == 1
    assert metric._results[0]["score"] == pytest.approx(0.9)
