"""Tests for GT-box loading and canonical conversion (I1 of box-diffusion pipeline)."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.data.datasets import PoseDataset, SetPoseDataset, set_pose_collate
from src.data.data_module import _boxes_to_padded_canonical_batch
from src.data.transforms import (
    bboxes_xywh_from_coco_anns,
    boxes_xywh_to_padded_canonical,
)

K = 14


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------

def test_bboxes_xywh_from_coco_anns_parses_bbox_list():
    anns = [
        {"bbox": [10.0, 20.0, 30.0, 40.0]},
        {"bbox": [0.0, 0.0, 5.0, 7.0]},
    ]
    out = bboxes_xywh_from_coco_anns(anns)
    assert out.shape == (2, 4)
    assert out.dtype == torch.float32
    assert torch.allclose(out[0], torch.tensor([10.0, 20.0, 30.0, 40.0]))


def test_bboxes_xywh_from_coco_anns_handles_missing_bbox():
    anns = [{"id": 1}, {"bbox": [1.0, 2.0, 3.0, 4.0]}]
    out = bboxes_xywh_from_coco_anns(anns)
    assert out.shape == (2, 4)
    assert torch.allclose(out[0], torch.zeros(4))


def test_boxes_xywh_to_padded_canonical_landlocked_centre():
    # H, W = 100, 200 -> max side = 200. Box covering the full image: x=0,y=0,w=200,h=100.
    # cx_pad = 100, cy_pad = 50; canonical: cx = 100/200*2 - 1 = 0; cy = 50/200*2 - 1 = -0.5.
    boxes = torch.tensor([[0.0, 0.0, 200.0, 100.0]])
    canon = boxes_xywh_to_padded_canonical(boxes, original_h=100, original_w=200)
    assert torch.allclose(canon, torch.tensor([[0.0, -0.5, 2.0, 1.0]]))


def test_boxes_xywh_to_padded_canonical_top_left_corner():
    # tiny box at origin: should map cx,cy to (-1 + w/side, -1 + h/side).
    boxes = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
    canon = boxes_xywh_to_padded_canonical(boxes, original_h=200, original_w=200)
    # cx_pad = 25, canonical = 25/200*2 - 1 = -0.75. w_canon = 50/200*2 = 0.5.
    assert torch.allclose(canon, torch.tensor([[-0.75, -0.75, 0.5, 0.5]]))


# ---------------------------------------------------------------------------
# SetPoseDataset emits gt_boxes_pixel and gt_box_labeled
# ---------------------------------------------------------------------------

def _make_synthetic_sample(img_path: Path, H: int, W: int) -> dict:
    Image.new("RGB", (W, H)).save(img_path)
    return {
        "image_id": 7,
        "img_path": str(img_path),
        "img_hw": (H, W),
        # 2 people, all-zero keypoints (content irrelevant for box tests)
        "keypoints":  torch.zeros(2, K, 2),
        "visibility": torch.full((2, K), 2.0),
        "gt_boxes_pixel": torch.tensor(
            [[10.0, 20.0, 30.0, 40.0],
             [0.0, 0.0, float(W), float(H)]],
            dtype=torch.float32,
        ),
    }


def test_set_pose_dataset_emits_box_fields(tmp_path):
    sample = _make_synthetic_sample(tmp_path / "a.png", H=100, W=200)
    base = PoseDataset(samples=[sample], num_keypoints=K)
    ds = SetPoseDataset(base)
    out = ds[0]
    assert "gt_boxes_pixel" in out and out["gt_boxes_pixel"].shape == (2, 4)
    assert "gt_box_labeled" in out and out["gt_box_labeled"].shape == (2,)
    assert torch.all(out["gt_box_labeled"] == 1.0)


def test_set_pose_collate_pads_boxes_to_pmax(tmp_path):
    s1 = _make_synthetic_sample(tmp_path / "a.png", H=100, W=200)
    s2 = dict(s1)
    s2["img_path"] = str(tmp_path / "b.png")
    Image.new("RGB", (200, 100)).save(s2["img_path"])
    # second sample has only 1 person → pad to P_max = 2
    s2["keypoints"] = torch.zeros(1, K, 2)
    s2["visibility"] = torch.full((1, K), 2.0)
    s2["gt_boxes_pixel"] = torch.tensor([[5.0, 5.0, 10.0, 10.0]])

    base = PoseDataset(samples=[s1, s2], num_keypoints=K)
    ds = SetPoseDataset(base)
    batch = set_pose_collate([ds[0], ds[1]])

    assert batch["gt_boxes_pixel"].shape == (2, 2, 4)
    assert batch["gt_box_labeled"].shape == (2, 2)
    # second sample padded slot is zero
    assert torch.allclose(batch["gt_boxes_pixel"][1, 1], torch.zeros(4))
    assert batch["gt_box_labeled"][1, 1].item() == 0.0
    assert batch["gt_box_labeled"][1, 0].item() == 1.0


# ---------------------------------------------------------------------------
# Padded-canvas canonical conversion at data-module stage
# ---------------------------------------------------------------------------

def test_boxes_to_padded_canonical_batch_shapes_and_bounds():
    # B=2 images of different sizes
    boxes_pixel = torch.tensor([
        [[10.0, 20.0, 30.0, 40.0], [0.0, 0.0, 100.0, 100.0]],
        [[50.0, 60.0, 80.0, 90.0], [0.0, 0.0,   0.0,   0.0]],  # padded slot
    ])
    img_hw = torch.tensor([[100.0, 200.0], [300.0, 400.0]])
    canon = _boxes_to_padded_canonical_batch(boxes_pixel, img_hw)
    assert canon.shape == (2, 2, 4)
    # First image, second box: full top-left 100x100 region in 200-side canvas.
    # cx_pad = 50; canon_cx = 50/200*2 - 1 = -0.5. w_canon = 100/200*2 = 1.0.
    assert torch.allclose(canon[0, 1], torch.tensor([-0.5, -0.5, 1.0, 1.0]))
    # Real boxes have centres in [-1, 1] and positive w, h
    valid = canon[[0, 0, 1], [0, 1, 0]]
    assert torch.all(valid[:, 0] >= -1.0) and torch.all(valid[:, 0] <= 1.0)
    assert torch.all(valid[:, 1] >= -1.0) and torch.all(valid[:, 1] <= 1.0)
    assert torch.all(valid[:, 2] > 0.0) and torch.all(valid[:, 3] > 0.0)
