"""Tests for set-prediction losses, set-format collate, and SetPoseDataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics import CROWDPOSE_SIGMAS
from src.training.losses import (
    SetPredictionLossWeights,
    compute_oks,
    hungarian_match,
    set_prediction_loss,
    set_prediction_loss_per_image,
)


SIGMAS = torch.from_numpy(CROWDPOSE_SIGMAS).float()
K = 14


# ---------------------------------------------------------------------------
# compute_oks
# ---------------------------------------------------------------------------


def test_compute_oks_shape_and_range():
    pred = torch.rand(5, K, 2) * 2 - 1
    gt = torch.rand(3, K, 2) * 2 - 1
    labeled = torch.ones(3, K)
    scale = torch.full((3,), 0.4)
    oks = compute_oks(pred, gt, labeled, scale, SIGMAS)
    assert oks.shape == (5, 3)
    assert torch.isfinite(oks).all()
    assert (oks >= 0).all() and (oks <= 1).all()


def test_compute_oks_perfect_match_is_one():
    gt = torch.rand(2, K, 2) * 2 - 1
    pred = gt.clone()  # identical poses for both slots
    labeled = torch.ones(2, K)
    scale = torch.full((2,), 0.3)
    oks = compute_oks(pred, gt, labeled, scale, SIGMAS)
    # diagonal: identical pose => OKS = 1
    assert torch.allclose(oks.diag(), torch.ones(2), atol=1e-5)


def test_compute_oks_handles_empty_gt():
    pred = torch.rand(4, K, 2) * 2 - 1
    oks = compute_oks(
        pred,
        torch.zeros(0, K, 2),
        torch.zeros(0, K),
        torch.zeros(0),
        SIGMAS,
    )
    assert oks.shape == (4, 0)


def test_compute_oks_handles_empty_pred():
    gt = torch.rand(3, K, 2)
    oks = compute_oks(
        torch.zeros(0, K, 2),
        gt, torch.ones(3, K), torch.full((3,), 0.3), SIGMAS,
    )
    assert oks.shape == (0, 3)


# ---------------------------------------------------------------------------
# hungarian_match
# ---------------------------------------------------------------------------


def test_hungarian_empty_returns_empty():
    oks = torch.rand(5, 0)
    slot_idx, gt_idx = hungarian_match(oks, torch.zeros(5))
    assert slot_idx.numel() == 0 and gt_idx.numel() == 0


def test_hungarian_match_size_min_n_p():
    oks = torch.rand(5, 3)
    slot_idx, gt_idx = hungarian_match(oks, torch.zeros(5))
    assert slot_idx.numel() == 3
    assert gt_idx.numel() == 3
    assert (slot_idx < 5).all() and (gt_idx < 3).all()
    # bijection within the matched set
    assert len(set(gt_idx.tolist())) == gt_idx.numel()


def test_hungarian_match_picks_diagonal_when_oks_is_identity():
    # 3 slots, 3 gt: identity OKS => optimal assignment is the diagonal.
    oks = torch.eye(3)
    slot_idx, gt_idx = hungarian_match(oks, torch.zeros(3), lambda_obj=0.0)
    pairs = sorted(zip(slot_idx.tolist(), gt_idx.tolist()))
    assert pairs == [(0, 0), (1, 1), (2, 2)]


# ---------------------------------------------------------------------------
# set_prediction_loss_per_image
# ---------------------------------------------------------------------------


def test_set_prediction_loss_per_image_no_gt():
    weights = SetPredictionLossWeights()
    out = set_prediction_loss_per_image(
        pred_objectness=torch.zeros(5),
        pred_keypoints=torch.zeros(5, K, 2),
        pred_vis_logits=torch.zeros(5, K),
        gt_keypoints=torch.zeros(0, K, 2),
        gt_labeled=torch.zeros(0, K),
        gt_visible=torch.zeros(0, K),
        person_scale=torch.zeros(0),
        sigmas=SIGMAS,
        weights=weights,
    )
    # all components finite; matched_count == 0; obj loss is non-zero (target all-zero)
    assert torch.isfinite(out["obj"])
    assert out["matched_count"].item() == 0
    assert out["mean_oks"].item() == 0.0


def test_set_prediction_loss_per_image_smoke_and_grad():
    weights = SetPredictionLossWeights()
    pred_obj = torch.zeros(5, requires_grad=True)
    pred_kp = (torch.rand(5, K, 2) * 2 - 1).requires_grad_(True)
    pred_vis = torch.zeros(5, K, requires_grad=True)
    gt_kp = torch.rand(2, K, 2) * 2 - 1
    out = set_prediction_loss_per_image(
        pred_objectness=pred_obj,
        pred_keypoints=pred_kp,
        pred_vis_logits=pred_vis,
        gt_keypoints=gt_kp,
        gt_labeled=torch.ones(2, K),
        gt_visible=torch.ones(2, K),
        person_scale=torch.full((2,), 0.4),
        sigmas=SIGMAS,
        weights=weights,
    )
    total = out["l1"] + out["oks"] + out["obj"] + out["vis"]
    assert torch.isfinite(total)
    total.backward()
    # L1 must give non-zero gradient on keypoints
    assert pred_kp.grad is not None and pred_kp.grad.abs().sum() > 0
    assert out["matched_count"].item() == 2


def test_set_prediction_loss_batched():
    weights = SetPredictionLossWeights()
    B, N = 2, 4
    kp_leaf = (torch.rand(B, N, K, 2) * 2 - 1).requires_grad_(True)
    pred = {
        "scores_logits": torch.zeros(B, N, requires_grad=True),
        "keypoints": kp_leaf,
        "vis_logits": torch.zeros(B, N, K, requires_grad=True),
    }
    P_max = 3
    targets = {
        "gt_keypoints": torch.rand(B, P_max, K, 2) * 2 - 1,
        "gt_labeled": torch.ones(B, P_max, K),
        "gt_visible": torch.ones(B, P_max, K),
        "person_scale": torch.full((B, P_max), 0.4),
        "person_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32),
    }
    out = set_prediction_loss(pred, targets, SIGMAS, weights)
    assert torch.isfinite(out["total"])
    out["total"].backward()
    assert kp_leaf.grad is not None and kp_leaf.grad.abs().sum() > 0
    # 2 + 1 = 3 matches expected
    assert out["matched"].item() == 3


# ---------------------------------------------------------------------------
# set-format collate
# ---------------------------------------------------------------------------


def test_set_pose_collate_pads_to_p_max():
    from src.data.set_pose_dataset import set_pose_collate

    sample_a = {
        "image": torch.zeros(3, 8, 8),
        "gt_keypoints": torch.rand(2, K, 2),
        "gt_labeled": torch.ones(2, K),
        "gt_visible": torch.ones(2, K),
        "person_scale": torch.full((2,), 0.3),
        "image_id": 1,
        "img_path": "a",
        "img_hw": torch.tensor([100.0, 200.0]),
    }
    sample_b = {
        "image": torch.zeros(3, 8, 8),
        "gt_keypoints": torch.rand(1, K, 2),
        "gt_labeled": torch.ones(1, K),
        "gt_visible": torch.ones(1, K),
        "person_scale": torch.full((1,), 0.3),
        "image_id": 2,
        "img_path": "b",
        "img_hw": torch.tensor([120.0, 220.0]),
    }
    batch = set_pose_collate([sample_a, sample_b])
    assert batch["image"].shape == (2, 3, 8, 8)
    assert batch["gt_keypoints"].shape == (2, 2, K, 2)
    assert batch["person_mask"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
    # padded slot in second image is zero
    assert batch["gt_keypoints"][1, 1].abs().sum().item() == 0.0
    assert batch["image_id"].tolist() == [1, 2]


def test_set_pose_collate_handles_zero_person_image():
    from src.data.set_pose_dataset import set_pose_collate

    s = {
        "image": torch.zeros(3, 8, 8),
        "gt_keypoints": torch.zeros(0, K, 2),
        "gt_labeled": torch.zeros(0, K),
        "gt_visible": torch.zeros(0, K),
        "person_scale": torch.zeros(0),
        "image_id": 1,
        "img_path": "a",
        "img_hw": torch.tensor([100.0, 200.0]),
    }
    batch = set_pose_collate([s])
    # P_max coerced to 1
    assert batch["gt_keypoints"].shape == (1, 1, K, 2)
    assert batch["person_mask"].sum().item() == 0.0


# ---------------------------------------------------------------------------
# SetPoseDataset (canonical (-1, 1) mapping)
# ---------------------------------------------------------------------------


def test_set_pose_dataset_canonical_mapping(tmp_path, monkeypatch):
    """Pixel keypoints centered in the image map to (0, 0) canonical."""
    from PIL import Image

    from src.data.datasets import PoseDataset
    from src.data.set_pose_dataset import SetPoseDataset

    img_path = tmp_path / "img.png"
    Image.new("RGB", (200, 100)).save(img_path)
    samples = [{
        "image_id": 1,
        "img_path": str(img_path),
        # one person with two labeled joints at image corners and centre
        "keypoints": torch.tensor([[
            [100.0, 50.0],   # centre -> (0, 0)
            [0.0, 0.0],      # top-left -> (-1, -1)
            [200.0, 100.0],  # bottom-right -> (+1, +1)
        ] + [[0.0, 0.0]] * (K - 3)], dtype=torch.float32),
        "visibility": torch.tensor(
            [[2, 2, 2] + [0] * (K - 3)], dtype=torch.float32,
        ),
    }]
    base = PoseDataset(samples=samples, num_keypoints=K)
    ds = SetPoseDataset(base, image_h=64, image_w=128)
    out = ds[0]
    kp = out["gt_keypoints"][0]  # [K, 2]
    assert torch.allclose(kp[0], torch.tensor([0.0, 0.0]), atol=1e-5)
    assert torch.allclose(kp[1], torch.tensor([-1.0, -1.0]), atol=1e-5)
    assert torch.allclose(kp[2], torch.tensor([1.0, 1.0]), atol=1e-5)
    assert out["image"].shape == (3, 64, 128)
    assert out["gt_labeled"][0, :3].tolist() == [1.0, 1.0, 1.0]
    assert out["gt_visible"][0, :3].tolist() == [1.0, 1.0, 1.0]
