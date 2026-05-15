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
    SetPredictionLoss,
    SetPredictionLossWeights,
    compute_oks,
    hungarian_match,
)


SIGMAS = torch.from_numpy(CROWDPOSE_SIGMAS).float()
K = 14


def _make_loss(weights: SetPredictionLossWeights | None = None) -> SetPredictionLoss:
    return SetPredictionLoss(weights or SetPredictionLossWeights(), SIGMAS)


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
# SetPredictionLoss — single image
# ---------------------------------------------------------------------------


def _single_image_pred_and_targets(
    N: int,
    P: int,
    pred_kp: torch.Tensor | None = None,
    pred_obj: torch.Tensor | None = None,
) -> tuple[dict, dict]:
    """Build minimal pred/targets dicts for a single-image batch."""
    if pred_kp is None:
        pred_kp = torch.zeros(1, N, K, 2)
    if pred_obj is None:
        pred_obj = torch.zeros(1, N)
    pred = {
        "scores_logits": pred_obj,
        "keypoints": pred_kp,
        "vis_logits": torch.zeros(1, N, K),
    }
    P_max = max(P, 1)
    targets = {
        "gt_keypoints": torch.zeros(1, P_max, K, 2),
        "gt_labeled":   torch.ones(1, P_max, K),
        "gt_visible":   torch.ones(1, P_max, K),
        "person_scale": torch.full((1, P_max), 0.4),
        "person_mask":  torch.tensor([[1.0] * P + [0.0] * (P_max - P)]),
    }
    return pred, targets


def test_set_prediction_loss_no_gt():
    loss_fn = _make_loss()
    pred, targets = _single_image_pred_and_targets(N=5, P=0)
    out = loss_fn(pred, targets)
    assert torch.isfinite(out["objectness_loss"])
    assert out["matched"].item() == 0
    assert out["mean_oks"].item() == 0.0
    assert out["ranking_loss"].item() == 0.0


def test_set_prediction_loss_smoke_and_grad():
    loss_fn = _make_loss()
    pred_obj = torch.zeros(1, 5, requires_grad=True)
    pred_kp  = (torch.rand(1, 5, K, 2) * 2 - 1).requires_grad_(True)
    pred = {
        "scores_logits": pred_obj,
        "keypoints":     pred_kp,
        "vis_logits":    torch.zeros(1, 5, K, requires_grad=True),
    }
    targets = {
        "gt_keypoints": torch.rand(1, 2, K, 2) * 2 - 1,
        "gt_labeled":   torch.ones(1, 2, K),
        "gt_visible":   torch.ones(1, 2, K),
        "person_scale": torch.full((1, 2), 0.4),
        "person_mask":  torch.ones(1, 2),
    }
    out = loss_fn(pred, targets)
    assert torch.isfinite(out["total"])
    out["total"].backward()
    assert pred_kp.grad is not None and pred_kp.grad.abs().sum() > 0
    assert out["matched"].item() == 2


def test_set_prediction_loss_batched():
    loss_fn = _make_loss()
    B, N = 2, 4
    kp_leaf = (torch.rand(B, N, K, 2) * 2 - 1).requires_grad_(True)
    pred = {
        "scores_logits": torch.zeros(B, N, requires_grad=True),
        "keypoints":     kp_leaf,
        "vis_logits":    torch.zeros(B, N, K, requires_grad=True),
    }
    P_max = 3
    targets = {
        "gt_keypoints": torch.rand(B, P_max, K, 2) * 2 - 1,
        "gt_labeled":   torch.ones(B, P_max, K),
        "gt_visible":   torch.ones(B, P_max, K),
        "person_scale": torch.full((B, P_max), 0.4),
        "person_mask":  torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32),
    }
    out = loss_fn(pred, targets)
    assert torch.isfinite(out["total"])
    out["total"].backward()
    assert kp_leaf.grad is not None and kp_leaf.grad.abs().sum() > 0
    assert out["matched"].item() == 3  # 2 + 1


def test_pos_weight_balances_objectness_gradient():
    """With N=20 and P=2, pos_weight=(20-2)/2=9 so matched slots get higher gradient."""
    loss_fn = _make_loss()
    N, P = 20, 2
    pred_obj = torch.zeros(1, N, requires_grad=True)
    pred = {
        "scores_logits": pred_obj,
        "keypoints":     torch.zeros(1, N, K, 2),
        "vis_logits":    torch.zeros(1, N, K),
    }
    targets = {
        "gt_keypoints": torch.zeros(1, P, K, 2),  # zero error -> high OKS -> matched
        "gt_labeled":   torch.ones(1, P, K),
        "gt_visible":   torch.ones(1, P, K),
        "person_scale": torch.full((1, P), 0.4),
        "person_mask":  torch.ones(1, P),
    }
    out = loss_fn(pred, targets)
    out["objectness_loss"].backward()
    grad = pred_obj.grad[0].abs()
    matched_grad   = grad[:P].mean().item()
    unmatched_grad = grad[P:].mean().item()
    assert matched_grad > unmatched_grad * 3, (
        f"Expected matched_grad ({matched_grad:.4f}) >> unmatched_grad ({unmatched_grad:.4f})"
    )


def test_ranking_loss_zero_when_no_gt():
    loss_fn = _make_loss()
    pred, targets = _single_image_pred_and_targets(N=5, P=0)
    out = loss_fn(pred, targets)
    assert out["ranking_loss"].item() == 0.0


def test_ranking_loss_decreases_when_matched_outscores_unmatched():
    """Ranking loss should be lower when matched slot has higher logit than unmatched."""
    loss_fn = _make_loss()
    N, P = 5, 1

    def _loss(matched_logit: float, unmatched_logit: float) -> float:
        logits = torch.full((1, N), unmatched_logit)
        logits[0, 0] = matched_logit  # slot 0 matched (zero pred error)
        pred = {
            "scores_logits": logits,
            "keypoints":     torch.zeros(1, N, K, 2),
            "vis_logits":    torch.zeros(1, N, K),
        }
        targets = {
            "gt_keypoints": torch.zeros(1, P, K, 2),
            "gt_labeled":   torch.ones(1, P, K),
            "gt_visible":   torch.ones(1, P, K),
            "person_scale": torch.full((1, P), 0.4),
            "person_mask":  torch.ones(1, P),
        }
        return loss_fn(pred, targets)["ranking_loss"].item()

    loss_correct = _loss(matched_logit=5.0,  unmatched_logit=-5.0)
    loss_wrong   = _loss(matched_logit=-5.0, unmatched_logit=5.0)
    assert loss_correct < loss_wrong, (
        f"Expected lower ranking_loss when matched > unmatched, "
        f"got {loss_correct:.4f} vs {loss_wrong:.4f}"
    )


# ---------------------------------------------------------------------------
# resize_and_pad_to_square
# ---------------------------------------------------------------------------


def test_resize_and_pad_to_square_landscape():
    from src.data.data_module import resize_and_pad_to_square
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode

    # 3 × 40 × 80  (landscape) → longest side 80 → scale 0.5 → 20 × 40 padded to 40 × 40
    img = torch.rand(3, 40, 80)
    out, mask = resize_and_pad_to_square(img, size=40)

    assert out.shape == (3, 40, 40)
    assert mask.shape == (40, 40)
    # real region occupies top half; bottom is padded
    assert mask[:20, :40].all()
    assert not mask[20:, :].any()
    # padded region is zero
    assert out[:, 20:, :].eq(0).all()
    # real region matches a direct resize of the original
    expected = TF.resize(img, [20, 40], interpolation=InterpolationMode.BICUBIC, antialias=True)
    assert torch.equal(out[:, :20, :40], expected)


def test_resize_and_pad_to_square_portrait():
    from src.data.data_module import resize_and_pad_to_square
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode

    # 3 × 80 × 40  (portrait) → longest side 80 → scale 0.5 → 40 × 20 padded to 40 × 40
    img = torch.rand(3, 80, 40)
    out, mask = resize_and_pad_to_square(img, size=40)

    assert out.shape == (3, 40, 40)
    assert mask.shape == (40, 40)
    assert mask[:40, :20].all()
    assert not mask[:, 20:].any()
    assert out[:, :, 20:].eq(0).all()
    # real region matches a direct resize of the original
    expected = TF.resize(img, [40, 20], interpolation=InterpolationMode.BICUBIC, antialias=True)
    assert torch.equal(out[:, :40, :20], expected)


def test_resize_and_pad_to_square_already_square():
    from src.data.data_module import resize_and_pad_to_square
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode

    img = torch.rand(3, 64, 64)
    out, mask = resize_and_pad_to_square(img, size=64)

    assert out.shape == (3, 64, 64)
    assert mask.all()   # no padding at all
    # content matches a direct resize (no-op at same size)
    expected = TF.resize(img, [64, 64], interpolation=InterpolationMode.BICUBIC, antialias=True)
    assert torch.equal(out, expected)


def test_resize_and_pad_to_square_batch_all_square():
    from src.data.data_module import resize_and_pad_to_square

    # Mix of aspect ratios; all outputs must be size × size squares
    imgs = [
        torch.rand(3, 40, 80),   # landscape
        torch.rand(3, 80, 40),   # portrait
        torch.rand(3, 64, 64),   # square
        torch.rand(3, 30, 50),   # landscape, different size
    ]
    size = 40
    outputs = [resize_and_pad_to_square(img, size=size) for img in imgs]

    for i, (out, mask) in enumerate(outputs):
        assert out.shape == (3, size, size), f"image {i}: expected ({3}, {size}, {size}), got {tuple(out.shape)}"
        assert mask.shape == (size, size),   f"image {i}: mask shape {tuple(mask.shape)}"


# ---------------------------------------------------------------------------
# compute_person_scale
# ---------------------------------------------------------------------------


def test_compute_person_scale_normal():
    from src.data.datasets import compute_person_scale

    # One person, 2 labeled joints at canonical (-1,-1) and (1,1) → extent = 2.0
    kp = torch.tensor([[[- 1.0, -1.0], [1.0, 1.0]]])   # [1, 2, 2]
    labeled = torch.ones(1, 2)
    scale = compute_person_scale(kp, labeled)
    assert scale.shape == (1,)
    assert abs(scale[0].item() - 2.0) < 1e-5


def test_compute_person_scale_falls_back_for_single_joint():
    from src.data.datasets import compute_person_scale

    kp = torch.tensor([[[0.5, 0.5], [0.0, 0.0]]])   # [1, 2, 2]
    labeled = torch.tensor([[1.0, 0.0]])              # only one labeled joint
    scale = compute_person_scale(kp, labeled, floor=0.1)
    assert abs(scale[0].item() - 0.1) < 1e-5


def test_compute_person_scale_multi_person():
    from src.data.datasets import compute_person_scale

    # P=2; person 0 has 2 labeled joints, person 1 has none
    kp = torch.tensor([
        [[-0.5, 0.0], [0.5, 0.0]],   # extent = 1.0
        [[0.0,  0.0], [0.0, 0.0]],
    ])   # [2, 2, 2]
    labeled = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    scale = compute_person_scale(kp, labeled, floor=0.1)
    assert abs(scale[0].item() - 1.0) < 1e-5
    assert abs(scale[1].item() - 0.1) < 1e-5


# ---------------------------------------------------------------------------
# set-format collate
# ---------------------------------------------------------------------------


def test_set_pose_collate_pads_to_p_max():
    from src.data.datasets import set_pose_collate

    sample_a = {
        "image": torch.zeros(3, 8, 8, dtype=torch.uint8),
        "gt_keypoints": torch.rand(2, K, 2),
        "gt_labeled": torch.ones(2, K),
        "gt_visible": torch.ones(2, K),
        "person_scale": torch.full((2,), 0.3),
        "image_id": 1,
        "img_path": "a",
        "img_hw": torch.tensor([100.0, 200.0]),
    }
    sample_b = {
        "image": torch.zeros(3, 8, 8, dtype=torch.uint8),
        "gt_keypoints": torch.rand(1, K, 2),
        "gt_labeled": torch.ones(1, K),
        "gt_visible": torch.ones(1, K),
        "person_scale": torch.full((1,), 0.3),
        "image_id": 2,
        "img_path": "b",
        "img_hw": torch.tensor([120.0, 220.0]),
    }
    batch = set_pose_collate([sample_a, sample_b])
    assert isinstance(batch["image"], list) and len(batch["image"]) == 2
    assert batch["gt_keypoints"].shape == (2, 2, K, 2)
    assert batch["person_mask"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
    # padded slot in second image is zero
    assert batch["gt_keypoints"][1, 1].abs().sum().item() == 0.0
    assert batch["image_id"].tolist() == [1, 2]


def test_set_pose_collate_handles_zero_person_image():
    from src.data.datasets import set_pose_collate

    s = {
        "image": torch.zeros(3, 8, 8, dtype=torch.uint8),
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
    from src.data.datasets import SetPoseDataset

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
    ds = SetPoseDataset(base)
    out = ds[0]
    kp = out["gt_keypoints"][0]  # [K, 2]
    assert torch.allclose(kp[0], torch.tensor([0.0, 0.0]), atol=1e-5)
    assert torch.allclose(kp[1], torch.tensor([-1.0, -1.0]), atol=1e-5)
    assert torch.allclose(kp[2], torch.tensor([1.0, 1.0]), atol=1e-5)
    assert isinstance(out["image"], torch.Tensor)
    assert out["image"].dtype == torch.uint8
    assert out["gt_labeled"][0, :3].tolist() == [1.0, 1.0, 1.0]
    assert out["gt_visible"][0, :3].tolist() == [1.0, 1.0, 1.0]
