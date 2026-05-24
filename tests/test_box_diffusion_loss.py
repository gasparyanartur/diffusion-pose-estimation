"""Tests for the box set-prediction diffusion loss (I4)."""

from __future__ import annotations

import torch

from src.training.losses import (
    SetBoxDiffusionLoss,
    SetBoxDiffusionLossWeights,
    box_cxcywh_to_xyxy,
    box_matching_cost,
    hungarian_box_match,
    pairwise_giou,
)


# ---------------------------------------------------------------------------
# Coord conversion
# ---------------------------------------------------------------------------


def test_box_cxcywh_to_xyxy_unit_box():
    boxes = torch.tensor([[0.0, 0.0, 2.0, 4.0]])
    xyxy = box_cxcywh_to_xyxy(boxes)
    assert torch.allclose(xyxy, torch.tensor([[-1.0, -2.0, 1.0, 2.0]]))


def test_box_cxcywh_to_xyxy_preserves_batch_shape():
    boxes = torch.randn(3, 5, 4)
    out = box_cxcywh_to_xyxy(boxes)
    assert out.shape == boxes.shape


# ---------------------------------------------------------------------------
# pairwise_giou
# ---------------------------------------------------------------------------


def test_pairwise_giou_identity_is_one():
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [-0.5, -0.5, 0.5, 0.5]])
    giou = pairwise_giou(boxes, boxes)
    assert torch.allclose(giou.diagonal(), torch.ones(2), atol=1e-5)


def test_pairwise_giou_disjoint_is_negative():
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[5.0, 5.0, 6.0, 6.0]])  # far away → giou < 0
    giou = pairwise_giou(a, b)
    assert giou.item() < 0.0
    assert giou.item() >= -1.0 - 1e-5


def test_pairwise_giou_empty_returns_zero_shape():
    a = torch.zeros(0, 4)
    b = torch.zeros(3, 4)
    assert pairwise_giou(a, b).shape == (0, 3)
    assert pairwise_giou(b, a).shape == (3, 0)


# ---------------------------------------------------------------------------
# Matching cost + hungarian_box_match
# ---------------------------------------------------------------------------


def test_box_matching_cost_shape():
    pred = torch.randn(4, 4)
    gt = torch.randn(2, 4)
    obj = torch.randn(4)
    cost = box_matching_cost(pred, gt, obj)
    assert cost.shape == (4, 2)
    assert torch.isfinite(cost).all()


def test_hungarian_box_match_picks_perfect_overlap_rows():
    # 4 slots, 2 GTs. Slots 1 and 3 are exactly the GT boxes; the others
    # are far away. The matcher must pick slots 1 and 3.
    gt = torch.tensor([
        [0.0, 0.0, 0.5, 0.5],
        [1.0, 1.0, 0.5, 0.5],
    ])
    pred = torch.tensor([
        [-3.0, -3.0, 0.1, 0.1],   # far
        [0.0, 0.0, 0.5, 0.5],     # = gt[0]
        [-4.0, -4.0, 0.1, 0.1],   # far
        [1.0, 1.0, 0.5, 0.5],     # = gt[1]
    ])
    obj = torch.zeros(4)
    cost = box_matching_cost(pred, gt, obj)
    slot_idx, gt_idx = hungarian_box_match(cost)
    assert slot_idx.numel() == 2
    pairs = set(zip(slot_idx.tolist(), gt_idx.tolist()))
    assert pairs == {(1, 0), (3, 1)}


def test_hungarian_box_match_empty():
    cost = torch.zeros(3, 0)
    slot_idx, gt_idx = hungarian_box_match(cost)
    assert slot_idx.numel() == 0 and gt_idx.numel() == 0


# ---------------------------------------------------------------------------
# SetBoxDiffusionLoss
# ---------------------------------------------------------------------------


def _make_pred_targets(B: int = 2, N: int = 4, P_max: int = 3) -> tuple[dict, dict]:
    torch.manual_seed(0)
    pred = {
        "pred_boxes": torch.randn(B, N, 4, requires_grad=True),
        "obj_logits": torch.randn(B, N, requires_grad=True),
    }
    gt_boxes = torch.zeros(B, P_max, 4)
    gt_labeled = torch.zeros(B, P_max)
    # batch 0: 2 real GTs, batch 1: 1 real GT.
    gt_boxes[0, 0] = torch.tensor([0.0, 0.0, 0.5, 0.5])
    gt_boxes[0, 1] = torch.tensor([0.3, 0.3, 0.4, 0.4])
    gt_labeled[0, :2] = 1.0
    gt_boxes[1, 0] = torch.tensor([-0.2, 0.1, 0.3, 0.3])
    gt_labeled[1, 0] = 1.0
    targets = {
        "gt_boxes_canon": gt_boxes,
        "gt_box_labeled": gt_labeled,
    }
    return pred, targets


def test_set_box_loss_returns_all_keys_and_finite():
    loss_fn = SetBoxDiffusionLoss()
    pred, targets = _make_pred_targets()
    out = loss_fn(pred, targets)
    expected_keys = {"total", "l1_loss", "giou_loss", "diff_l2_loss", "obj_loss", "ranking_loss", "matched"}
    assert expected_keys.issubset(out.keys())
    for k in expected_keys:
        assert torch.isfinite(out[k]).all(), f"{k} is not finite: {out[k]}"
    assert out["matched"].item() == 3  # 2 + 1


def test_set_box_loss_backward_flows_into_pred_boxes_and_logits():
    loss_fn = SetBoxDiffusionLoss()
    pred, targets = _make_pred_targets()
    out = loss_fn(pred, targets)
    out["total"].backward()
    assert pred["pred_boxes"].grad is not None
    assert pred["obj_logits"].grad is not None
    assert pred["pred_boxes"].grad.abs().sum().item() > 0.0
    assert pred["obj_logits"].grad.abs().sum().item() > 0.0


def test_set_box_loss_empty_gt_yields_zero_box_components_and_finite_obj():
    loss_fn = SetBoxDiffusionLoss()
    B, N = 2, 4
    pred = {
        "pred_boxes": torch.randn(B, N, 4, requires_grad=True),
        "obj_logits": torch.randn(B, N, requires_grad=True),
    }
    targets = {
        "gt_boxes_canon": torch.zeros(B, 3, 4),
        "gt_box_labeled": torch.zeros(B, 3),  # all unlabeled
    }
    out = loss_fn(pred, targets)
    assert out["matched"].item() == 0
    assert out["l1_loss"].item() == 0.0
    assert out["giou_loss"].item() == 0.0
    assert out["diff_l2_loss"].item() == 0.0
    assert torch.isfinite(out["obj_loss"]).all()
    # Only obj_loss should contribute to the total.
    expected_total = loss_fn.weights.obj_bce * out["obj_loss"]
    assert torch.allclose(out["total"], expected_total)


def test_set_box_loss_perfect_predictions_give_near_zero_box_losses():
    # Build pred that exactly equals GT in matched slots; obj_logits very
    # positive on matched slots, very negative elsewhere.
    weights = SetBoxDiffusionLossWeights()
    loss_fn = SetBoxDiffusionLoss(weights)

    gt = torch.tensor([
        [[0.0, 0.0, 0.4, 0.4], [0.5, -0.3, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]],
    ])
    labeled = torch.tensor([[1.0, 1.0, 0.0]])
    pred_boxes = torch.tensor([
        [
            [-0.5, -0.5, 0.1, 0.1],          # filler
            [0.0, 0.0, 0.4, 0.4],            # = gt[0]
            [0.5, -0.3, 0.2, 0.2],           # = gt[1]
            [-0.5, 0.5, 0.1, 0.1],           # filler
        ]
    ], requires_grad=True)
    obj_logits = torch.tensor([[-5.0, 5.0, 5.0, -5.0]], requires_grad=True)
    pred = {"pred_boxes": pred_boxes, "obj_logits": obj_logits}
    targets = {"gt_boxes_canon": gt, "gt_box_labeled": labeled}

    out = loss_fn(pred, targets)
    assert out["matched"].item() == 2
    assert out["l1_loss"].item() < 1e-5
    assert out["giou_loss"].item() < 1e-5
    assert out["diff_l2_loss"].item() < 1e-5
    # obj_loss should be small (logits strongly match the binary target).
    assert out["obj_loss"].item() < 0.1


# ---------------------------------------------------------------------------
# ranking_loss
# ---------------------------------------------------------------------------


def test_ranking_loss_positive_when_unmatched_logits_dominate():
    """ranking_loss > 0 when unmatched slots have higher logits than matched."""
    loss_fn = SetBoxDiffusionLoss()
    # Single image: 1 GT box, 4 slots total.
    # Matched slot is slot 0 (low logit); unmatched slots 1-3 have high logits.
    gt = torch.tensor([[[0.0, 0.0, 0.4, 0.4], [0.0, 0.0, 0.0, 0.0]]])
    labeled = torch.tensor([[1.0, 0.0]])
    pred_boxes = torch.zeros(1, 4, 4, requires_grad=True)
    # slot 0 logit=-5 (matched), slots 1-3 logit=+5 (unmatched)
    obj_logits = torch.tensor([[-5.0, 5.0, 5.0, 5.0]], requires_grad=True)
    pred = {"pred_boxes": pred_boxes, "obj_logits": obj_logits}
    targets = {"gt_boxes_canon": gt, "gt_box_labeled": labeled}
    out = loss_fn(pred, targets)
    assert out["ranking_loss"].item() > 0.0


def test_ranking_loss_small_when_matched_logits_dominate():
    """ranking_loss is near zero when matched slot has much higher logit."""
    loss_fn = SetBoxDiffusionLoss()
    gt = torch.tensor([[[0.0, 0.0, 0.4, 0.4], [0.0, 0.0, 0.0, 0.0]]])
    labeled = torch.tensor([[1.0, 0.0]])
    pred_boxes = torch.zeros(1, 4, 4, requires_grad=True)
    # slot 0 logit=+10 (matched), slots 1-3 logit=-10 (unmatched)
    obj_logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]], requires_grad=True)
    pred = {"pred_boxes": pred_boxes, "obj_logits": obj_logits}
    targets = {"gt_boxes_canon": gt, "gt_box_labeled": labeled}
    out = loss_fn(pred, targets)
    assert out["ranking_loss"].item() < 0.01


def test_ranking_loss_is_zero_when_no_gt():
    """ranking_loss is 0 when there are no GT boxes (no matched slots)."""
    loss_fn = SetBoxDiffusionLoss()
    B, N = 2, 4
    pred = {
        "pred_boxes": torch.randn(B, N, 4),
        "obj_logits": torch.randn(B, N),
    }
    targets = {
        "gt_boxes_canon": torch.zeros(B, 3, 4),
        "gt_box_labeled": torch.zeros(B, 3),
    }
    out = loss_fn(pred, targets)
    assert out["ranking_loss"].item() == 0.0
