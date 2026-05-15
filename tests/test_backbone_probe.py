"""Tests for ProbeHead, BackboneProbeModule, and compute_probe_diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.cnn import CNNFeatureExtractor
from src.training.backbone_probe import BackboneProbeModule, ProbeHead
from src.training.diagnostics import compute_probe_diagnostics
from src.training.losses import SetPredictionLossWeights
from src.metrics import CROWDPOSE_SIGMAS

import numpy as np

K = 14
SIGMAS = torch.from_numpy(CROWDPOSE_SIGMAS).float()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_probe_module(N: int = 20, feature_dim: int = 256) -> BackboneProbeModule:
    from src.decoders.pose_slot import PoseSlotDecoder
    backbone = CNNFeatureExtractor(out_channels=128)
    head = ProbeHead(in_channels=128, feature_dim=feature_dim, pool_h=8, pool_w=8, num_heads=4, conv_depth=1)
    decoder = PoseSlotDecoder(in_dim=feature_dim, num_slots=N, num_keypoints=K, hidden_dim=256)
    return BackboneProbeModule(
        backbone=backbone,
        probe_head=head,
        slot_decoder=decoder,
        weights=SetPredictionLossWeights(),
        lr=1e-3,
    )


def _fake_batch(B: int = 2, P_max: int = 3, H: int = 64, W: int = 48) -> dict:
    return {
        "image": torch.rand(B, 3, H, W),
        "gt_keypoints": torch.rand(B, P_max, K, 2) * 2 - 1,
        "gt_labeled": torch.ones(B, P_max, K),
        "gt_visible": torch.ones(B, P_max, K),
        "person_scale": torch.full((B, P_max), 0.4),
        "person_mask": torch.ones(B, P_max),
        "img_hw": torch.tensor([[100.0, 200.0]] * B),
        "image_id": torch.arange(1, B + 1, dtype=torch.long),
        "img_path": [chr(ord("a") + i) for i in range(B)],
    }


def _fake_pred(B: int = 2, N: int = 20) -> dict[str, torch.Tensor]:
    return {
        "keypoints":       torch.rand(B, N, K, 2) * 2 - 1,
        "scores":          torch.sigmoid(torch.randn(B, N)),
        "scores_logits":   torch.randn(B, N),
        "keypoint_scores": torch.sigmoid(torch.randn(B, N, K)),
        "vis_logits":      torch.randn(B, N, K),
    }


# ---------------------------------------------------------------------------
# ProbeHead
# ---------------------------------------------------------------------------


def test_probe_head_output_shape():
    head = ProbeHead(in_channels=128, feature_dim=256, pool_h=16, pool_w=16, num_heads=8)
    feat = {"spatial": torch.rand(3, 128, 32, 24)}
    out = head(feat)
    assert out.shape == (3, 256), f"Expected [3,256], got {out.shape}"


def test_probe_head_varied_input_sizes():
    """ProbeHead must handle any spatial resolution via adaptive pooling."""
    head = ProbeHead(in_channels=64, feature_dim=128, pool_h=8, pool_w=8, num_heads=4, conv_depth=1)
    for h, w in [(12, 10), (40, 30), (7, 7)]:
        feat = {"spatial": torch.rand(2, 64, h, w)}
        out = head(feat)
        assert out.shape == (2, 128), f"Failed for input {h}×{w}"


def test_probe_head_conv_depth_zero():
    """conv_depth=0 should still work (skip conv blocks entirely)."""
    head = ProbeHead(in_channels=32, feature_dim=64, pool_h=4, pool_w=4, num_heads=4, conv_depth=0)
    out = head({"spatial": torch.rand(1, 32, 10, 10)})
    assert out.shape == (1, 64)


def test_probe_head_gradient_flows():
    head = ProbeHead(in_channels=32, feature_dim=64, pool_h=4, pool_w=4, num_heads=4, conv_depth=1)
    feat = {"spatial": torch.rand(2, 32, 8, 8, requires_grad=False)}
    out = head(feat)
    loss = out.sum()
    loss.backward()
    # All head params should have gradients
    for name, p in head.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# BackboneProbeModule
# ---------------------------------------------------------------------------


def test_backbone_probe_forward_shapes():
    module = _make_probe_module(N=20)
    batch = _fake_batch(B=2)
    pred = module._forward(batch)
    assert pred["keypoints"].shape == (2, 20, K, 2)
    assert pred["scores"].shape == (2, 20)
    assert pred["vis_logits"].shape == (2, 20, K)


def test_backbone_probe_training_step_runs():
    module = _make_probe_module(N=20)
    batch = _fake_batch(B=2)
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


def test_backbone_probe_validation_step_runs():
    module = _make_probe_module(N=20)
    batch = _fake_batch(B=2)
    module.validation_step(batch, batch_idx=0)


def test_backbone_probe_configure_optimizers():
    module = _make_probe_module()
    module.lr_T_max = 50
    opt = module.configure_optimizers()
    assert "optimizer" in opt
    assert "lr_scheduler" in opt


def test_backbone_probe_gt_sanity_check_shapes():
    from src.decoders.pose_slot import PoseSlotDecoder
    backbone = CNNFeatureExtractor(out_channels=128)
    head = ProbeHead(in_channels=128, feature_dim=64, pool_h=4, pool_w=4, num_heads=4)
    decoder = PoseSlotDecoder(in_dim=64, num_slots=10, num_keypoints=K, hidden_dim=64)
    module = BackboneProbeModule(
        backbone=backbone, probe_head=head, slot_decoder=decoder, gt_sanity_check=True,
    )
    batch = _fake_batch(B=2, P_max=3)
    pred = module._forward(batch)
    assert pred["keypoints"].shape == (2, 10, K, 2)


# ---------------------------------------------------------------------------
# BackboneProbeConfig Hydra wiring
# ---------------------------------------------------------------------------


def test_backbone_probe_config_build():
    from src.training.training_config import BackboneProbeConfig
    from src.data import CrowdPoseDataConfig

    data_cfg = CrowdPoseDataConfig(
        root="data",
        train_ann="crowdpose/annotations/crowdpose_train.json",
        val_ann="crowdpose/annotations/crowdpose_val.json",
        test_ann="crowdpose/annotations/crowdpose_test.json",
        img_dir="crowdpose/images",
        target_format="set",
    )
    data_module = data_cfg.build()
    backbone = CNNFeatureExtractor(out_channels=128)
    probe_cfg = BackboneProbeConfig(
        feature_dim=64,
        pool_h=4,
        pool_w=4,
        num_heads=4,
        hidden_dim=64,
        num_slots=5,
    )
    module = probe_cfg.build(backbone, data_module)
    assert isinstance(module, BackboneProbeModule)
    assert module.slot_decoder.num_slots == 5
    assert module.probe_head.feature_dim == 64


# ---------------------------------------------------------------------------
# compute_probe_diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_keys_present():
    pred = _fake_pred(B=2, N=20)
    batch = _fake_batch(B=2, P_max=3)
    diag = compute_probe_diagnostics(pred, batch, SIGMAS)
    expected = {"missed_person_rate", "duplicate_person_rate", "vis_accuracy", "vis_tp", "vis_fp", "vis_tn", "vis_fn"}
    assert expected.issubset(diag.keys())
    for j in range(K):
        assert f"per_joint_oks_{j:02d}" in diag


def test_diagnostics_all_gt_matched_low_missed_rate():
    """When model is given GT predictions, missed rate should be 0."""
    B, P_max, N = 2, 3, 20
    batch = _fake_batch(B=B, P_max=P_max)
    obj_logits = torch.full((B, N), -10.0)
    obj_logits[:, :P_max] = 10.0
    # Copy GT keypoints into first P_max slots; fill remainder with -1
    gt_kp = batch["gt_keypoints"]  # [B, P_max, K, 2]
    kp = torch.full((B, N, K, 2), -1.0)
    kp[:, :P_max] = gt_kp
    pred = {
        "keypoints":       kp,
        "scores":          torch.sigmoid(obj_logits),
        "scores_logits":   obj_logits,
        "keypoint_scores": torch.sigmoid(torch.zeros(B, N, K)),
        "vis_logits":      torch.zeros(B, N, K),
    }
    diag = compute_probe_diagnostics(pred, batch, SIGMAS, objectness_threshold=0.5)
    assert diag["missed_person_rate"].item() == 0.0


def test_diagnostics_no_gt():
    """P=0 for all images should return zero for all rates."""
    pred = _fake_pred(B=2, N=20)
    batch = _fake_batch(B=2, P_max=3)
    batch["person_mask"] = torch.zeros(2, 3)
    diag = compute_probe_diagnostics(pred, batch, SIGMAS)
    assert diag["missed_person_rate"].item() == 0.0
    assert diag["duplicate_person_rate"].item() == 0.0


def test_diagnostics_all_unmatched_full_missed_rate():
    """When objectness is below threshold for all slots, all GT persons are missed."""
    B, P_max, N = 1, 3, 20
    batch = _fake_batch(B=B, P_max=P_max)
    obj_logits = torch.full((B, N), -10.0)  # all below 0.5 threshold
    pred = {
        "keypoints":       torch.rand(B, N, K, 2) * 2 - 1,
        "scores":          torch.sigmoid(obj_logits),
        "scores_logits":   obj_logits,
        "keypoint_scores": torch.sigmoid(torch.zeros(B, N, K)),
        "vis_logits":      torch.zeros(B, N, K),
    }
    diag = compute_probe_diagnostics(pred, batch, SIGMAS, objectness_threshold=0.5)
    assert diag["missed_person_rate"].item() == 1.0


def test_diagnostics_vis_accuracy_perfect():
    """When visibility logits agree with GT, accuracy should be 1.0."""
    B, P_max, N = 1, 2, 20
    batch = _fake_batch(B=B, P_max=P_max)
    # Force matching: first P_max slots get high objectness and GT keypoints
    obj_logits = torch.full((B, N), -10.0)
    obj_logits[:, :P_max] = 10.0
    gt_vis = batch["gt_visible"]  # [B, P_max, K]
    # logits = +10 where visible, -10 where not
    vis_logits = torch.where(gt_vis.bool(), torch.tensor(10.0), torch.tensor(-10.0))
    vis_full = torch.full((B, N, K), -10.0)
    vis_full[:, :P_max] = vis_logits

    kp = torch.full((B, N, K, 2), -1.0)
    kp[:, :P_max] = batch["gt_keypoints"]
    pred = {
        "keypoints":       kp,
        "scores":          torch.sigmoid(obj_logits),
        "scores_logits":   obj_logits,
        "keypoint_scores": torch.sigmoid(vis_full),
        "vis_logits":      vis_full,
    }
    diag = compute_probe_diagnostics(pred, batch, SIGMAS)
    assert diag["vis_accuracy"].item() == pytest.approx(1.0, abs=0.01)
