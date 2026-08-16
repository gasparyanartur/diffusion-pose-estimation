"""Tests for ConvProbe, QueryProbe, BackboneProbePipeline, and compute_probe_diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.common.positional import make_patch_centers
from src.models.features.cnn import CNNFeatureExtractor
from src.models.pipelines.pose_estimators.backbone_probe import (
    BackboneProbePipeline,
    ConvProbe,
    QueryProbe,
)
from src.eval.diagnostics import compute_probe_diagnostics
from src.training.losses import SetPredictionLossWeights
from src.eval.metrics import CROWDPOSE_SIGMAS

import numpy as np

K = 14
SIGMAS = torch.from_numpy(CROWDPOSE_SIGMAS).float()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_probe_module(N: int = 20, feature_dim: int = 256) -> BackboneProbePipeline:
    backbone = CNNFeatureExtractor(out_channels=128)
    head = ConvProbe(
        in_channels=128, feature_dim=feature_dim, target_size=8,
        num_heads=4, num_conv_layers=1, num_slots=N, num_keypoints=K, hidden_dim=64,
    )
    return BackboneProbePipeline(
        backbone=backbone,
        probe_head=head,
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
# ConvNeXtBlock
# ---------------------------------------------------------------------------


def test_convnext_block_dropout_does_not_change_shape():
    from src.models.pipelines.pose_estimators.backbone_probe import ConvNeXtBlock
    block = ConvNeXtBlock(dim=32, ffn_mult=4, dropout=0.5)
    x = torch.rand(2, 32, 8, 8)
    out = block(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# CrossAttentionBlock
# ---------------------------------------------------------------------------


def test_cross_attention_block_shape():
    from src.models.pipelines.pose_estimators.backbone_probe import CrossAttentionBlock
    block = CrossAttentionBlock(dim=64, num_heads=4)
    q  = torch.rand(2, 10, 64)
    kv = torch.rand(2, 20, 64)
    out = block(q, kv)
    assert out.shape == (2, 10, 64)


def test_cross_attention_block_single_kv_token():
    """KV sequence length of 1 (used in joint→instance step) must not error."""
    from src.models.pipelines.pose_estimators.backbone_probe import CrossAttentionBlock
    block = CrossAttentionBlock(dim=64, num_heads=4)
    q  = torch.rand(4, 14, 64)
    kv = torch.rand(4, 1, 64)
    out = block(q, kv)
    assert out.shape == (4, 14, 64)


# ---------------------------------------------------------------------------
# QueryProbe
# ---------------------------------------------------------------------------


def test_query_probe_output_shapes():
    head = QueryProbe(in_channels=128, feature_dim=64, num_heads=4, num_slots=10, num_keypoints=K)
    spatial = torch.rand(2, 128, 16, 12)
    feat = {
        "spatial": spatial,
        "patch_centers": make_patch_centers(16, 12, spatial.device, spatial.dtype),
    }
    out = head(feat)
    assert out["keypoints"].shape == (2, 10, K, 2)
    assert out["scores"].shape == (2, 10)
    assert out["vis_logits"].shape == (2, 10, K)


def test_query_probe_varied_input_sizes():
    head = QueryProbe(in_channels=64, feature_dim=64, num_heads=4, num_slots=5, num_keypoints=K)
    for h, w in [(8, 6), (16, 12), (32, 24)]:
        spatial = torch.rand(2, 64, h, w)
        out = head({
            "spatial": spatial,
            "patch_centers": make_patch_centers(h, w, spatial.device, spatial.dtype),
        })
        assert out["scores"].shape == (2, 5), f"Failed for {h}x{w}"


def test_query_probe_gradient_flows():
    head = QueryProbe(in_channels=32, feature_dim=64, num_heads=4, num_slots=5, num_keypoints=K)
    spatial = torch.rand(2, 32, 8, 6, requires_grad=True)
    out = head({
        "spatial": spatial,
        "patch_centers": make_patch_centers(8, 6, spatial.device, spatial.detach().dtype),
    })
    
    loss = out["scores"].sum() + out["keypoints"].sum() + out["vis_logits"].sum()
    loss.backward()
    for name, p in head.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# ConvProbe
# ---------------------------------------------------------------------------


def test_probe_head_output_shape():
    head = ConvProbe(in_channels=128, feature_dim=256, target_size=16, num_heads=8)
    feat = {"spatial": torch.rand(3, 128, 32, 24)}
    out = head(feat)
    assert out["keypoints"].shape == (3, head.num_slots, head.num_keypoints, 2)
    assert out["scores"].shape == (3, head.num_slots)


def test_probe_head_varied_input_sizes():
    """ProbeHead must handle any spatial resolution."""
    head = ConvProbe(in_channels=64, feature_dim=128, target_size=8, num_heads=4, num_conv_layers=1)
    for h, w in [(12, 10), (40, 30), (7, 7)]:
        feat = {"spatial": torch.rand(2, 64, h, w)}
        out = head(feat)
        assert out["scores"].shape == (2, head.num_slots), f"Failed for input {h}×{w}"


def test_probe_head_zero_conv_layers():
    """num_conv_layers=0 should still work (skip conv blocks, go straight to attention)."""
    head = ConvProbe(in_channels=32, feature_dim=64, target_size=4, num_heads=4, num_conv_layers=0)
    out = head({"spatial": torch.rand(1, 32, 10, 10)})
    assert out["scores"].shape == (1, head.num_slots)


def test_probe_head_large_input_pools_down():
    """Input larger than target_size should be progressively pooled down."""
    head = ConvProbe(in_channels=64, feature_dim=128, target_size=8, num_conv_layers=5, num_heads=4)
    out = head({"spatial": torch.rand(2, 64, 64, 64)})
    assert out["scores"].shape == (2, head.num_slots)


def test_probe_head_small_input_no_extra_pool():
    """Input already at target_size: no pooling should occur."""
    head = ConvProbe(in_channels=32, feature_dim=64, target_size=16, num_conv_layers=3, num_heads=4)
    out = head({"spatial": torch.rand(2, 32, 8, 8)})
    assert out["scores"].shape == (2, head.num_slots)


def test_probe_head_gradient_flows():
    head = ConvProbe(in_channels=32, feature_dim=64, target_size=4, num_heads=4, num_conv_layers=1)
    feat = {"spatial": torch.rand(2, 32, 8, 8, requires_grad=False)}
    out = head(feat)
    loss = out["scores"].sum() + out["keypoints"].sum() + out["vis_logits"].sum()
    loss.backward()
    # All head params should have gradients
    for name, p in head.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# BackboneProbePipeline
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


def test_backbone_probe_training_step_with_augmentation():
    """training_step should still produce a finite loss when augmentation is active."""
    backbone = CNNFeatureExtractor(out_channels=128)
    head = ConvProbe(
        in_channels=128, feature_dim=64, target_size=4, num_heads=4, num_conv_layers=1,
        num_slots=20, num_keypoints=K, hidden_dim=64, proj_dropout=0.3,
    )
    module = BackboneProbePipeline(
        backbone=backbone, probe_head=head,
        weights=SetPredictionLossWeights(), lr=1e-3,
        noise_std=0.05, brightness=0.3, contrast=0.3, saturation=0.2,
    )
    batch = _fake_batch(B=2)
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------


def test_augmentations_gaussian_noise_shape_and_clamp():
    from src.data.augmentations import add_gaussian_noise
    img = torch.rand(4, 3, 32, 32)
    out = add_gaussian_noise(img, std=0.1)
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_augmentations_color_jitter_shape():
    from src.data.augmentations import color_jitter
    img = torch.rand(4, 3, 32, 32)
    out = color_jitter(img, brightness=0.3, contrast=0.3, saturation=0.2)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


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
    backbone = CNNFeatureExtractor(out_channels=128)
    head = ConvProbe(
        in_channels=128, feature_dim=64, target_size=4, num_heads=4,
        num_slots=10, num_keypoints=K, hidden_dim=64,
    )
    module = BackboneProbePipeline(
        backbone=backbone, probe_head=head, gt_sanity_check=True,
    )
    batch = _fake_batch(B=2, P_max=3)
    pred = module._forward(batch)
    assert pred["keypoints"].shape == (2, 10, K, 2)


# ---------------------------------------------------------------------------
# BackboneProbeConfig Hydra wiring
# ---------------------------------------------------------------------------


def test_backbone_probe_config_build():
    from src.models.pipelines.pose_estimators.backbone_probe import BackboneProbeConfig
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
        target_size=4,
        num_heads=4,
        hidden_dim=64,
        num_slots=5,
    )
    module = probe_cfg.build(backbone, data_module)
    assert isinstance(module, BackboneProbePipeline)
    assert module.probe_head.num_slots == 5
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
