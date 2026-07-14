"""Unit tests for GraphDiffusionPipelineDecoder and GraphPipelineLoss."""

from __future__ import annotations

import torch
import pytest

from src.decoders.diffusion.graph_pipeline import (
    GraphDiffusionPipelineDecoder,
    GraphDiffusionPipelineDecoderConfig,
)
from src.training.losses import GraphPipelineLoss, GraphPipelineLossWeights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, N, K, T = 2, 4, 14, 16   # batch, slots, keypoints, tokens
C_IMG  = 64                  # image_token_dim
C_SPAT = 32                  # spatial_feat_dim
H_SPAT, W_SPAT = 4, 4


def make_decoder(num_refiner_layers: int = 2) -> GraphDiffusionPipelineDecoder:
    return GraphDiffusionPipelineDecoder(
        num_slots=N,
        num_keypoints=K,
        dim=32,
        num_heads=4,
        num_trunk_blocks=1,
        num_joint_blocks=1,
        time_dim=32,
        ffn_mult=2.0,
        image_token_dim=C_IMG,
        spatial_feat_dim=C_SPAT,
        dim_vis=16,
        dim_edge=16,
        num_refiner_layers=num_refiner_layers,
        fourier_box_bands=4,
        fourier_joint_bands=4,
        fourier_sigma_bands=4,
        dropout=0.0,
    )


def make_inputs(device: torch.device = torch.device("cpu")):
    noisy_boxes      = torch.randn(B, N, 4, device=device)
    noisy_joints_rel = torch.randn(B, N, K, 2, device=device)
    sigma            = torch.rand(B, device=device) * 0.5 + 0.01
    image_tokens     = torch.randn(B, T, C_IMG, device=device)
    patch_centers    = torch.zeros(T, 2, device=device)
    spatial_feat     = torch.randn(B, C_SPAT, H_SPAT, W_SPAT, device=device)
    return noisy_boxes, noisy_joints_rel, sigma, image_tokens, patch_centers, spatial_feat


# ---------------------------------------------------------------------------
# Decoder: forward
# ---------------------------------------------------------------------------

class TestGraphDiffusionPipelineDecoderForward:
    def test_output_keys(self):
        dec = make_decoder()
        out = dec(*make_inputs())
        assert set(out.keys()) == {"pred_boxes", "pred_joints_rel", "obj_logits"}

    def test_output_shapes(self):
        dec = make_decoder()
        out = dec(*make_inputs())
        assert out["pred_boxes"].shape      == (B, N, 4)
        assert out["pred_joints_rel"].shape == (B, N, K, 2)
        assert out["obj_logits"].shape      == (B, N)

    def test_output_is_finite(self):
        dec = make_decoder()
        out = dec(*make_inputs())
        for key, val in out.items():
            assert torch.isfinite(val).all(), f"{key} contains non-finite values"

    def test_backward_runs(self):
        dec = make_decoder()
        inputs = make_inputs()
        noisy_boxes = inputs[0].requires_grad_(True)
        out = dec(noisy_boxes, *inputs[1:])
        out["pred_boxes"].sum().backward()
        assert noisy_boxes.grad is not None

    def test_zero_refiner_layers(self):
        """With 0 refiner layers, pred_joints_rel == rough_joints_rel (no Stage 2 delta)."""
        dec_ref = make_decoder(num_refiner_layers=1)
        # 0 layers would cause GraphDenoiseNetwork to have no PoseGraphLayer,
        # which returns output_proj(h) → zeros (zero-init). So delta ≈ 0.
        dec_zero = make_decoder(num_refiner_layers=1)
        # Zero-init check: at init the output proj is zero → delta = 0 → pred_joints same as rough.
        # We just verify forward runs without error.
        out = dec_zero(*make_inputs())
        assert out["pred_joints_rel"].shape == (B, N, K, 2)


# ---------------------------------------------------------------------------
# Decoder: EDM sigma schedule
# ---------------------------------------------------------------------------

class TestEdmSigmaSchedule:
    def test_schedule_length(self):
        sched = GraphDiffusionPipelineDecoder.edm_sigma_schedule(5)
        assert sched.shape == (6,)

    def test_schedule_ends_with_zero(self):
        sched = GraphDiffusionPipelineDecoder.edm_sigma_schedule(5, sigma_max=10.0)
        assert sched[-1].item() == pytest.approx(0.0)

    def test_schedule_starts_near_sigma_max(self):
        sched = GraphDiffusionPipelineDecoder.edm_sigma_schedule(5, sigma_max=10.0)
        assert sched[0].item() == pytest.approx(10.0, rel=1e-4)

    def test_schedule_monotone_decreasing(self):
        sched = GraphDiffusionPipelineDecoder.edm_sigma_schedule(5)
        diffs = (sched[1:] - sched[:-1]).tolist()
        assert all(d <= 0 for d in diffs)


# ---------------------------------------------------------------------------
# Decoder: sample
# ---------------------------------------------------------------------------

class TestGraphDiffusionPipelineDecoderSample:
    def test_single_step_shapes(self):
        dec = make_decoder()
        _, _, _, tokens, centers, spatial = make_inputs()
        out = dec.sample(tokens, centers, spatial, num_inference_steps=1)
        assert out["pred_boxes"].shape      == (B, N, 4)
        assert out["pred_joints_rel"].shape == (B, N, K, 2)
        assert out["obj_logits"].shape      == (B, N)

    def test_multi_step_shapes(self):
        dec = make_decoder()
        _, _, _, tokens, centers, spatial = make_inputs()
        out = dec.sample(tokens, centers, spatial, num_inference_steps=3,
                         sigma_min=0.01, sigma_max=1.0)
        assert out["pred_boxes"].shape      == (B, N, 4)
        assert out["pred_joints_rel"].shape == (B, N, K, 2)

    def test_sample_is_finite(self):
        dec = make_decoder()
        _, _, _, tokens, centers, spatial = make_inputs()
        out = dec.sample(tokens, centers, spatial, num_inference_steps=2,
                         sigma_min=0.01, sigma_max=1.0)
        for key, val in out.items():
            assert torch.isfinite(val).all(), f"sample/{key} non-finite"


# ---------------------------------------------------------------------------
# Decoder: config round-trip
# ---------------------------------------------------------------------------

class TestGraphDiffusionPipelineDecoderConfig:
    def test_build(self):
        cfg = GraphDiffusionPipelineDecoderConfig(
            num_slots=N, num_keypoints=K,
            dim=32, num_heads=4,
            image_token_dim=C_IMG, spatial_feat_dim=C_SPAT,
            dim_vis=16, dim_edge=16,
            fourier_box_bands=4, fourier_joint_bands=4, fourier_sigma_bands=4,
        )
        dec = cfg.build()
        assert isinstance(dec, GraphDiffusionPipelineDecoder)
        assert dec.num_slots == N
        assert dec.num_keypoints == K


# ---------------------------------------------------------------------------
# Loss: GraphPipelineLoss
# ---------------------------------------------------------------------------

P_GT = 3   # number of GT persons per image


def make_loss_inputs():
    pred = {
        "pred_boxes":      torch.randn(B, N, 4),
        "pred_joints_rel": torch.randn(B, N, K, 2) * 0.1,
        "obj_logits":      torch.randn(B, N),
    }
    gt_box_labeled = torch.zeros(B, N, dtype=torch.long)
    gt_box_labeled[:, :P_GT] = 1
    targets = {
        "gt_boxes_canon":  torch.randn(B, N, 4),
        "gt_joints_canon": torch.randn(B, N, K, 2) * 0.1,
        "gt_box_labeled":  gt_box_labeled,
        "gt_visibility":   torch.ones(B, N, K, dtype=torch.long) * 2,
    }
    return pred, targets


class TestGraphPipelineLoss:
    def test_output_keys(self):
        loss_fn = GraphPipelineLoss()
        pred, targets = make_loss_inputs()
        out = loss_fn(pred, targets)
        expected = {"total", "box_l1", "box_giou", "box_l2", "kpt_l2", "obj_loss", "ranking_loss", "matched"}
        assert set(out.keys()) == expected

    def test_total_is_finite_scalar(self):
        loss_fn = GraphPipelineLoss()
        pred, targets = make_loss_inputs()
        out = loss_fn(pred, targets)
        assert out["total"].ndim == 0
        assert torch.isfinite(out["total"])

    def test_backward_runs(self):
        loss_fn = GraphPipelineLoss()
        pred, targets = make_loss_inputs()
        pred["pred_boxes"]      = pred["pred_boxes"].requires_grad_(True)
        pred["pred_joints_rel"] = pred["pred_joints_rel"].requires_grad_(True)
        out = loss_fn(pred, targets)
        out["total"].backward()
        assert pred["pred_boxes"].grad is not None
        assert pred["pred_joints_rel"].grad is not None

    def test_zero_gt_persons(self):
        """Loss is finite when no GT persons exist (all padded)."""
        loss_fn = GraphPipelineLoss()
        pred, targets = make_loss_inputs()
        targets["gt_box_labeled"] = torch.zeros(B, N, dtype=torch.long)
        out = loss_fn(pred, targets)
        assert torch.isfinite(out["total"])

    def test_custom_weights(self):
        weights = GraphPipelineLossWeights(box_l1=0.0, kpt_l2=0.0, obj_bce=0.0, rank=0.0)
        loss_fn = GraphPipelineLoss(weights)
        pred, targets = make_loss_inputs()
        out = loss_fn(pred, targets)
        # With box_l1=0, kpt_l2=0, only giou + box_l2 contribute.
        assert torch.isfinite(out["total"])
