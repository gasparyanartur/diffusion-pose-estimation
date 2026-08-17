"""Contract tests for the canonical tensor-native graph-pose pipeline."""

from __future__ import annotations

import pytest
import torch

from src.models.features.cnn import CNNFeatureExtractor, CNNFeatureExtractorConfig
from src.models.pipelines.pose_estimators.graph import (
    GraphPosePipeline,
    GraphPosePipelineConfig,
)


BATCH_SIZE = 2
NUM_SLOTS = 4
NUM_KEYPOINTS = 14
FEATURE_CHANNELS = 16


def _make_pipeline(**overrides) -> GraphPosePipeline:
    defaults = dict(
        backbone=CNNFeatureExtractor(out_channels=FEATURE_CHANNELS),
        num_slots=NUM_SLOTS,
        num_keypoints=NUM_KEYPOINTS,
        dim=32,
        num_heads=4,
        num_trunk_blocks=1,
        num_joint_blocks=1,
        time_dim=32,
        image_token_dim=FEATURE_CHANNELS,
        spatial_feat_dim=FEATURE_CHANNELS,
        dim_vis=16,
        dim_edge=16,
        num_refiner_layers=1,
        fourier_box_bands=4,
        fourier_joint_bands=4,
        fourier_sigma_bands=4,
        dropout=0.0,
        num_inference_steps=3,
    )
    defaults.update(overrides)
    return GraphPosePipeline(**defaults)


def _fake_batch() -> dict:
    max_people = 3
    boxes = torch.zeros(BATCH_SIZE, max_people, 4)
    boxes[:, :2] = torch.tensor([0.0, 0.0, 0.4, 0.4])
    joints = torch.rand(BATCH_SIZE, max_people, NUM_KEYPOINTS, 2) * 0.8 - 0.4
    labels = torch.zeros(BATCH_SIZE, max_people)
    labels[:, :2] = 1
    return {
        "image": torch.rand(BATCH_SIZE, 3, 32, 32),
        "gt_boxes_canon": boxes,
        "gt_joints_canon": joints,
        "gt_box_labeled": labels,
        "gt_visibility": torch.ones(BATCH_SIZE, max_people, NUM_KEYPOINTS),
        "img_hw": torch.full((BATCH_SIZE, 2), 32.0),
        "image_id": torch.arange(BATCH_SIZE),
    }


def test_pipeline_config_builds_complete_model():
    config = GraphPosePipelineConfig(
        backbone=CNNFeatureExtractorConfig(out_channels=FEATURE_CHANNELS),
        num_slots=NUM_SLOTS,
        dim=32,
        num_heads=4,
        num_trunk_blocks=1,
        num_joint_blocks=1,
        time_dim=32,
        dim_vis=16,
        dim_edge=16,
        num_refiner_layers=1,
        fourier_box_bands=4,
        fourier_joint_bands=4,
        fourier_sigma_bands=4,
    )
    pipeline = config.build(max_epochs=1)
    assert isinstance(pipeline, GraphPosePipeline)
    assert isinstance(pipeline.backbone, CNNFeatureExtractor)
    assert pipeline.num_slots == NUM_SLOTS


def test_forward_returns_finite_graph_pose_prediction_contract():
    pipeline = _make_pipeline()
    output = pipeline(
        torch.rand(BATCH_SIZE, 3, 32, 32),
        torch.randn(BATCH_SIZE, NUM_SLOTS, 4),
        torch.randn(BATCH_SIZE, NUM_SLOTS, NUM_KEYPOINTS, 2),
        torch.tensor([0.1, 0.5]),
    )
    assert set(output) == {"pred_boxes", "pred_joints_rel", "obj_logits"}
    assert output["pred_boxes"].shape == (BATCH_SIZE, NUM_SLOTS, 4)
    assert output["pred_joints_rel"].shape == (BATCH_SIZE, NUM_SLOTS, NUM_KEYPOINTS, 2)
    assert output["obj_logits"].shape == (BATCH_SIZE, NUM_SLOTS)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_schedule_and_sampling_contracts():
    pipeline = _make_pipeline()
    schedule = pipeline.edm_sigma_schedule(3, sigma_min=0.01, sigma_max=1.0)
    assert schedule.shape == (4,)
    assert schedule[-1].item() == 0
    images = torch.rand(BATCH_SIZE, 3, 32, 32)
    for steps in (1, 3):
        output = pipeline.sample(images, num_inference_steps=steps)
        assert output["pred_boxes"].shape == (BATCH_SIZE, NUM_SLOTS, 4)
        assert output["pred_joints_rel"].shape == (BATCH_SIZE, NUM_SLOTS, NUM_KEYPOINTS, 2)
        assert torch.isfinite(output["pred_boxes"]).all()
    with pytest.raises(ValueError, match="num_inference_steps"):
        pipeline.sample(images, num_inference_steps=0)


def test_training_step_returns_a_finite_differentiable_loss():
    loss = _make_pipeline().training_step(_fake_batch(), batch_idx=0)
    assert torch.isfinite(loss)
    assert loss.requires_grad
