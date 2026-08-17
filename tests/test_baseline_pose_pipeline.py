"""Contract tests for the canonical fixed-slot baseline pipeline."""

from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.models.features.cnn import CNNFeatureExtractor, CNNFeatureExtractorConfig
from src.models.pipelines.pose_estimators.baseline import (
    BaselinePosePipeline,
    BaselinePosePipelineConfig,
)
from src.training import TrainingConfig


BATCH_SIZE = 2
NUM_SLOTS = 4
NUM_KEYPOINTS = 14
FEATURE_CHANNELS = 16
CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _make_pipeline(**overrides) -> BaselinePosePipeline:
    defaults = dict(
        backbone=CNNFeatureExtractor(out_channels=FEATURE_CHANNELS),
        spatial_channels=FEATURE_CHANNELS,
        feature_dim=32,
        num_heads=4,
        hidden_dim=64,
        num_slots=NUM_SLOTS,
        num_keypoints=NUM_KEYPOINTS,
    )
    defaults.update(overrides)
    return BaselinePosePipeline(**defaults)


def _fake_batch() -> dict[str, torch.Tensor]:
    max_people = 3
    person_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    return {
        "image": torch.rand(BATCH_SIZE, 3, 32, 32),
        "gt_keypoints": torch.rand(BATCH_SIZE, max_people, NUM_KEYPOINTS, 2) * 2 - 1,
        "gt_labeled": torch.ones(BATCH_SIZE, max_people, NUM_KEYPOINTS),
        "gt_visible": torch.ones(BATCH_SIZE, max_people, NUM_KEYPOINTS),
        "person_scale": torch.full((BATCH_SIZE, max_people), 0.4),
        "person_mask": person_mask,
        "img_hw": torch.full((BATCH_SIZE, 2), 32.0),
        "image_id": torch.arange(BATCH_SIZE),
    }


def test_pipeline_config_builds_from_the_single_root_pipeline_group():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        root_config = compose(config_name="config")

    assert root_config.pipeline._target_.endswith("BaselinePosePipelineConfig")
    pipeline_config = BaselinePosePipelineConfig.from_hydra_config(root_config.pipeline)
    training_config = TrainingConfig.from_hydra_config(root_config.training)
    pipeline = pipeline_config.build(max_epochs=training_config.max_epochs)

    assert isinstance(pipeline, BaselinePosePipeline)
    assert isinstance(pipeline.backbone, CNNFeatureExtractor)


def test_direct_config_builds_a_complete_pipeline():
    config = BaselinePosePipelineConfig(
        backbone=CNNFeatureExtractorConfig(out_channels=FEATURE_CHANNELS),
        feature_dim=32,
        num_heads=4,
        hidden_dim=64,
        num_slots=NUM_SLOTS,
    )
    pipeline = config.build(max_epochs=1)

    assert isinstance(pipeline, BaselinePosePipeline)
    assert pipeline.num_slots == NUM_SLOTS
    assert pipeline.num_keypoints == NUM_KEYPOINTS


def test_forward_returns_the_fixed_slot_tensor_contract():
    output = _make_pipeline()(torch.rand(BATCH_SIZE, 3, 32, 32))

    assert set(output) == {
        "keypoints", "scores", "scores_logits", "keypoint_scores", "vis_logits",
    }
    assert output["keypoints"].shape == (BATCH_SIZE, NUM_SLOTS, NUM_KEYPOINTS, 2)
    assert output["scores"].shape == (BATCH_SIZE, NUM_SLOTS)
    assert output["scores_logits"].shape == (BATCH_SIZE, NUM_SLOTS)
    assert output["keypoint_scores"].shape == (BATCH_SIZE, NUM_SLOTS, NUM_KEYPOINTS)
    assert output["vis_logits"].shape == (BATCH_SIZE, NUM_SLOTS, NUM_KEYPOINTS)
    assert (output["keypoints"] >= -1).all() and (output["keypoints"] <= 1).all()
    assert (output["scores"] >= 0).all() and (output["scores"] <= 1).all()
    assert (output["keypoint_scores"] >= 0).all() and (output["keypoint_scores"] <= 1).all()


def test_training_step_uses_matching_loss_and_returns_a_differentiable_loss():
    pipeline = _make_pipeline()
    loss = pipeline.training_step(_fake_batch(), batch_idx=0)

    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert torch.isfinite(pipeline.matched_oks(pipeline(_fake_batch()["image"]), _fake_batch()))


def test_test_step_updates_the_evaluation_boundary_and_resets_metrics():
    class _MetricRecorder:
        def __init__(self) -> None:
            self.updated = False
            self.reset_called = False

        def update(self, **kwargs) -> None:
            self.updated = True
            assert kwargs["pred_kpts"].shape == (BATCH_SIZE, NUM_SLOTS, NUM_KEYPOINTS, 2)
            assert kwargs["pred_scores"].shape == (BATCH_SIZE, NUM_SLOTS, NUM_KEYPOINTS)
            assert kwargs["bbox_scores"].shape == (BATCH_SIZE, NUM_SLOTS)

        def compute(self) -> dict[str, torch.Tensor]:
            return {"crowdpose/AP": torch.tensor(0.0)}

        def reset(self) -> None:
            self.reset_called = True

    pipeline = _make_pipeline()
    recorder = _MetricRecorder()
    pipeline._test_metric = recorder  # type: ignore[assignment]
    pipeline.test_step(_fake_batch(), batch_idx=0)
    pipeline.on_test_epoch_end()

    assert recorder.updated
    assert recorder.reset_called
