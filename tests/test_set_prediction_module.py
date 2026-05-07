"""Tests for SetPredictionTrainingModule + Hydra config wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data import CrowdPoseDataConfig, DataConfig
from src.pose_estimator import PoseEstimatorConfig
from src.training import (
    SetPredictionLossWeights,
    SetPredictionTrainingModule,
    SetPredictionTrainingModuleConfig,
    TrainingModuleConfig,
)


CONF_DIR = str(_ROOT / "conf")
K = 14


def _build_estimator():
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(
            config_name="config",
            overrides=["pose_estimator=cnn_set_prediction"],
        )
    return PoseEstimatorConfig.from_hydra_config(cfg.pose_estimator).build()


def _fake_set_batch(B: int = 2, P_max: int = 3) -> dict:
    return {
        "image": torch.rand(B, 3, 64, 48),
        "gt_keypoints": (torch.rand(B, P_max, K, 2) * 2 - 1),
        "gt_labeled": torch.ones(B, P_max, K),
        "gt_visible": torch.ones(B, P_max, K),
        "person_scale": torch.full((B, P_max), 0.4),
        "person_mask": torch.tensor(
            [[1, 1, 0], [1, 0, 0]], dtype=torch.float32,
        )[:B],
        "img_hw": torch.tensor([[100.0, 200.0]] * B),
        "image_id": torch.tensor([1, 2][:B], dtype=torch.long),
        "img_path": ["a", "b"][:B],
    }


# ---------------------------------------------------------------------------
# Module smoke
# ---------------------------------------------------------------------------


def test_set_prediction_module_training_step_runs():
    estimator = _build_estimator()
    module = SetPredictionTrainingModule(
        pose_estimator=estimator,
        weights=SetPredictionLossWeights(),
        lr=1e-3,
    )
    batch = _fake_set_batch(B=2)
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


def test_set_prediction_module_validation_step_runs():
    estimator = _build_estimator()
    module = SetPredictionTrainingModule(
        pose_estimator=estimator, lr=1e-3,
    )
    module.validation_step(_fake_set_batch(B=2), batch_idx=0)


def test_set_prediction_module_optimizer_returns_dict():
    estimator = _build_estimator()
    module = SetPredictionTrainingModule(
        pose_estimator=estimator, lr=1e-3, lr_T_max=10,
    )
    opt_dict = module.configure_optimizers()
    assert "optimizer" in opt_dict
    assert "lr_scheduler" in opt_dict


# ---------------------------------------------------------------------------
# Hydra root composition
# ---------------------------------------------------------------------------


def test_set_prediction_hydra_compose():
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "pose_estimator=cnn_set_prediction",
                "training_module=set_prediction",
                "data=crowdpose_set_prediction",
            ],
        )
    assert cfg.training_module._target_.endswith("SetPredictionTrainingModuleConfig")
    assert cfg.data._target_.endswith("CrowdPoseDataConfig")
    assert cfg.data.target_format == "set"

    tm_cfg = TrainingModuleConfig.from_hydra_config(cfg.training_module)
    assert isinstance(tm_cfg, SetPredictionTrainingModuleConfig)
    data_cfg = DataConfig.from_hydra_config(cfg.data)
    assert isinstance(data_cfg, CrowdPoseDataConfig)
    assert data_cfg.target_format == "set"
    assert data_cfg.image_h == 256


def test_set_prediction_config_build_with_real_data_module():
    """End-to-end: build pose estimator -> training module via configs."""
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "pose_estimator=cnn_set_prediction",
                "training_module=set_prediction",
                "data=crowdpose_set_prediction",
            ],
        )
    pose_cfg = PoseEstimatorConfig.from_hydra_config(cfg.pose_estimator)
    tm_cfg = TrainingModuleConfig.from_hydra_config(cfg.training_module)
    data_cfg = DataConfig.from_hydra_config(cfg.data)
    estimator = pose_cfg.build()
    data_module = data_cfg.build()
    module = tm_cfg.build(estimator, data_module)
    assert isinstance(module, SetPredictionTrainingModule)


# ---------------------------------------------------------------------------
# AP wiring smoke (do not need a real ann file for this — guarded by None)
# ---------------------------------------------------------------------------


def test_test_step_no_ap_metric_runs():
    estimator = _build_estimator()
    module = SetPredictionTrainingModule(
        pose_estimator=estimator, lr=1e-3,
        val_ann_file=None, test_ann_file=None,
    )
    module.test_step(_fake_set_batch(B=2), batch_idx=0)
    module.on_test_epoch_end()  # no metric -> no-op


def test_matched_oks_diagnostic():
    estimator = _build_estimator()
    module = SetPredictionTrainingModule(pose_estimator=estimator, lr=1e-3)
    batch = _fake_set_batch(B=2)
    pred = estimator(image=batch["image"])
    oks = module.matched_oks(pred, batch)
    assert torch.isfinite(oks)
    assert 0.0 <= oks.item() <= 1.0
