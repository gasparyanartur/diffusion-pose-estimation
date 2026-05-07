"""Smoke tests for scripts/benchmark_backbone.py.

We don't run a full training loop here (no real data + heavy compute);
instead we verify:
  * the script module imports without side-effects
  * the Hydra root config ``conf/benchmark_backbone.yaml`` composes
  * the composed config builds typed configs and a real LightningModule
  * the +skip_training flag is exposed as a top-level field
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.data import DataConfig
from src.pose_estimator import PoseEstimatorConfig
from src.training import SetPredictionTrainingModule, TrainingModuleConfig

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "conf"


def _compose(overrides: list[str] | None = None):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        return compose(config_name="benchmark_backbone", overrides=overrides or [])


def test_script_imports_cleanly():
    # Importing must not run main() or touch CUDA.
    spec = importlib.util.spec_from_file_location(
        "benchmark_backbone_script",
        ROOT / "scripts" / "benchmark_backbone.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
    sys.modules.pop("benchmark_backbone_script", None)


def test_benchmark_config_composes_with_set_prediction_defaults():
    cfg = _compose()
    # Defaults wired by conf/benchmark_backbone.yaml. The pose_estimator
    # group has no top-level _target_; its three stages each carry one.
    assert cfg.pose_estimator.feature_extractor._target_.endswith(
        "CNNFeatureExtractorConfig"
    )
    assert cfg.pose_estimator.feature_embedder._target_.endswith(
        "SpatialAttentionEmbedderConfig"
    )
    assert cfg.pose_estimator.decoder._target_.endswith("PoseSlotDecoderConfig")
    assert cfg.training_module._target_.endswith("SetPredictionTrainingModuleConfig")
    assert cfg.data._target_.endswith("CrowdPoseDataConfig")
    assert cfg.data.target_format == "set"
    assert cfg.skip_training is False


def test_benchmark_config_skip_training_override():
    cfg = _compose(overrides=["skip_training=true"])
    assert cfg.skip_training is True


def test_benchmark_config_builds_module():
    cfg = _compose()
    pose_cfg = PoseEstimatorConfig.from_hydra_config(cfg.pose_estimator)
    tm_cfg = TrainingModuleConfig.from_hydra_config(cfg.training_module)
    data_cfg = DataConfig.from_hydra_config(cfg.data)

    pose_estimator = pose_cfg.build()
    data_module = data_cfg.build(pin_memory=False)
    module = tm_cfg.build(pose_estimator, data=data_module)

    assert isinstance(module, SetPredictionTrainingModule)
    assert module.num_keypoints == 14
