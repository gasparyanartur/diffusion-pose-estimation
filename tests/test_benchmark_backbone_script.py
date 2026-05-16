"""Smoke tests for scripts/benchmark_backbone.py.

We don't run a full training loop here (no real data + heavy compute);
instead we verify:
  * the script module imports without side-effects
  * the Hydra root config ``conf/benchmark_backbone.yaml`` composes
  * the composed config builds typed configs and a real LightningModule
  * the +skip_training flag is exposed as a top-level field
  * _resolve_checkpoint expands ... wildcards correctly
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.data import DataConfig
from src.features import BaseFeatureExtractorConfig
from src.base_config import BaseConfig
from src.training import BackboneProbeConfig, BackboneProbeModule, TrainingModuleConfig

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
    # conf/benchmark_backbone.yaml wires backbone: cnn + training_module: backbone_probe.
    assert cfg.backbone._target_.endswith("ResNetFeatureExtractorConfig")
    assert cfg.training_module._target_.endswith("BackboneProbeConfig")
    assert cfg.data._target_.endswith("CrowdPoseDataConfig")
    assert cfg.data.target_format == "set"
    assert cfg.skip_training is False


def test_benchmark_config_skip_training_override():
    cfg = _compose(overrides=["skip_training=true"])
    assert cfg.skip_training is True


def test_benchmark_config_builds_module():
    cfg = _compose()
    backbone_cfg = BaseFeatureExtractorConfig.from_hydra_config(cfg.backbone)
    tm_cfg = BaseConfig.from_hydra_config(cfg.training_module)
    data_cfg = DataConfig.from_hydra_config(cfg.data)

    backbone = backbone_cfg.build()
    data_module = data_cfg.build(pin_memory=False)
    module = tm_cfg.build(backbone, data=data_module)

    assert isinstance(module, BackboneProbeModule)
    assert module.num_keypoints == 14


# ---------------------------------------------------------------------------
# resolve_checkpoint tests
# ---------------------------------------------------------------------------

from src.utils import resolve_checkpoint


def test_resolve_checkpoint_no_wildcard(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("x")
    assert resolve_checkpoint(str(ckpt)) == str(ckpt)


def test_resolve_checkpoint_wildcard_finds_most_recent(tmp_path):
    import time

    for ts in ["run_a", "run_b"]:
        ckpt = tmp_path / ts / "checkpoints" / "last.ckpt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_text(ts)
        time.sleep(0.02)  # ensure distinct mtime

    pattern = str(tmp_path / "..." / "checkpoints" / "last.ckpt")
    result = resolve_checkpoint(pattern)
    assert "run_b" in result  # most-recently modified wins


def test_resolve_checkpoint_wildcard_raises_when_no_match(tmp_path):
    pattern = str(tmp_path / "..." / "nonexistent.ckpt")
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(pattern)
