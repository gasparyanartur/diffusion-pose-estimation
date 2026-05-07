"""Tests for the new training/data config layer.

Validates:
  * polymorphic dispatch via from_hydra_config when root has _target_
  * DiffusionTrainingModuleConfig.build returns a PoseEstimatorTrainingModule
    with matching attributes
  * CrowdPoseDataConfig.build returns a CrowdPoseDataModule with the expected
    contract attributes
  * Hydra root config composes successfully for both diffusion and
    diffusion_minimal training presets
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data import (
    CrowdPoseDataConfig,
    CrowdPoseDataModule,
    DataConfig,
    PoseDataModule,
)
from src.decoders import GraphDiffusionDecoderConfig, MlpDenoiseNetworkConfig
from src.embedders import NoneEmbedderConfig
from src.features import NoneFeatureExtractorConfig
from src.pose_estimator import PoseEstimatorConfig
from src.training import (
    AugmentationConfig,
    DiffusionTrainingModuleConfig,
    LossConfig,
    PoseEstimatorTrainingModule,
    TrainingModuleConfig,
)


CONF_DIR = str(_ROOT / "conf")
FIXTURE_ANN = _ROOT / "tests" / "fixtures" / "mini_crowdpose_val.json"


def _tiny_pose_estimator():
    return PoseEstimatorConfig(
        feature_extractor=NoneFeatureExtractorConfig(),
        feature_embedder=NoneEmbedderConfig(),
        decoder=GraphDiffusionDecoderConfig(
            denoise_network=MlpDenoiseNetworkConfig(num_layers=1, hidden_dim=16),
            num_train_steps=10,
            hidden_dim=16,
            dim_pos=16,
            dim_time=16,
        ),
    ).build()


# ---------------------------------------------------------------------------
# Hydra composition (root + config groups)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("training_module_group", ["diffusion", "diffusion_minimal"])
def test_hydra_root_composes(training_module_group):
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(
            config_name="config",
            overrides=[f"training_module={training_module_group}"],
        )
    # _target_ keys present at root of training_module + data
    assert cfg.training_module._target_.endswith("DiffusionTrainingModuleConfig")
    assert cfg.data._target_.endswith("CrowdPoseDataConfig")
    # Sub-configs flat (no _target_), sanity-check a couple
    assert "augmentations" in cfg.training_module
    assert "losses" in cfg.training_module
    assert "batch_size" in cfg.data


def test_diffusion_minimal_overrides_apply():
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(
            config_name="config",
            overrides=["training_module=diffusion_minimal"],
        )
    assert cfg.training_module.max_epochs == 2
    assert cfg.training_module.ap_eval_interval == 0
    assert cfg.training_module.augmentations.aug_jitter is False
    assert cfg.training_module.losses.use_bone_loss is False


# ---------------------------------------------------------------------------
# TrainingModuleConfig polymorphic dispatch
# ---------------------------------------------------------------------------


def test_training_module_config_from_hydra_dispatches_subclass():
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(config_name="config")
    tm_cfg = TrainingModuleConfig.from_hydra_config(cfg.training_module)
    assert isinstance(tm_cfg, DiffusionTrainingModuleConfig)
    assert isinstance(tm_cfg.augmentations, AugmentationConfig)
    assert isinstance(tm_cfg.losses, LossConfig)


def test_diffusion_training_module_build_propagates_attrs():
    pose_estimator = _tiny_pose_estimator()
    tm_cfg = DiffusionTrainingModuleConfig(
        lr=3e-4, weight_decay=2e-4, max_epochs=7, seed=99,
    )
    data = CrowdPoseDataConfig(
        root=str(FIXTURE_ANN.parent),
        train_ann=FIXTURE_ANN.name,
        val_ann=FIXTURE_ANN.name,
        test_ann=FIXTURE_ANN.name,
        img_dir=".",
        batch_size=4,
        num_workers=0,
        ap_batch_size=2,
    ).build()
    module = tm_cfg.build(pose_estimator, data=data)
    assert isinstance(module, PoseEstimatorTrainingModule)
    assert module.lr == 3e-4
    assert module.weight_decay == 2e-4
    # max_epochs flows through cosine_lr=True -> lr_T_max
    assert module.lr_T_max == 7
    # AP-batch size pulled from data
    assert module._ap_batch_size == 2


# ---------------------------------------------------------------------------
# DataConfig polymorphic dispatch
# ---------------------------------------------------------------------------


def test_data_config_from_hydra_dispatches_subclass():
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(config_name="config")
    data_cfg = DataConfig.from_hydra_config(cfg.data)
    assert isinstance(data_cfg, CrowdPoseDataConfig)
    assert data_cfg.batch_size == 64


def test_crowdpose_data_module_contract():
    cfg = CrowdPoseDataConfig(
        root=str(FIXTURE_ANN.parent),
        train_ann=FIXTURE_ANN.name,
        val_ann=FIXTURE_ANN.name,
        test_ann=FIXTURE_ANN.name,
        img_dir=".",
        val_pred_cache=None,
        test_pred_cache=None,
        batch_size=2,
        num_workers=0,
        ap_batch_size=2,
    )
    dm = cfg.build()
    assert isinstance(dm, CrowdPoseDataModule)
    assert isinstance(dm, PoseDataModule)
    assert dm.batch_size == 2
    assert dm.ap_batch_size == 2
    assert dm.num_keypoints == 14
    assert dm.val_pred_cache_resolved is None
    assert dm.test_pred_cache_resolved is None
    # Setup + at least one batch
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert "keypoints" in batch
    assert "visibility" in batch


def test_missing_pred_cache_resolves_to_none(tmp_path):
    cfg = CrowdPoseDataConfig(
        root=str(FIXTURE_ANN.parent),
        train_ann=FIXTURE_ANN.name,
        val_ann=FIXTURE_ANN.name,
        test_ann=FIXTURE_ANN.name,
        img_dir=".",
        val_pred_cache=str(tmp_path / "does_not_exist.pkl"),
        test_pred_cache=None,
        batch_size=2,
        num_workers=0,
        ap_batch_size=2,
    )
    dm = cfg.build()
    assert dm.val_pred_cache_resolved is None
    assert dm.test_pred_cache_resolved is None


def test_subset_frac_slices_dataset():
    cfg = CrowdPoseDataConfig(
        root=str(FIXTURE_ANN.parent),
        train_ann=FIXTURE_ANN.name,
        val_ann=FIXTURE_ANN.name,
        test_ann=FIXTURE_ANN.name,
        img_dir=".",
        batch_size=1,
        num_workers=0,
        ap_batch_size=1,
        subset_frac=0.5,
    )
    full = CrowdPoseDataConfig(
        root=str(FIXTURE_ANN.parent),
        train_ann=FIXTURE_ANN.name,
        val_ann=FIXTURE_ANN.name,
        test_ann=FIXTURE_ANN.name,
        img_dir=".",
        batch_size=1,
        num_workers=0,
        ap_batch_size=1,
    ).build()
    full.setup()
    full_n = len(full._train_ds.samples)

    dm = cfg.build()
    dm.setup()
    expected = max(1, round(full_n * 0.5))
    assert len(dm._train_ds.samples) == expected
    assert len(dm._val_ds.samples) == expected
    assert len(dm._test_ds.samples) == expected


def test_subset_frac_minimum_keeps_one_sample():
    cfg = CrowdPoseDataConfig(
        root=str(FIXTURE_ANN.parent),
        train_ann=FIXTURE_ANN.name,
        val_ann=FIXTURE_ANN.name,
        test_ann=FIXTURE_ANN.name,
        img_dir=".",
        batch_size=1,
        num_workers=0,
        ap_batch_size=1,
        subset_frac=0.0001,
    )
    dm = cfg.build()
    dm.setup()
    assert len(dm._train_ds.samples) == 1


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0])
def test_subset_frac_validation_rejects_invalid(bad):
    with pytest.raises((ValueError, Exception)):
        CrowdPoseDataConfig(
            root=str(FIXTURE_ANN.parent),
            train_ann=FIXTURE_ANN.name,
            val_ann=FIXTURE_ANN.name,
            test_ann=FIXTURE_ANN.name,
            img_dir=".",
            subset_frac=bad,
        )
