"""Tests for SetPredictionTrainingModule + Hydra config wiring."""

from __future__ import annotations

import sys
import tempfile
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
    masks = [[1, 1, 0], [1, 0, 0], [1, 1, 0]]
    return {
        "image": torch.rand(B, 3, 64, 48),
        "gt_keypoints": (torch.rand(B, P_max, K, 2) * 2 - 1),
        "gt_labeled": torch.ones(B, P_max, K),
        "gt_visible": torch.ones(B, P_max, K),
        "person_scale": torch.full((B, P_max), 0.4),
        "person_mask": torch.tensor(masks[:B], dtype=torch.float32),
        "img_hw": torch.tensor([[100.0, 200.0]] * B),
        "image_id": torch.tensor(list(range(1, B + 1)), dtype=torch.long),
        "img_path": [chr(ord("a") + i) for i in range(B)],
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
    assert data_cfg.image_h is None


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


# ---------------------------------------------------------------------------
# GT sanity-check mode
# ---------------------------------------------------------------------------


def test_gt_sanity_check_pred_shape():
    """_make_gt_pred output shapes must match normal decoder output."""
    estimator = _build_estimator()
    module = SetPredictionTrainingModule(
        pose_estimator=estimator, lr=1e-3, gt_sanity_check=True,
    )
    batch = _fake_set_batch(B=2)
    pred = module._forward(batch)
    N = estimator.decoder.num_slots
    assert pred["keypoints"].shape == (2, N, K, 2)
    assert pred["scores"].shape == (2, N)
    assert pred["keypoint_scores"].shape == (2, N, K)


def test_gt_sanity_check_objectness_sign():
    """Real person slots get positive logits; empty slots get negative."""
    estimator = _build_estimator()
    module = SetPredictionTrainingModule(
        pose_estimator=estimator, lr=1e-3, gt_sanity_check=True,
    )
    # batch has person_mask [[1,1,0],[1,0,0]] — image 1 has only 1 person
    # and num_slots=2, so slot 1 of image 1 is empty.
    batch = _fake_set_batch(B=2)
    pred = module._forward(batch)
    obj = pred["scores_logits"]  # [B, N]
    # image 0: both slots occupied
    assert obj[0, 0].item() > 0
    assert obj[0, 1].item() > 0
    # image 1: slot 0 occupied, slot 1 empty
    assert obj[1, 0].item() > 0
    assert obj[1, 1].item() < 0  # empty slot must be negative


def test_gt_sanity_check_keypoints_equal_gt():
    """Keypoints in filled slots must equal the GT (up to float precision)."""
    estimator = _build_estimator()
    module = SetPredictionTrainingModule(
        pose_estimator=estimator, lr=1e-3, gt_sanity_check=True,
    )
    batch = _fake_set_batch(B=2)
    pred = module._forward(batch)
    # First P_max slots should match
    P_max = batch["gt_keypoints"].shape[1]
    n_fill = min(estimator.decoder.num_slots, P_max)
    assert torch.allclose(
        pred["keypoints"][:, :n_fill], batch["gt_keypoints"][:, :n_fill], atol=1e-5
    )


# ---------------------------------------------------------------------------
# Prediction plotting
# ---------------------------------------------------------------------------


def test_save_plots_creates_files():
    from src.vis import save_prediction_plot
    H, W, N, K = 64, 48, 5, 14
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.png"
        save_prediction_plot(
            out_path=out,
            image_chw=torch.rand(3, H, W),
            pred_kpts_pixel=torch.rand(N, K, 2) * W,
            pred_scores=torch.rand(N),
            gt_kpts_pixel=torch.rand(2, K, 2) * W,
            gt_labeled=torch.ones(2, K),
            top_k=3,
        )
        assert out.exists() and out.stat().st_size > 0


def test_test_step_saves_plots():
    estimator = _build_estimator()
    with tempfile.TemporaryDirectory() as tmpdir:
        module = SetPredictionTrainingModule(
            pose_estimator=estimator, lr=1e-3,
            plot_dir=tmpdir, max_plot_images=2,
        )
        module.on_test_start()
        batch = _fake_set_batch(B=3)  # 3 images, cap is 2
        module.test_step(batch, batch_idx=0)
        saved = list(Path(tmpdir).glob("*.png"))
        assert len(saved) == 2  # capped at max_plot_images


# ---------------------------------------------------------------------------
# max_persons data filter
# ---------------------------------------------------------------------------


def test_maybe_filter_persons_keeps_le_max():
    from src.data.datasets import PoseDataset
    from src.data.data_module import CrowdPoseDataModule
    # Build a tiny in-memory dataset with varying person counts
    samples = [
        {"image_id": i, "img_path": "", "keypoints": torch.zeros(n, K, 2), "visibility": torch.zeros(n, K)}
        for i, n in enumerate([1, 2, 3, 4, 2, 1])
    ]
    ds = PoseDataset(samples=samples, num_keypoints=K)
    dm = CrowdPoseDataModule.__new__(CrowdPoseDataModule)
    dm.max_persons = 2
    dm._maybe_filter_persons(ds, "test")
    assert all(s["keypoints"].shape[0] <= 2 for s in ds.samples)
    assert len(ds.samples) == 4  # 1,2,2,1 kept; 3,4 dropped
