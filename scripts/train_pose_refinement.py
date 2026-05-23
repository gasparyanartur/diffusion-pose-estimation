#!/usr/bin/env python3
"""
Train the pose-refinement diffusion model on CrowdPose.

Config is managed by Hydra (conf/config.yaml + config groups under conf/).
Run from the project root so that relative data paths resolve correctly.

Quick start (all defaults):
    python scripts/train_pose_refinement.py

Fast smoke run (2 epochs, batch 4, no augmentations, no AP eval):
    python scripts/train_pose_refinement.py training_module=diffusion_minimal data=crowdpose_minimal

Switch config groups:
    python scripts/train_pose_refinement.py pose_estimator=default
    python scripts/train_pose_refinement.py pose_estimator=dino_bilinear
    python scripts/train_pose_refinement.py hardware=gpu

Override individual values:
    python scripts/train_pose_refinement.py run_name=exp1 training_module.lr=3e-4 training_module.max_epochs=200
    python scripts/train_pose_refinement.py pose_estimator.decoder.num_train_steps=200
    python scripts/train_pose_refinement.py pose_estimator.decoder.denoise_network.num_layers=6
    python scripts/train_pose_refinement.py training_module.augmentations.aug_jitter=false training_module.losses.use_bone_loss=false
    python scripts/train_pose_refinement.py data.batch_size=8 data.num_workers=0

Resume from a checkpoint:
    python scripts/train_pose_refinement.py training_module.resume=results/pose_estimator_training/run/ts/checkpoints/last.ckpt
"""

from __future__ import annotations

# Must be set before any CUDA context is created (i.e. before torch.cuda.init).
# expandable_segments conflicts with mixed large/small allocations (e.g.
# DINO backbone + CNN patch crops) and triggers an internal allocator assert
# (!block->expandable_segment_) in PyTorch ≥ 2.1.
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

import json
import logging
import sys
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.data import DataConfig
from src.pose_estimator import PoseEstimatorConfig
from src.training import TrainingModuleConfig


# ---------------------------------------------------------------------------
# Custom callbacks
# ---------------------------------------------------------------------------


class RunningLossLogger(pl.Callback):
    """Log a rolling mean of val/loss over the last ``window`` epochs.

    The averaged value is logged as ``running_loss_{window}`` and is suitable
    for use as a ModelCheckpoint monitor so that one-off noisy validation
    epochs do not trigger spurious best-checkpoint saves.
    """

    def __init__(self, window: int = 5, source: str = "val/loss") -> None:
        super().__init__()
        self.window = window
        self.source = source
        self._history: list[float] = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        val = trainer.callback_metrics.get(self.source)
        if val is None:
            return
        self._history.append(float(val))
        if len(self._history) > self.window:
            self._history.pop(0)
        smooth = sum(self._history) / len(self._history)
        pl_module.log(f"running_loss_{self.window}", smooth, prog_bar=False)


class GradNormLogger(pl.Callback):
    """Log the global gradient L2 norm before each optimizer step."""

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer) -> None:
        total_sq: float = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_sq += p.grad.detach().norm(2).item() ** 2
        pl_module.log("train/grad_norm", total_sq ** 0.5, on_step=True, on_epoch=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _devices_arg(raw) -> int | str | list[int]:
    """Convert the hardware.devices config value to the type Lightning expects."""
    if raw == "auto":
        return "auto"
    s = str(raw)
    if "," in s:
        return [int(d) for d in s.split(",")]
    return int(s)


def _setup_logging(log_file: Path) -> logging.Logger:
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logging(out_dir / "train.log")
    log.info("Run name  : %s", cfg.run_name)
    log.info("Output dir: %s", out_dir)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Build typed configs (polymorphic dispatch on _target_).
    pose_cfg = PoseEstimatorConfig.from_hydra_config(cfg.pose_estimator)
    tm_cfg = TrainingModuleConfig.from_hydra_config(cfg.training_module)
    data_cfg = DataConfig.from_hydra_config(cfg.data)

    pl.seed_everything(tm_cfg.seed, workers=True)

    use_gpu = cfg.hardware.accelerator in ("gpu", "cuda")
    pose_estimator = pose_cfg.build()
    data_module = data_cfg.build(pin_memory=use_gpu)
    module = tm_cfg.build(pose_estimator, data=data_module)

    log.info("FeatureExtractor : %s", type(pose_estimator.feature_extractor).__name__)
    log.info("FeatureEmbedder  : %s", type(pose_estimator.feature_embedder).__name__)
    log.info("Decoder          : %s", type(pose_estimator.decoder).__name__)
    n_params = sum(p.numel() for p in pose_estimator.parameters() if p.requires_grad)
    log.info("PoseEstimator    —  %s trainable parameters", f"{n_params:,}")
    log.info("DataModule       : %s  (batch=%d, num_workers=%d)",
             type(data_module).__name__, data_cfg.batch_size, data_cfg.num_workers)
    log.info("TrainingModule   : %s", type(module).__name__)

    monitor_key = f"running_loss_{tm_cfg.running_loss_window}"
    callbacks: list[pl.Callback] = [
        GradNormLogger(),
        RunningLossLogger(window=tm_cfg.running_loss_window),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"epoch{{epoch:03d}}-{monitor_key}={{{monitor_key}:.4f}}",
            monitor=monitor_key,
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if tm_cfg.early_stop_patience > 0:
        callbacks.append(EarlyStopping(
            monitor="val/loss",
            patience=tm_cfg.early_stop_patience,
            mode="min",
            verbose=True,
        ))

    loggers = [
        TensorBoardLogger(save_dir=str(out_dir), name="tb_logs", version=""),
        CSVLogger(save_dir=str(out_dir),          name="csv_logs", version=""),
    ]

    grad_clip = tm_cfg.grad_clip if tm_cfg.grad_clip > 0 else None
    trainer = pl.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=_devices_arg(cfg.hardware.devices),
        precision=cfg.hardware.precision,
        max_epochs=tm_cfg.max_epochs,
        gradient_clip_val=grad_clip,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=tm_cfg.log_every_n_steps,
        deterministic=False,
    )

    log.info(
        "Training: epochs=%d  batch=%d  lr=%g  cosine_lr=%s",
        tm_cfg.max_epochs, data_cfg.batch_size, tm_cfg.lr, tm_cfg.cosine_lr,
    )
    resume = tm_cfg.resume
    if resume:
        log.info("Resuming from: %s", resume)

    if tm_cfg.compile:
        log.info("torch.compile: compiling decoder.predict_delta (dynamic=True)...")
        decoder = module.pose_estimator.decoder
        decoder.predict_delta = torch.compile(decoder.predict_delta, dynamic=True)

    trainer.fit(module, datamodule=data_module, ckpt_path=resume)

    best_ckpt  = trainer.checkpoint_callback.best_model_path   # type: ignore[union-attr]
    best_score = trainer.checkpoint_callback.best_model_score  # type: ignore[union-attr]
    log.info("Best val/loss  : %.6f", best_score)
    log.info("Best checkpoint: %s", best_ckpt)

    if best_ckpt:
        log.info("Evaluating best checkpoint on test set...")
        output = trainer.test(module, datamodule=data_module, ckpt_path=best_ckpt)
        for metric, value in output[0].items():
            log.info("test  %-30s %.6f", metric, value)
        dump_path = out_dir / "test_results.json"
        with open(dump_path, "w") as fh:
            json.dump(output[0], fh, indent=4)
        log.info("Test results saved to %s", dump_path)
    else:
        log.warning("No checkpoint found, skipping test evaluation.")


if __name__ == "__main__":
    main()
