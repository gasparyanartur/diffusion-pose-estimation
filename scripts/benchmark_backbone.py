#!/usr/bin/env python3
"""
Benchmark the set-prediction backbone (CNN + spatial-attention embedder +
pose-slot decoder) on CrowdPose.

This is the productionised version of ``notebook/backbone_benchmark.ipynb``.
The full pipeline lives in ``src/`` (extractor / embedder / decoder, set-
prediction loss, training module, AP wiring); this script is just a thin
Hydra entry-point.

Quick start:
    python scripts/benchmark_backbone.py

Override any field:
    python scripts/benchmark_backbone.py training_module.lr=3e-4 \
        training_module.max_epochs=100 data.batch_size=8

Skip training and just evaluate a checkpoint:
    python scripts/benchmark_backbone.py skip_training=true \
        training_module.resume=results/backbone_benchmark/.../checkpoints/last.ckpt
"""

from __future__ import annotations

import os
# Must be set before any CUDA context is created.
# expandable_segments conflicts with mixed large/small allocations and triggers
# an internal allocator assert (!block->expandable_segment_) in PyTorch ≥ 2.1.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

import json
import logging
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.data import DataConfig
from src.pose_estimator import PoseEstimatorConfig
from src.training import TrainingModuleConfig


def _devices_arg(raw) -> int | str | list[int]:
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
    logger = logging.getLogger("benchmark_backbone")
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


@hydra.main(version_base="1.3", config_path="../conf", config_name="benchmark_backbone")
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logging(out_dir / "benchmark.log")
    log.info("Run name  : %s", cfg.run_name)
    log.info("Output dir: %s", out_dir)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

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
    log.info("PoseEstimator    : %s trainable parameters", f"{n_params:,}")
    log.info("DataModule       : %s  (batch=%d, target_format=%s)",
             type(data_module).__name__, data_cfg.batch_size,
             getattr(data_cfg, "target_format", "?"))
    log.info("TrainingModule   : %s", type(module).__name__)

    skip_training = bool(cfg.get("skip_training", False))
    resume = tm_cfg.resume

    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch{epoch:03d}-val_loss{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    loggers = [
        TensorBoardLogger(save_dir=str(out_dir), name="tb_logs", version=""),
        CSVLogger(save_dir=str(out_dir),         name="csv_logs", version=""),
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

    if skip_training:
        if not resume:
            log.error(
                "skip_training=true requires training_module.resume to point "
                "at a checkpoint."
            )
            sys.exit(1)
        log.info("Skipping training; loading checkpoint: %s", resume)
        output = trainer.test(module, datamodule=data_module, ckpt_path=resume)
        ckpt_used = resume
    else:
        log.info(
            "Training: epochs=%d  batch=%d  lr=%g",
            tm_cfg.max_epochs, data_cfg.batch_size, tm_cfg.lr,
        )
        if resume:
            log.info("Resuming from: %s", resume)

        trainer.fit(module, datamodule=data_module, ckpt_path=resume)

        best_ckpt  = trainer.checkpoint_callback.best_model_path   # type: ignore[union-attr]
        best_score = trainer.checkpoint_callback.best_model_score  # type: ignore[union-attr]
        log.info("Best val/loss  : %s", best_score)
        log.info("Best checkpoint: %s", best_ckpt)
        if not best_ckpt:
            log.warning("No checkpoint found; skipping test evaluation.")
            return
        log.info("Evaluating best checkpoint on test set...")
        output = trainer.test(module, datamodule=data_module, ckpt_path=best_ckpt)
        ckpt_used = best_ckpt

    if output:
        for metric, value in output[0].items():
            log.info("test  %-30s %s", metric, value)
        dump_path = out_dir / "test_results.json"
        with open(dump_path, "w") as fh:
            json.dump({"checkpoint": ckpt_used, "metrics": output[0]}, fh, indent=4)
        log.info("Test results saved to %s", dump_path)


if __name__ == "__main__":
    main()
