#!/usr/bin/env python3
"""
Train the fixed-slot baseline pose pipeline on CrowdPose.

Config is managed by Hydra through one pipeline configuration.
Run from the project root so that relative data paths resolve correctly.

Quick start:
    python scripts/train_pose_refinement.py

Smoke run:
    python scripts/train_pose_refinement.py pipeline.max_epochs=2 data=crowdpose_set_prediction_debug

Override individual values:
    python scripts/train_pose_refinement.py pipeline.lr=3e-4 pipeline.max_epochs=100
"""

from __future__ import annotations

# Must be set before any CUDA context is created (i.e. before torch.cuda.init).
# expandable_segments conflicts with mixed large/small allocations (e.g.
# Large feature-extractor runs with CNN patch crops can trigger an internal
# allocator assert
# (!block->expandable_segment_) in PyTorch ≥ 2.1.
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

import json
import logging
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.data import DataConfig
from src.models.pipelines.pose_estimators.baseline import BaselinePosePipelineConfig


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

    pipeline_cfg = BaselinePosePipelineConfig.from_hydra_config(cfg.pipeline)
    data_cfg = DataConfig.from_hydra_config(cfg.data)

    pl.seed_everything(pipeline_cfg.seed, workers=True)

    use_gpu = cfg.hardware.accelerator in ("gpu", "cuda")
    data_module = data_cfg.build(pin_memory=use_gpu)
    module = pipeline_cfg.build(
        val_ann_file=str(data_module.val_ann_path),
        test_ann_file=str(data_module.test_ann_path),
        max_persons=getattr(data_module, "max_persons", None),
    )

    n_params = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    log.info("Pipeline         : %s", type(module).__name__)
    log.info("Backbone         : %s", type(module.backbone).__name__)
    log.info("Trainable params : %s", f"{n_params:,}")
    log.info("DataModule       : %s  (batch=%d, num_workers=%d)",
             type(data_module).__name__, data_cfg.batch_size, data_cfg.num_workers)

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
        CSVLogger(save_dir=str(out_dir),          name="csv_logs", version=""),
    ]

    grad_clip = pipeline_cfg.grad_clip if pipeline_cfg.grad_clip > 0 else None
    trainer = pl.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=_devices_arg(cfg.hardware.devices),
        precision=cfg.hardware.precision,
        max_epochs=pipeline_cfg.max_epochs,
        gradient_clip_val=grad_clip,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=pipeline_cfg.log_every_n_steps,
        deterministic=False,
    )

    log.info(
        "Training: epochs=%d  batch=%d  lr=%g  cosine_lr=%s",
        pipeline_cfg.max_epochs, data_cfg.batch_size, pipeline_cfg.lr, pipeline_cfg.cosine_lr,
    )
    trainer.fit(module, datamodule=data_module)

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
