#!/usr/bin/env python3
"""
Benchmark visual backbones for multi-person 2D pose estimation on CrowdPose.

The probe architecture is fixed (ProbeHead + PoseSlotDecoder).
Only the backbone varies between runs.  See conf/backbone/ for variants.

Quick start:
    python scripts/benchmark_backbone.py

Override training hyperparameters:
    python scripts/benchmark_backbone.py training_module.lr=3e-4 \
        training_module.max_epochs=100 data.batch_size=8

Swap backbone:
    python scripts/benchmark_backbone.py backbone=cnn

Skip training and evaluate a checkpoint:
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

from src.utils import resolve_checkpoint

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.data import DataConfig
from src.features import BaseFeatureExtractorConfig
from src.training.training_config import BackboneProbeConfig


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

    backbone_cfg = BaseFeatureExtractorConfig.from_hydra_config(cfg.backbone)
    probe_cfg    = BackboneProbeConfig.from_hydra_config(cfg.training_module)
    data_cfg     = DataConfig.from_hydra_config(cfg.data)

    pl.seed_everything(probe_cfg.seed, workers=True)

    use_gpu    = cfg.hardware.accelerator in ("gpu", "cuda")
    if use_gpu:
        torch.backends.cudnn.benchmark = True
    backbone   = backbone_cfg.build()
    data_module = data_cfg.build(pin_memory=use_gpu)
    module     = probe_cfg.build(backbone, data=data_module)

    if use_gpu and cfg.get("compile", False):
        module = torch.compile(module)
        log.info("torch.compile enabled")
    elif use_gpu:
        log.info("torch.compile disabled (set compile=true to enable)")

    n_backbone = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_head     = sum(p.numel() for p in module.probe_head.parameters() if p.requires_grad)
    n_decoder  = sum(p.numel() for p in module.slot_decoder.parameters() if p.requires_grad)
    log.info("Backbone      : %s  (%s trainable params)", type(backbone).__name__, f"{n_backbone:,}")
    log.info("ProbeHead     : %s trainable params", f"{n_head:,}")
    log.info("SlotDecoder   : %s trainable params", f"{n_decoder:,}")
    log.info("DataModule    : %s  (batch=%d)", type(data_module).__name__, data_cfg.batch_size)
    log.info("val_ap_every  : %d epochs", probe_cfg.val_ap_every_n_epochs)

    skip_training = bool(cfg.get("skip_training", False))
    resume = resolve_checkpoint(probe_cfg.resume) if probe_cfg.resume else None
    if resume and resume != probe_cfg.resume:
        log.info("Checkpoint resolved: %s  ->  %s", probe_cfg.resume, resume)

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

    grad_clip = probe_cfg.grad_clip if probe_cfg.grad_clip > 0 else None
    trainer = pl.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=_devices_arg(cfg.hardware.devices),
        precision=cfg.hardware.precision,
        max_epochs=probe_cfg.max_epochs,
        gradient_clip_val=grad_clip,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=probe_cfg.log_every_n_steps,
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
        module.plot_dir = str(out_dir / "plots")
        output = trainer.test(module, datamodule=data_module, ckpt_path=resume)
        ckpt_used = resume
    else:
        log.info(
            "Training: epochs=%d  batch=%d  lr=%g",
            probe_cfg.max_epochs, data_cfg.batch_size, probe_cfg.lr,
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
        module.plot_dir = str(out_dir / "plots")
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

