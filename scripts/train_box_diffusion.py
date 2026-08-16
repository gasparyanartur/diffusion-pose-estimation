#!/usr/bin/env python3
"""
Train the canonical tensor-native box-diffusion pipeline on CrowdPose.

Quick start:
    python scripts/train_box_diffusion.py

Smoke run on a tiny subset:
    python scripts/train_box_diffusion.py data=crowdpose_set_prediction_debug \
        pipeline.max_epochs=2 pipeline.val_ap_every_n_epochs=1 \
        hardware.accelerator=cpu hardware.devices=1 hardware.precision=32

Skip training and evaluate a checkpoint:
    python scripts/train_box_diffusion.py skip_training=true \
        pipeline.resume=results/box_diffusion_benchmark/.../checkpoints/last.ckpt
"""

from __future__ import annotations

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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from lightning_fabric.plugins.io import TorchCheckpointIO

from src.configs.base_config import BaseConfig
from src.data import DataConfig
from src.models.pipelines.object_detectors.box_diffusion import BoxDiffusionPipelineConfig
from src.utils import resolve_checkpoint


class _TrustedCheckpointIO(TorchCheckpointIO):
    """Load checkpoints with weights_only=False (trusted local files)."""

    def load_checkpoint(self, path, map_location=None, weights_only=None):
        return super().load_checkpoint(path, map_location=map_location, weights_only=False)


class _CheckpointAlsoOnLastEpoch(ModelCheckpoint):
    """ModelCheckpoint that also evaluates and saves on the final training epoch.

    ``every_n_epochs`` normally prevents the checkpoint from running if the
    current epoch is not on a period boundary.  This subclass forces evaluation
    on the very last epoch so that the best-checkpoint is updated even when
    ``max_epochs`` is not divisible by ``every_n_epochs``.
    """

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        is_last = (trainer.current_epoch + 1) == trainer.max_epochs
        on_period = (
            self._every_n_epochs >= 1
            and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        )
        if is_last and not on_period:
            # Force evaluation on the last epoch even if not on a period boundary.
            if (
                not self._should_skip_saving_checkpoint(trainer)
                and self._should_save_on_train_epoch_end(trainer)
            ):
                monitor_candidates = self._monitor_candidates(trainer)
                self._save_topk_checkpoint(trainer, monitor_candidates)
        else:
            super().on_train_epoch_end(trainer, pl_module)


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
    logger = logging.getLogger("train_box_diffusion")
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


@hydra.main(version_base="1.3", config_path="../conf", config_name="benchmark_box_diffusion")
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logging(out_dir / "train.log")
    log.info("Run name  : %s", cfg.run_name)
    log.info("Output dir: %s", out_dir)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    pipeline_cfg = BoxDiffusionPipelineConfig.from_hydra_config(cfg.pipeline)
    data_cfg = DataConfig.from_hydra_config(cfg.data)

    pl.seed_everything(pipeline_cfg.seed, workers=True)

    use_gpu = cfg.hardware.accelerator in ("gpu", "cuda")
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    data_module = data_cfg.build(pin_memory=use_gpu)
    module = pipeline_cfg.build(
        val_ann_file=str(data_module.val_ann_path),
        test_ann_file=str(data_module.test_ann_path),
    )

    if use_gpu and cfg.get("compile", False):
        module = torch.compile(module)
        log.info("torch.compile enabled")

    n_backbone = sum(p.numel() for p in module.backbone.parameters() if p.requires_grad)
    n_denoiser = sum(p.numel() for p in module.denoiser.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in module.parameters() if p.requires_grad)
    log.info("Backbone : %s  (%s trainable params)", type(module.backbone).__name__, f"{n_backbone:,}")
    log.info("Denoiser : %s  (%s trainable params)", type(module.denoiser).__name__, f"{n_denoiser:,}")
    log.info("Total    : %s trainable params", f"{n_total:,}")
    log.info("DataModule: %s  (batch=%d)", type(data_module).__name__, data_cfg.batch_size)
    log.info("val_ap_every: %d epochs", pipeline_cfg.val_ap_every_n_epochs)

    skip_training = bool(cfg.get("skip_training", False))
    resume = resolve_checkpoint(pipeline_cfg.resume) if pipeline_cfg.resume else None
    if resume and resume != pipeline_cfg.resume:
        log.info("Checkpoint resolved: %s  ->  %s", pipeline_cfg.resume, resume)

    n_steps = pipeline_cfg.num_inference_steps
    callbacks: list[pl.Callback] = [
        _CheckpointAlsoOnLastEpoch(
            dirpath=ckpt_dir,
            filename="epoch{epoch:03d}-val_bbox_AP",
            monitor=f"ap-val/{n_steps}-step/bbox/AP",
            mode="max",
            save_top_k=1,
            save_last=False,
            every_n_epochs=pipeline_cfg.val_ap_every_n_epochs,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    loggers = [
        TensorBoardLogger(save_dir=str(out_dir), name="tb_logs", version=""),
        CSVLogger(save_dir=str(out_dir), name="csv_logs", version=""),
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
        plugins=[_TrustedCheckpointIO()],
    )

    if skip_training:
        if not resume:
            log.error("skip_training=true requires pipeline.resume.")
            sys.exit(1)
        log.info("Skipping training; loading checkpoint: %s", resume)
        output = trainer.test(module, datamodule=data_module, ckpt_path=resume)
        ckpt_used = resume
    else:
        log.info(
            "Training: epochs=%d  batch=%d  lr=%g",
            pipeline_cfg.max_epochs, data_cfg.batch_size, pipeline_cfg.lr,
        )
        if resume:
            log.info("Resuming from: %s", resume)
        trainer.fit(module, datamodule=data_module, ckpt_path=resume)

        best_ckpt = trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
        best_score = trainer.checkpoint_callback.best_model_score  # type: ignore[union-attr]
        log.info("Best ap-val/%d-step/bbox/AP: %s", n_steps, best_score)
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
