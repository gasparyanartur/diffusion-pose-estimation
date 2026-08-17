#!/usr/bin/env python3
"""
Train the graph-diffusion pipeline on CrowdPose.

Architecture: backbone → `GraphPosePipeline`, which jointly denoises boxes and
keypoints with EDM diffusion, attention, and skeleton refinement.

Quick start:
    python scripts/training/train_graph_pipeline.py

Smoke run on a tiny subset (CPU):
    python scripts/training/train_graph_pipeline.py data=crowdpose_set_prediction_debug \
        training.max_epochs=2 pipeline.val_ap_every_n_epochs=1 \
        hardware.accelerator=cpu hardware.devices=1 hardware.precision=32

Skip training and evaluate a checkpoint:
    python scripts/training/train_graph_pipeline.py skip_training=true \
        training.resume=results/graph_pipeline/.../checkpoints/last.ckpt
"""

from __future__ import annotations

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import DataConfig
from src.models.pipelines.pose_estimators.graph import GraphPosePipelineConfig
from src.training import TrainingConfig, TrainingRunner, TrainingSpec
from src.training.callbacks import TrustedCheckpointIO
from src.utils import resolve_checkpoint


@hydra.main(version_base="1.3", config_path="../../conf", config_name="benchmark_graph_pipeline")
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    pipeline_cfg = GraphPosePipelineConfig.from_hydra_config(cfg.pipeline)
    training_cfg = TrainingConfig.from_hydra_config(cfg.training)
    requested_resume = training_cfg.resume
    resume = resolve_checkpoint(requested_resume) if requested_resume else None
    training_cfg = training_cfg.model_copy(update={
        "resume": resume,
        "skip_training": training_cfg.skip_training or bool(cfg.get("skip_training", False)),
    })
    runner = TrainingRunner(training_cfg, out_dir)
    log = runner.logger()
    log.info("Run name  : %s", cfg.run_name)
    log.info("Output dir: %s", out_dir)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    data_cfg     = DataConfig.from_hydra_config(cfg.data)
    runner.seed_everything()

    use_gpu = training_cfg.accelerator in ("gpu", "cuda")
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    data_module = data_cfg.build(pin_memory=use_gpu)
    module = pipeline_cfg.build(
        val_ann_file=str(data_module.val_ann_path),
        test_ann_file=str(data_module.test_ann_path),
        max_epochs=training_cfg.max_epochs,
    )

    if use_gpu and cfg.get("compile", False):
        module = torch.compile(module)
        log.info("torch.compile enabled")

    n_backbone = sum(p.numel() for p in module.backbone.parameters() if p.requires_grad)
    n_denoiser = sum(p.numel() for p in module.denoiser.parameters() if p.requires_grad)
    n_total    = sum(p.numel() for p in module.parameters() if p.requires_grad)
    log.info("Backbone : %s  (%s trainable params)", type(module.backbone).__name__, f"{n_backbone:,}")
    log.info("Denoiser : %s  (%s trainable params)", type(module.denoiser).__name__, f"{n_denoiser:,}")
    log.info("Total    : %s trainable params", f"{n_total:,}")
    log.info("DataModule: %s  (batch=%d)", type(data_module).__name__, data_cfg.batch_size)
    log.info("val_ap_every: %d epochs", pipeline_cfg.val_ap_every_n_epochs)

    if resume and resume != requested_resume:
        log.info("Checkpoint resolved: %s  ->  %s", requested_resume, resume)

    last_checkpoint = ModelCheckpoint(
        dirpath=runner.checkpoint_directory,
        filename="last",
        save_last=False,
        save_top_k=1,
        every_n_epochs=1,
        monitor=None,
        enable_version_counter=False,
    )
    result = runner.run(
        module,
        data_module,
        max_epochs=training_cfg.max_epochs,
        spec=TrainingSpec(
            callbacks=[last_checkpoint],
            plugins=[TrustedCheckpointIO()],
        ),
    )
    log.info("Best checkpoint: %s", result.checkpoint)
    if result.result_path:
        log.info("Test results saved to %s", result.result_path)


if __name__ == "__main__":
    main()
