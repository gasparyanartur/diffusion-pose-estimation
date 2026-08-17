#!/usr/bin/env python3
"""
Benchmark visual backbones for multi-person 2D pose estimation on CrowdPose.

The probe architecture is fixed (ProbeHead + fixed-slot prediction head).
Only the backbone varies between runs.  See conf/backbone/ for variants.

Quick start:
    python scripts/investigation/benchmark_backbone.py

Override training hyperparameters:
    python scripts/investigation/benchmark_backbone.py probe.lr=3e-4 \
        training.max_epochs=100 data.batch_size=8

Swap backbone:
    python scripts/investigation/benchmark_backbone.py backbone=cnn

Skip training and evaluate a checkpoint:
    python scripts/investigation/benchmark_backbone.py skip_training=true \
        training.resume=results/backbone_benchmark/.../checkpoints/last.ckpt
"""

from __future__ import annotations

import os

# Must be set before any CUDA context is created.
# expandable_segments conflicts with mixed large/small allocations and triggers
# an internal allocator assert (!block->expandable_segment_) in PyTorch ≥ 2.1.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.data import DataConfig
from src.models.features.base import BaseFeatureExtractorConfig
from src.configs.base_config import BaseConfig
from src.training import TrainingConfig, TrainingRunner, TrainingSpec
from src.utils import resolve_checkpoint


@hydra.main(version_base="1.3", config_path="../../conf", config_name="benchmark_backbone")
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    training_cfg = TrainingConfig.from_hydra_config(cfg.training)
    probe_cfg = BaseConfig.from_hydra_config(cfg.probe)
    data_cfg = DataConfig.from_hydra_config(cfg.data)
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

    backbone_cfg = BaseFeatureExtractorConfig.from_hydra_config(cfg.backbone)
    runner.seed_everything()

    use_gpu    = cfg.hardware.accelerator in ("gpu", "cuda")
    if use_gpu:
        torch.backends.cudnn.benchmark = True
    backbone   = backbone_cfg.build()
    data_module = data_cfg.build(pin_memory=use_gpu)
    module     = probe_cfg.build(
        backbone, data=data_module, max_epochs=training_cfg.max_epochs
    )
    module.plot_dir = str(out_dir / "plots")

    if use_gpu and cfg.get("compile", False):
        module = torch.compile(module)
        log.info("torch.compile enabled")
    elif use_gpu:
        log.info("torch.compile disabled (set compile=true to enable)")

    n_backbone = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_head     = sum(p.numel() for p in module.probe_head.parameters() if p.requires_grad)
    n_total    = sum(p.numel() for p in module.parameters() if p.requires_grad)
    log.info("Backbone      : %s  (%s trainable params)", type(backbone).__name__, f"{n_backbone:,}")
    log.info("ProbeHead     : %s trainable params", f"{n_head:,}")
    log.info("Total         : %s trainable params", f"{n_total:,}")
    log.info("DataModule    : %s  (batch=%d)", type(data_module).__name__, data_cfg.batch_size)
    log.info("val_ap_every  : %d epochs", probe_cfg.val_ap_every_n_epochs)

    log.info(
        "Training: epochs=%d  batch=%d  lr=%g",
        training_cfg.max_epochs, data_cfg.batch_size, probe_cfg.lr,
    )
    result = runner.run(
        module,
        data_module,
        max_epochs=training_cfg.max_epochs,
        spec=TrainingSpec(result_filename="test_results.json"),
    )
    log.info("Best checkpoint: %s", result.checkpoint)
    if result.result_path:
        log.info("Test results saved to %s", result.result_path)


if __name__ == "__main__":
    main()
