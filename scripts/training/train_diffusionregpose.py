#!/usr/bin/env python3
"""Train the native DiffusionRegPose pipeline on CrowdPose."""

from __future__ import annotations

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.data import DataConfig
from src.models.pipelines.pose_estimators.diffusionregpose import DiffusionRegPosePipelineConfig
from src.training import TrainingConfig, TrainingRunner, TrainingSpec
from src.training.callbacks import TrustedCheckpointIO
from src.utils import resolve_checkpoint


@hydra.main(version_base="1.3", config_path="../../conf", config_name="benchmark_diffusionregpose")
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    pipeline_cfg = DiffusionRegPosePipelineConfig.from_hydra_config(cfg.pipeline)
    training_cfg = TrainingConfig.from_hydra_config(cfg.training)
    requested_resume = training_cfg.resume
    resume = resolve_checkpoint(requested_resume) if requested_resume else None
    training_cfg = training_cfg.model_copy(update={
        "resume": resume,
        "skip_training": training_cfg.skip_training or bool(cfg.get("skip_training", False)),
    })
    runner = TrainingRunner(training_cfg, out_dir)
    log = runner.logger()
    log.info("Output dir: %s", out_dir)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    runner.seed_everything()
    use_gpu = training_cfg.accelerator in ("gpu", "cuda")
    data_cfg = DataConfig.from_hydra_config(cfg.data)
    data_module = data_cfg.build(pin_memory=use_gpu)
    module = pipeline_cfg.build(
        max_epochs=training_cfg.max_epochs,
        val_ann_file=str(data_module.val_ann_path),
    )
    log.info("Trainable parameters: %s", f"{sum(p.numel() for p in module.parameters() if p.requires_grad):,}")

    result = runner.run(
        module,
        data_module,
        max_epochs=training_cfg.max_epochs or 1,
        spec=TrainingSpec(plugins=[TrustedCheckpointIO()]),
    )
    log.info("Best checkpoint: %s", result.checkpoint)
    if result.result_path:
        log.info("Test results saved to %s", result.result_path)


if __name__ == "__main__":
    main()
