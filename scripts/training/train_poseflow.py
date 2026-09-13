#!/usr/bin/env python3
"""Train the endpoint-flow PoseFlow pipeline on CrowdPose."""

from __future__ import annotations

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
from src.training import TrainingConfig, TrainingRunner, TrainingSpec
from src.training.callbacks import TrustedCheckpointIO
from src.utils import resolve_checkpoint


@hydra.main(version_base="1.3", config_path="../../conf", config_name="benchmark_poseflow")
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    pipeline_cfg = PoseFlowPipelineConfig.from_hydra_config(cfg.pipeline)
    training_cfg = TrainingConfig.from_hydra_config(cfg.training)
    requested_resume = training_cfg.resume
    training_cfg = training_cfg.model_copy(update={
        "resume": resolve_checkpoint(requested_resume) if requested_resume else None,
        "skip_training": training_cfg.skip_training or bool(cfg.get("skip_training", False)),
    })
    runner = TrainingRunner(training_cfg, out_dir)
    log = runner.logger()
    log.info("Output dir: %s", out_dir)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    runner.seed_everything()
    data_cfg = DataConfig.from_hydra_config(cfg.data)
    data_module = data_cfg.build(pin_memory=training_cfg.accelerator in ("gpu", "cuda"))
    module = pipeline_cfg.build(
        max_epochs=training_cfg.max_epochs,
        val_ann_file=str(data_module.val_ann_path),
    )
    result = runner.run(
        module, data_module, max_epochs=training_cfg.max_epochs or 1,
        spec=TrainingSpec(plugins=[TrustedCheckpointIO()]),
    )
    log.info("Best checkpoint: %s", result.checkpoint)


if __name__ == "__main__":
    main()
