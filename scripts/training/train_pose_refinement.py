#!/usr/bin/env python3
"""
Train the fixed-slot baseline pose pipeline on CrowdPose.

Config is managed by Hydra through one pipeline configuration.
Run from the project root so that relative data paths resolve correctly.

Quick start:
    python scripts/training/train_pose_refinement.py

Smoke run:
    python scripts/training/train_pose_refinement.py training.max_epochs=2 data=crowdpose_set_prediction_debug

Override individual values:
    python scripts/training/train_pose_refinement.py pipeline.lr=3e-4 training.max_epochs=100
"""

from __future__ import annotations

# Must be set before any CUDA context is created (i.e. before torch.cuda.init).
# expandable_segments conflicts with mixed large/small allocations (e.g.
# Large feature-extractor runs with CNN patch crops can trigger an internal
# allocator assert
# (!block->expandable_segment_) in PyTorch ≥ 2.1.
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.data import DataConfig
from src.models.pipelines.pose_estimators.baseline import BaselinePosePipelineConfig
from src.training import TrainingConfig, TrainingRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    pipeline_cfg = BaselinePosePipelineConfig.from_hydra_config(cfg.pipeline)
    training_cfg = TrainingConfig.from_hydra_config(cfg.training)
    runner = TrainingRunner(training_cfg, out_dir)
    log = runner.logger()
    log.info("Run name  : %s", cfg.run_name)
    log.info("Output dir: %s", out_dir)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    data_cfg = DataConfig.from_hydra_config(cfg.data)
    runner.seed_everything()

    use_gpu = training_cfg.accelerator in ("gpu", "cuda")
    data_module = data_cfg.build(pin_memory=use_gpu)
    module = pipeline_cfg.build(
        val_ann_file=str(data_module.val_ann_path),
        test_ann_file=str(data_module.test_ann_path),
        max_persons=getattr(data_module, "max_persons", None),
        max_epochs=training_cfg.max_epochs,
    )

    n_params = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    log.info("Pipeline         : %s", type(module).__name__)
    log.info("Backbone         : %s", type(module.backbone).__name__)
    log.info("Trainable params : %s", f"{n_params:,}")
    log.info("DataModule       : %s  (batch=%d, num_workers=%d)",
             type(data_module).__name__, data_cfg.batch_size, data_cfg.num_workers)

    log.info(
        "Training: epochs=%d  batch=%d  lr=%g  cosine_lr=%s",
        training_cfg.max_epochs, data_cfg.batch_size, pipeline_cfg.lr, pipeline_cfg.cosine_lr,
    )
    result = runner.run(
        module,
        data_module,
        max_epochs=training_cfg.max_epochs,
    )
    log.info("Best checkpoint: %s", result.checkpoint)
    if result.result_path:
        log.info("Test results saved to %s", result.result_path)


if __name__ == "__main__":
    main()
