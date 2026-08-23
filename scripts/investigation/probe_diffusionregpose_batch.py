#!/usr/bin/env python3
"""Probe one native DiffusionRegPose optimizer step at a requested batch size."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch import Tensor

from src.data import DataConfig
from src.models.pipelines.pose_estimators.diffusionregpose import (
    DiffusionRegPosePipelineConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _move_to_device(value, device):
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _optimizer(module):
    configured = module.configure_optimizers()
    if isinstance(configured, dict):
        return configured["optimizer"]
    return configured


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the batch-size probe")
    device = torch.device(args.device)
    config_dir = Path(__file__).resolve().parents[2] / "conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config: DictConfig = compose(
            config_name="benchmark_diffusionregpose_lowres",
            overrides=[
                f"data.batch_size={args.batch_size}",
                "training.accumulate_grad_batches=1",
            ],
        )

    data_config = DataConfig.from_hydra_config(config.data)
    data_module = data_config.build(pin_memory=device.type == "cuda")
    data_module.setup()
    data_module._trainer = SimpleNamespace(training=True)
    raw_batch = next(iter(data_module.train_dataloader()))
    batch = data_module.on_after_batch_transfer(raw_batch, 0)
    batch = _move_to_device(batch, device)

    pipeline_config = DiffusionRegPosePipelineConfig.from_hydra_config(config.pipeline)
    module = pipeline_config.build(max_epochs=1).to(device).train()
    optimizer = _optimizer(module)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    losses = module._losses_for_batch(batch)
    losses["total"].backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
    else:
        peak_memory = 0.0
    print(
        f"batch_size={args.batch_size} image_shape={tuple(batch['image'].shape)} "
        f"loss={losses['total'].item():.6f} peak_cuda_gib={peak_memory:.3f}"
    )


if __name__ == "__main__":
    main()
