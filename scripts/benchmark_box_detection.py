#!/usr/bin/env python3
"""
Benchmark a box detector on CrowdPose and report COCO bbox AP.

The detector is selected via the ``box_detector`` config group, making it
easy to compare DiffusionDet against our trained box-diffusion model.

Examples:
    # DiffusionDet on val split (requires Detectron2 + checkpoint):
    python scripts/benchmark_box_detection.py \\
        box_detector=diffusiondet \\
        box_detector.checkpoint=checkpoints/diffdet_coco_res50.pth

    # Our box-diffusion model on test split:
    python scripts/benchmark_box_detection.py \\
        box_detector=box_diffusion \\
        box_detector.checkpoint=results/.../checkpoints/last.ckpt \\
        split=test

    # CPU run:
    python scripts/benchmark_box_detection.py \\
        box_detector=diffusiondet \\
        box_detector.device=cpu
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.configs.base_config import BaseConfig
from src.models.pipelines.object_detectors import BoxDetectorConfig
from src.eval.metrics import BoxAPMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(log_file: Path) -> logging.Logger:
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("benchmark_box_detection")
    log.setLevel(logging.INFO)
    log.propagate = False
    if not log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        log.addHandler(ch)
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def _build_image_id_to_path(ann_file: Path, img_dir: Path) -> dict[int, Path]:
    """Map each COCO image_id to its file path using the annotation JSON."""
    with open(ann_file) as f:
        ann = json.load(f)
    return {
        entry["id"]: img_dir / entry["file_name"]
        for entry in ann["images"]
    }


def _batch(items: list, batch_size: int):
    """Yield successive slices of *items*."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="benchmark_box_detection",
)
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logging(out_dir / "benchmark.log")
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # --- resolve paths ---
    data_root = Path(cfg.data.root)
    img_dir = data_root / cfg.data.img_dir
    if cfg.split == "val":
        ann_file = data_root / cfg.data.val_ann
    elif cfg.split == "test":
        ann_file = data_root / cfg.data.test_ann
    else:
        sys.exit(f"[error] split must be 'val' or 'test', got '{cfg.split}'")

    log.info("Split:    %s", cfg.split)
    log.info("Ann file: %s", ann_file)
    log.info("Img dir:  %s", img_dir)

    # --- build detector ---
    detector_cfg: BoxDetectorConfig = BaseConfig.from_hydra_config(cfg.box_detector)
    log.info("Building detector: %s", type(detector_cfg).__name__)
    detector = detector_cfg.build()
    log.info("Detector ready on device: %s", detector.device)

    # --- load annotations and build image-id map ---
    id_to_path = _build_image_id_to_path(ann_file, img_dir)
    image_ids = sorted(id_to_path.keys())
    log.info("Images to evaluate: %d", len(image_ids))

    # --- run inference and accumulate metric ---
    metric = BoxAPMetric(ann_file=str(ann_file))
    batch_size = cfg.get("batch_size", 8)

    for batch_ids in _batch(image_ids, batch_size):
        paths = [id_to_path[img_id] for img_id in batch_ids]
        result = detector.predict(image_paths=paths, image_ids=batch_ids)

        ids_tensor = torch.tensor(batch_ids, dtype=torch.long)
        metric.update(
            pred_boxes_xywh=result["boxes_xywh"],
            pred_scores=result["scores"],
            image_ids=ids_tensor,
        )

    # --- compute and save ---
    metrics = metric.compute()
    if not metrics:
        log.warning("No predictions accumulated — metrics empty.")
        metrics = {}

    for key, val in metrics.items():
        log.info("  %s: %.4f", key, val)

    out_file = out_dir / "metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved to %s", out_file)


if __name__ == "__main__":
    main()
