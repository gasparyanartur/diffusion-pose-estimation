#!/usr/bin/env python3
"""
Train the pose-refinement diffusion model on CrowdPose.

Config is managed by Hydra (conf/config.yaml + config groups under conf/).
Run from the project root so that relative data paths resolve correctly.

Quick start (all defaults):
    python scripts/train_pose_refinement.py

Switch config groups:
    python scripts/train_pose_refinement.py model=large
    python scripts/train_pose_refinement.py hardware=gpu
    python scripts/train_pose_refinement.py dino=dinov3_vits16

Override individual values:
    python scripts/train_pose_refinement.py run_name=exp_large train.lr=3e-4 train.max_epochs=200
    python scripts/train_pose_refinement.py model.dim_hidden=256 model.num_layers=6

Resume from a checkpoint:
    python scripts/train_pose_refinement.py train.resume=results/diffusion_training/run/ts/checkpoints/last.ckpt
"""

from __future__ import annotations

# Must be set before any CUDA context is created (i.e. before torch.cuda.init).
# expandable_segments conflicts with mixed large/small allocations (e.g.
# DINO backbone + CNN patch crops) and triggers an internal allocator assert
# (!block->expandable_segment_) in PyTorch ≥ 2.1.
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

import json
import logging
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.data import PoseDataset
from src.model import KeypointDiffusionConfig, KeypointDiffusionModel, PoseDiffusionModule


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
    pl.seed_everything(cfg.train.seed, workers=True)

    # ------------------------------------------------------------------
    # Output directory (Hydra creates it; its location is set by hydra.run.dir
    # in conf/config.yaml, defaulting to out_dir/run_name/timestamp)
    # ------------------------------------------------------------------
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logging(out_dir / "train.log")
    log.info("Run name  : %s", cfg.run_name)
    log.info("Output dir: %s", out_dir)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # ------------------------------------------------------------------
    # Data paths  (to_absolute_path resolves relative to the launch cwd)
    # ------------------------------------------------------------------
    data_root = Path(to_absolute_path(cfg.data.root))
    train_ann = data_root / cfg.data.train_ann
    val_ann   = data_root / cfg.data.val_ann
    test_ann  = data_root / cfg.data.test_ann
    img_dir   = data_root / cfg.data.img_dir

    for path, label in [
        (train_ann, "data.train_ann"),
        (val_ann,   "data.val_ann"),
        (test_ann,  "data.test_ann"),
        (img_dir,   "data.img_dir"),
    ]:
        if not path.exists():
            log.error("%s path does not exist: %s", label, path)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    log.info("Loading datasets from %s ...", data_root)
    train_ds = PoseDataset.from_path(train_ann, img_root=img_dir)
    val_ds   = PoseDataset.from_path(val_ann,   img_root=img_dir)
    test_ds  = PoseDataset.from_path(test_ann,  img_root=img_dir)
    log.info(
        "Dataset sizes — Train: %s  Val: %s  Test: %s",
        f"{len(train_ds):,}", f"{len(val_ds):,}", f"{len(test_ds):,}",
    )

    use_gpu = cfg.hardware.accelerator in ("gpu", "cuda")
    loader_kwargs = dict(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        load_images=cfg.model.use_cnn_cond,  # pre-load images in workers only when CNN is active
        pin_memory=use_gpu,
    )
    train_loader = train_ds.create_dataloader(**loader_kwargs, shuffle=True,  drop_last=True)
    val_loader   = val_ds.create_dataloader(**loader_kwargs,   shuffle=False, drop_last=False)
    test_loader  = test_ds.create_dataloader(**loader_kwargs,  shuffle=False, drop_last=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    use_image_cond = cfg.dino.model_name is not None or cfg.model.use_cnn_cond
    model_config = KeypointDiffusionConfig(
        name=cfg.run_name,
        **OmegaConf.to_container(cfg.model, resolve=True),
    )
    model = KeypointDiffusionModel(config=model_config, num_keypoints=15)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("%s  —  %s trainable parameters", type(model).__name__, f"{n_params:,}")
    if use_image_cond:
        log.info(
            "DINO: model=%s  radius=%d  train_cache=%s  val_cache=%s  test_cache=%s",
            cfg.dino.model_name, cfg.dino.neighborhood_radius,
            cfg.dino.cache_train or "(live)", cfg.dino.cache_val or "(live)",
            cfg.dino.cache_test or "(live)",
        )

    module = PoseDiffusionModule(
        model=model,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        lr_T_max=cfg.train.max_epochs if cfg.train.cosine_lr else None,
        lr_warmup_epochs=cfg.train.lr_warmup_epochs,
        lr_min_frac=cfg.train.lr_min_frac,
        cfg_guidance_scale=cfg.train.cfg_guidance_scale,
        val_ann_file=str(val_ann),
        test_ann_file=str(test_ann),
        val_pred_cache=str(cfg.data.val_pred_cache)   if cfg.data.val_pred_cache  else None,
        test_pred_cache=str(cfg.data.test_pred_cache) if cfg.data.test_pred_cache else None,
        num_inference_steps=cfg.train.num_inference_steps,
        start_inference_noise_frac=cfg.train.start_inference_noise_frac,
        raw_refiner=cfg.train.raw_refiner,
        ap_batch_size=cfg.data.ap_batch_size,
        ap_eval_interval=cfg.data.ap_eval_interval,
        ap_step_counts=list(cfg.train.ap_step_counts) if cfg.train.ap_step_counts else None,
        dino_model_name=cfg.dino.model_name,
        dino_neighborhood_radius=cfg.dino.neighborhood_radius,
        dino_cache_train=str(cfg.dino.cache_train) if cfg.dino.cache_train else None,
        dino_cache_val=str(cfg.dino.cache_val)     if cfg.dino.cache_val   else None,
        dino_cache_test=str(cfg.dino.cache_test)   if cfg.dino.cache_test  else None,
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch{epoch:03d}-val_loss{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if cfg.train.early_stop_patience > 0:
        callbacks.append(EarlyStopping(
            monitor="val/loss",
            patience=cfg.train.early_stop_patience,
            mode="min",
            verbose=True,
        ))

    # ------------------------------------------------------------------
    # Loggers
    # ------------------------------------------------------------------
    loggers = [
        TensorBoardLogger(save_dir=str(out_dir), name="tb_logs", version=""),
        CSVLogger(save_dir=str(out_dir),          name="csv_logs", version=""),
    ]

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    grad_clip = cfg.train.grad_clip if cfg.train.grad_clip > 0 else None
    trainer = pl.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=_devices_arg(cfg.hardware.devices),
        precision=cfg.hardware.precision,
        max_epochs=cfg.train.max_epochs,
        gradient_clip_val=grad_clip,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=cfg.train.log_every_n_steps,
        deterministic=False,
    )

    log.info(
        "Training: epochs=%d  batch=%d  lr=%g  cosine_lr=%s",
        cfg.train.max_epochs, cfg.train.batch_size, cfg.train.lr, cfg.train.cosine_lr,
    )
    resume = str(cfg.train.resume) if cfg.train.resume else None
    if resume:
        log.info("Resuming from: %s", resume)

    trainer.fit(module, train_loader, val_loader, ckpt_path=resume)

    # ------------------------------------------------------------------
    # Test on best checkpoint
    # ------------------------------------------------------------------
    best_ckpt  = trainer.checkpoint_callback.best_model_path   # type: ignore[union-attr]
    best_score = trainer.checkpoint_callback.best_model_score  # type: ignore[union-attr]
    log.info("Best val/loss  : %.6f", best_score)
    log.info("Best checkpoint: %s", best_ckpt)

    if best_ckpt:
        log.info("Evaluating best checkpoint on test set...")
        output = trainer.test(module, dataloaders=test_loader, ckpt_path=best_ckpt)
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
