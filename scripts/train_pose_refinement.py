#!/usr/bin/env python3
"""
Train the pose-refinement diffusion model on CrowdPose.

Data paths default to the project layout defined in src/defaults.py and can be
overridden with the flags below.  Model and training hyperparameters all have
sensible defaults; only pass what you want to change.

Quick start (all defaults):
    python scripts/train_pose_refinement.py

Custom experiment:
    python scripts/train_pose_refinement.py \\
        --run-name exp_large \\
        --batch-size 32 \\
        --lr 3e-4 \\
        --max-epochs 200 \\
        --dim-hidden 256 --num-layers 6 \\
        --devices 1

Resume from a checkpoint:
    python scripts/train_pose_refinement.py \\
        --resume checkpoints/my_run/last.ckpt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src import defaults
from src.data import PoseDataset
from src.model import KeypointDiffusionConfig, KeypointDiffusionModel, PoseDiffusionModule


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Data ---
    data = p.add_argument_group("data")
    data.add_argument(
        "--data-root",
        type=Path,
        default=defaults.DATA_ROOT,
        help="Root directory for all dataset files (default: %(default)s).",
    )
    data.add_argument(
        "--train-ann",
        default="crowdpose/annotations/crowdpose_train.json",
        help="Train annotation file, relative to --data-root (default: %(default)s).",
    )
    data.add_argument(
        "--val-ann",
        default="crowdpose/annotations/crowdpose_val.json",
        help="Validation annotation file, relative to --data-root (default: %(default)s).",
    )
    data.add_argument(
        "--test-ann",
        default="crowdpose/annotations/crowdpose_test.json",
        help="Test annotation file, relative to --data-root (default: %(default)s).",
    )
    data.add_argument(
        "--val-pred-cache",
        type=Path,
        default=None,
        metavar="PKL",
        help=(
            "Prediction cache pickle for the validation split, produced by "
            "scripts/extract_predictions.py.  When set together with --val-ann, "
            "OKS-AP of refined predictions is logged at every validation epoch end."
        ),
    )
    data.add_argument(
        "--test-pred-cache",
        type=Path,
        default=None,
        metavar="PKL",
        help="Same as --val-pred-cache but for the test split.",
    )
    data.add_argument(
        "--ap-batch-size",
        type=int,
        default=256,
        metavar="N",
        help="Batch size when iterating the prediction cache for AP evaluation (default: %(default)s).",
    )
    data.add_argument(
        "--ap-eval-interval",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Run val AP evaluation every N epochs.  "
            "0 disables val AP (AP is still computed once on the test set after training).  "
            "Default: %(default)s."
        ),
    )
    data.add_argument(
        "--img-dir",
        default="crowdpose/images",
        help="Image directory, relative to --data-root (default: %(default)s).",
    )
    data.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: %(default)s).",
    )

    # --- Model architecture ---
    arch = p.add_argument_group("model architecture")
    arch.add_argument("--dim-pos",    type=int, default=128, help="Position embedding dim (default: %(default)s).")
    arch.add_argument("--dim-index",  type=int, default=128, help="Keypoint-index embedding dim (default: %(default)s).")
    arch.add_argument("--dim-time",   type=int, default=128, help="Timestep embedding dim (default: %(default)s).")
    arch.add_argument("--dim-bone",   type=int, default=128, help="Bone-vector embedding dim (default: %(default)s).")
    arch.add_argument("--dim-hidden", type=int, default=128, help="GNN hidden dim (default: %(default)s).")
    arch.add_argument("--num-layers", type=int, default=4,   help="GNN depth (default: %(default)s).")
    arch.add_argument(
        "--num-train-steps",
        type=int,
        default=100,
        help="Number of diffusion timesteps (default: %(default)s).",
    )

    # --- Optimiser / schedule ---
    opt = p.add_argument_group("optimiser")
    opt.add_argument("--lr",           type=float, default=1e-3,  help="Peak learning rate (default: %(default)s).")
    opt.add_argument("--weight-decay", type=float, default=0.0,   help="AdamW weight decay (default: %(default)s).")
    opt.add_argument(
        "--cosine-lr",
        action="store_true",
        help="Apply cosine annealing LR schedule over --max-epochs.",
    )
    opt.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        metavar="S",
        help=(
            "DDIM denoising steps used during AP evaluation.  "
            "Defaults to round(--inference-noise-frac × --num-train-steps) when not set."
        ),
    )
    opt.add_argument(
        "--inference-noise-frac",
        type=float,
        default=0.1,
        metavar="F",
        help=(
            "Fraction of diffusion timesteps to use as the DDIM starting point "
            "during AP evaluation.  Noise at level F×T is added to the canonical "
            "detector predictions before DDIM; this keeps the model in-distribution "
            "while staying close to the detector output.  "
            "0 = no noise (model never called), 1 = full noise (AP=0 for untrained model).  "
            "Default: %(default)s."
        ),
    )

    # --- Training ---
    train = p.add_argument_group("training")
    train.add_argument("--batch-size",   type=int, default=256,  help="Images per batch (default: %(default)s).")
    train.add_argument("--max-epochs",   type=int, default=100, help="Maximum training epochs (default: %(default)s).")
    train.add_argument("--seed",         type=int, default=42,  help="Global random seed (default: %(default)s).")
    train.add_argument("--grad-clip",    type=float, default=1.0, help="Gradient clipping norm (default: %(default)s; 0 = disabled).")
    train.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        metavar="N",
        help="Stop training when val_loss does not improve for N epochs.  0 = disabled (default).",
    )

    # --- Hardware ---
    hw = p.add_argument_group("hardware")
    hw.add_argument("--accelerator", default="auto",    help="Lightning accelerator: auto|gpu|cpu (default: %(default)s).")
    hw.add_argument("--devices",     default="auto",    help="Devices to use, e.g. '1', '0,1', or 'auto' (default: %(default)s).")
    hw.add_argument(
        "--precision",
        default="32-true",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help="Training precision (default: %(default)s).",
    )

    # --- Output ---
    out = p.add_argument_group("output")
    out.add_argument("--run-name", default=None, help="Sub-directory name under --out-dir (default: model config name).")
    out.add_argument(
        "--out-dir",
        type=Path,
        default=defaults.RESULTS_DIR / "diffusion_training",
        help="Root directory for checkpoints and logs (default: %(default)s).",
    )
    out.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a .ckpt file.",
    )
    out.add_argument(
        "--log-every-n-steps",
        type=int,
        default=50,
        help="Log metrics every N training steps (default: %(default)s).",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _devices_arg(raw: str) -> int | str | list[int]:
    """Convert the --devices string to the type Lightning expects."""
    if raw == "auto":
        return "auto"
    if "," in raw:
        return [int(d) for d in raw.split(",")]
    return int(raw)


def _setup_logging(log_file: Path) -> logging.Logger:
    """Return a logger that writes INFO+ to stdout *and* to *log_file*."""
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:  # avoid duplicate handlers on re-use
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _build_hparams(args: argparse.Namespace, run_name: str, timestamp: str) -> dict:
    """Serialise all CLI args plus derived identifiers into a flat dict."""
    hparams = {"run_name": run_name, "timestamp": timestamp}
    for key, val in vars(args).items():
        hparams[key] = str(val) if isinstance(val, Path) else val
    return hparams


def _save_hparams(hparams: dict, path: Path) -> None:
    """Write *hparams* to *path* as YAML."""
    with open(path, "w") as fh:
        yaml.dump(hparams, fh, default_flow_style=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)

    # ------------------------------------------------------------------
    # Paths & logging  (out_dir created first so the log file has a home)
    # ------------------------------------------------------------------
    run_name  = args.run_name or f"diffusion_d{args.dim_hidden}_l{args.num_layers}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = args.out_dir / run_name / timestamp
    ckpt_dir  = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logging(out_dir / "train.log")
    log.info("Run name  : %s", run_name)
    log.info("Output dir: %s", out_dir)

    # ------------------------------------------------------------------
    # Validate data paths
    # ------------------------------------------------------------------
    data_root = args.data_root
    train_ann = data_root / args.train_ann
    val_ann   = data_root / args.val_ann
    test_ann  = data_root / args.test_ann
    img_dir   = data_root / args.img_dir

    for path, label in [
        (train_ann, "--train-ann"), (val_ann, "--val-ann"),
        (test_ann, "--test-ann"),   (img_dir, "--img-dir"),
    ]:
        if not path.exists():
            log.error("%s path does not exist: %s", label, path)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Hyperparameters  (saved to YAML before anything else runs)
    # ------------------------------------------------------------------
    hparams = _build_hparams(args, run_name, timestamp)
    hparams_path = out_dir / "hparams.yaml"
    _save_hparams(hparams, hparams_path)
    log.info("Hyperparameters saved to %s", hparams_path)
    for key, val in hparams.items():
        log.info("  %-24s %s", key, val)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    log.info("Loading train annotations from %s", train_ann)
    train_dataset = PoseDataset.from_path(train_ann, img_root=img_dir)
    log.info("Loading val annotations   from %s", val_ann)
    val_dataset   = PoseDataset.from_path(val_ann,   img_root=img_dir)
    log.info("Loading test annotations  from %s", test_ann)
    test_dataset  = PoseDataset.from_path(test_ann,  img_root=img_dir)
    log.info(
        "Dataset sizes — Train: %s  Val: %s  Test: %s",
        f"{len(train_dataset):,}", f"{len(val_dataset):,}", f"{len(test_dataset):,}",
    )

    train_loader = train_dataset.create_dataloader(
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers
    )
    val_loader = val_dataset.create_dataloader(
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    test_loader = test_dataset.create_dataloader(
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_config = KeypointDiffusionConfig(
        name=run_name,
        dim_pos=args.dim_pos,
        dim_index=args.dim_index,
        dim_time=args.dim_time,
        dim_bone=args.dim_bone,
        dim_hidden=args.dim_hidden,
        num_layers=args.num_layers,
        num_train_steps=args.num_train_steps,
    )
    model = KeypointDiffusionModel(config=model_config, num_keypoints=15)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("%s  —  %s trainable parameters", type(model).__name__, f"{n_params:,}")

    module = PoseDiffusionModule(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_T_max=args.max_epochs if args.cosine_lr else None,
        val_ann_file=str(data_root / args.val_ann),
        test_ann_file=str(data_root / args.test_ann),
        val_pred_cache=str(args.val_pred_cache)  if args.val_pred_cache  else None,
        test_pred_cache=str(args.test_pred_cache) if args.test_pred_cache else None,
        num_inference_steps=args.num_inference_steps,
        inference_noise_frac=args.inference_noise_frac,
        ap_batch_size=args.ap_batch_size,
        ap_eval_interval=args.ap_eval_interval,
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch{epoch:03d}-val_loss{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if args.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=args.early_stop_patience,
                mode="min",
                verbose=True,
            )
        )

    # ------------------------------------------------------------------
    # Loggers
    # ------------------------------------------------------------------
    loggers = [
        TensorBoardLogger(save_dir=str(out_dir), name="tb_logs", version=""),
        CSVLogger(save_dir=str(out_dir), name="csv_logs", version=""),
    ]

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    grad_clip = args.grad_clip if args.grad_clip > 0 else None

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=_devices_arg(args.devices),
        precision=args.precision,
        max_epochs=args.max_epochs,
        gradient_clip_val=grad_clip,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=False,  # deterministic=True is slower; flip if needed
    )

    log.info("Epochs    : %d", args.max_epochs)
    log.info("Batch size: %d  |  LR: %g", args.batch_size, args.lr)
    if args.cosine_lr:
        log.info("LR schedule: cosine annealing over %d epochs", args.max_epochs)
    if args.early_stop_patience > 0:
        log.info("Early stop patience: %d", args.early_stop_patience)
    if args.resume:
        log.info("Resuming from: %s", args.resume)

    trainer.fit(module, train_loader, val_loader, ckpt_path=str(args.resume) if args.resume else None)

    # ------------------------------------------------------------------
    # Post-training summary
    # ------------------------------------------------------------------
    best_ckpt  = trainer.checkpoint_callback.best_model_path   # type: ignore[union-attr]
    best_score = trainer.checkpoint_callback.best_model_score  # type: ignore[union-attr]
    log.info("Best val_loss : %.6f", best_score)
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
