#!/usr/bin/env python3
"""
Benchmark a trained diffusion refiner with a single set of inference parameters.

Intended to be driven by scripts/run_benchmark_sweep.sh, which calls this
script once per task line from a .sweep file.

The Hydra config is read from <run_dir>/.hydra/config.yaml.  By default,
run_dir is inferred as checkpoint.parent.parent (the checkpoint lives at
<run_dir>/checkpoints/<name>.ckpt).

Results are written as JSON to:
    results/diffusion-refiner/<run_name>/<task_id>.json   (default)

Usage (single run):
    python scripts/benchmark_diffusion_refiner.py \\
        --checkpoint results/.../checkpoints/last.ckpt \\
        --pred-cache results/cache/rtmpose-m_crowdpose_val_predictions.pkl \\
        --ann-file   data/crowdpose/annotations/crowdpose_val.json \\
        --steps 1 --noise-frac 0.2 --passes 1 --task-id 1

Override Hydra run directory:
    ... --run-dir results/pose_estimator_training/my_run/20260505_120000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from src.metrics import KeypointAPMetric
from src.pose_estimator import PoseEstimatorConfig
from src.training.training_module import PoseEstimatorTrainingModule


# ---------------------------------------------------------------------------
# Path inference
# ---------------------------------------------------------------------------


def _infer_run_dir(checkpoint_path: Path) -> Path:
    """Standard layout: <run_dir>/checkpoints/<name>.ckpt."""
    return checkpoint_path.parent.parent


def _infer_run_name(checkpoint_path: Path) -> str:
    """Layout: results/.../<run_name>/<ts>/checkpoints/<name>.ckpt."""
    return checkpoint_path.parent.parent.parent.name


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _override_dino_cache(pose_estimator_cfg: Any, dino_cache: str) -> None:
    """Replace cache_paths on a CachedDinoFeatureExtractorConfig before build()."""
    fe = pose_estimator_cfg.feature_extractor
    if hasattr(fe, "cache_paths"):
        fe.cache_paths = [dino_cache]


def load_module(
    checkpoint_path: Path,
    run_dir: Path,
    pred_cache: str,
    ann_file: str,
    dino_cache_override: str | None,
    batch_size: int,
    device: torch.device,
) -> PoseEstimatorTrainingModule:
    """Reconstruct a PoseEstimatorTrainingModule from a checkpoint and its Hydra config."""
    hydra_config_path = run_dir / ".hydra" / "config.yaml"
    if not hydra_config_path.exists():
        raise FileNotFoundError(
            f"Hydra config not found at {hydra_config_path}.\n"
            f"Use --run-dir to point to the correct Hydra output directory."
        )

    cfg = OmegaConf.load(hydra_config_path)
    pose_cfg = PoseEstimatorConfig.from_hydra_config(cfg.pose_estimator)
    if dino_cache_override is not None:
        _override_dino_cache(pose_cfg, dino_cache_override)
    pose_estimator = pose_cfg.build()

    module = PoseEstimatorTrainingModule.load_from_checkpoint(
        checkpoint_path,
        pose_estimator=pose_estimator,
        val_ann_file=ann_file,
        test_ann_file=ann_file,
        val_pred_cache=pred_cache,
        test_pred_cache=pred_cache,
        ap_batch_size=batch_size,
        map_location="cpu",
    )

    module.to(device)
    module.eval()
    return module


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_eval(
    module: PoseEstimatorTrainingModule,
    pred_cache: str,
    ann_file: str,
    ddim_steps: int,
    noise_frac: float,
    passes: int,
) -> dict[str, Any]:
    """Run a single evaluation and return a result dict."""
    module.num_inference_steps = ddim_steps
    module._start_inference_noise_frac = noise_frac

    metric = KeypointAPMetric(ann_file)
    desc = f"steps={ddim_steps}  noise_frac={noise_frac:.2f}  passes={passes}"
    module._run_ap_eval(pred_cache, metric, desc=desc, num_steps=passes)
    ap = metric.compute()

    return {
        "steps": ddim_steps,
        "noise_frac": noise_frac,
        "passes": passes,
        "guidance_scale": module.cfg_guidance_scale,
        "raw_refiner": module._raw_refiner,
        "disable_noise": module._disable_noise,
        **{k: round(float(v), 4) for k, v in ap.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a .ckpt file.")
    p.add_argument("--pred-cache", required=True,
                   help="Path to a detector predictions .pkl cache.")
    p.add_argument("--ann-file", required=True,
                   help="Path to a COCO-style annotations .json.")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Hydra run directory containing .hydra/config.yaml. "
                        "Default: checkpoint.parent.parent.")
    p.add_argument("--steps", type=int, default=1,
                   help="DDIM denoising steps per refinement pass. (default: 1)")
    p.add_argument("--noise-frac", type=float, default=0.2,
                   help="Start noise fraction. (default: 0.2)")
    p.add_argument("--passes", type=int, default=1,
                   help="Number of iterative refinement passes. (default: 1)")
    p.add_argument("--task-id", type=int, default=1,
                   help="Task ID used as the output filename. (default: 1)")
    p.add_argument("--dino-cache", default=None,
                   help="Override DINO feature cache path (single dir applied to "
                        "the feature extractor's cache_paths).")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for AP evaluation. (default: 64)")
    p.add_argument("--disable-noise", action="store_true", default=False,
                   help="Skip noise injection; DDIM starts from clean detector output.")
    p.add_argument("--raw-refiner", action="store_true", default=False,
                   help="Skip noise; force 1 step (pure single-pass denoising). "
                        "Overrides the checkpoint value.")
    p.add_argument("--cfg-guidance-scale", type=float, default=None,
                   help="CFG guidance scale. Overrides the checkpoint value. "
                        "1.0 = no guidance. Default: use checkpoint value.")
    p.add_argument("--device", default=None,
                   help="Device: cpu | cuda | cuda:N. Default: cuda if available, else cpu.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directory to write <task_id>.json. "
                        "Default: results/diffusion-refiner/<run_name>/")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = args.run_dir or _infer_run_dir(args.checkpoint)
    run_name = _infer_run_name(args.checkpoint)
    out_dir = args.out_dir or (Path("results") / "diffusion-refiner" / run_name)
    out_path = out_dir / f"{args.task_id}.json"

    print(f"Device     : {device}")
    print(f"Run dir    : {run_dir}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Pred cache : {args.pred_cache}")
    print(f"Ann file   : {args.ann_file}")
    print(f"Steps      : {args.steps}")
    print(f"Noise frac : {args.noise_frac}")
    print(f"Passes     : {args.passes}")
    print(f"Output     : {out_path}")
    print()

    module = load_module(
        checkpoint_path=args.checkpoint,
        run_dir=run_dir,
        pred_cache=args.pred_cache,
        ann_file=args.ann_file,
        dino_cache_override=args.dino_cache,
        batch_size=args.batch_size,
        device=device,
    )

    if args.disable_noise:
        module._disable_noise = True
        print("disable_noise=True: DDIM starts from clean detector output (no noise).")

    if args.raw_refiner:
        module._raw_refiner = True
        print("raw_refiner=True: single backward pass, no DDIM noise.")

    if args.cfg_guidance_scale is not None:
        module.cfg_guidance_scale = args.cfg_guidance_scale
        print(f"cfg_guidance_scale overridden to {args.cfg_guidance_scale}")

    with torch.no_grad():
        result = run_eval(
            module=module,
            pred_cache=args.pred_cache,
            ann_file=args.ann_file,
            ddim_steps=args.steps,
            noise_frac=args.noise_frac,
            passes=args.passes,
        )

    print("Results:")
    for key, value in result.items():
        print(f"  {key:<30} {value}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
