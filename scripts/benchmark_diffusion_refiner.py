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
        --checkpoint results/.../checkpoints/best.ckpt \\
        --pred-cache results/cache/rtmpose-m_crowdpose_val_predictions.pkl \\
        --ann-file   data/crowdpose/annotations/crowdpose_val.json \\
        --steps 4 --noise-frac 0.4 --passes 2 --task-id 1

Override Hydra run directory:
    ... --run-dir results/diffusion_training/my_run/20260313_120000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics import KeypointAPMetric
from src.model import KeypointDiffusionConfig, KeypointDiffusionModel, PoseDiffusionModule


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _infer_run_dir(checkpoint_path: Path) -> Path:
    """Infer the Hydra run directory.

    Assumes the standard layout: <run_dir>/checkpoints/<name>.ckpt.
    """
    return checkpoint_path.parent.parent


def _infer_run_name(checkpoint_path: Path) -> str:
    """Infer the run name from the checkpoint path.

    Assumes: results/diffusion_training/<run_name>/<ts>/checkpoints/<name>.ckpt
    """
    return checkpoint_path.parent.parent.parent.name


def load_module(
    checkpoint_path: Path,
    run_dir: Path,
    pred_cache: str,
    ann_file: str,
    dino_cache_override: str | None,
    batch_size: int,
    device: torch.device,
) -> PoseDiffusionModule:
    """Reconstruct a PoseDiffusionModule from a checkpoint and its Hydra config."""
    hydra_config_path = run_dir / ".hydra" / "config.yaml"
    if not hydra_config_path.exists():
        raise FileNotFoundError(
            f"Hydra config not found at {hydra_config_path}.\n"
            f"Use --run-dir to point to the correct Hydra output directory."
        )

    cfg = OmegaConf.load(hydra_config_path)
    model_config = KeypointDiffusionConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    model = KeypointDiffusionModel(config=model_config, num_keypoints=15)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp: dict[str, Any] = ckpt["hyper_parameters"]

    # CLI --dino-cache overrides all three split caches at once.
    dino_cache_train = dino_cache_override or hp.get("dino_cache_train")
    dino_cache_val   = dino_cache_override or hp.get("dino_cache_val")
    dino_cache_test  = dino_cache_override or hp.get("dino_cache_test")

    module = PoseDiffusionModule(
        model=model,
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
        lr_T_max=hp.get("lr_T_max"),
        lr_warmup_epochs=hp.get("lr_warmup_epochs", 0),
        lr_min_frac=hp.get("lr_min_frac", 1e-2),
        cfg_guidance_scale=hp.get("cfg_guidance_scale", 1.0),
        val_ann_file=ann_file,
        test_ann_file=ann_file,
        val_pred_cache=pred_cache,
        test_pred_cache=pred_cache,
        num_inference_steps=hp.get("num_inference_steps"),
        ap_batch_size=batch_size,
        start_inference_noise_frac=hp.get("start_inference_noise_frac", 0.4),
        raw_refiner=hp.get("raw_refiner", False),
        dino_model_name=hp.get("dino_model_name"),
        dino_neighborhood_radius=hp.get("dino_neighborhood_radius", 1),
        dino_cache_train=dino_cache_train,
        dino_cache_val=dino_cache_val,
        dino_cache_test=dino_cache_test,
    )

    module.load_state_dict(ckpt["state_dict"])
    module.to(device)
    module.eval()
    return module


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_eval(
    module: PoseDiffusionModule,
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
    p.add_argument("--steps", type=int, default=5,
                   help="DDIM denoising steps per refinement pass. (default: 5)")
    p.add_argument("--noise-frac", type=float, default=0.4,
                   help="Start noise fraction. (default: 0.4)")
    p.add_argument("--passes", type=int, default=1,
                   help="Number of iterative refinement passes. (default: 1)")
    p.add_argument("--task-id", type=int, default=1,
                   help="Task ID used as the output filename. (default: 1)")
    p.add_argument("--dino-cache", default=None,
                   help="Override DINO patch cache path (applied to all splits).")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for AP evaluation. (default: 64)")
    p.add_argument("--zero-delta", action="store_true", default=False,
                   help="Force delta=0 at all times (identity refinement; sanity check).")
    p.add_argument("--disable-noise", action="store_true", default=False,
                   help="Skip noise injection; DDIM starts from clean detector output.")
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

    if args.zero_delta:
        module.model.config.zero_delta = True
        print("zero_delta=True: model outputs identity (delta=0).")

    if args.disable_noise:
        module._disable_noise = True
        print("disable_noise=True: DDIM starts from clean detector output (no noise injected).")

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
