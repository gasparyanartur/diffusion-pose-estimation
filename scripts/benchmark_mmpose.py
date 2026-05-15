#!/usr/bin/env python3
"""
Run a registered MMPose benchmark configuration.

Loads a predefined config from src/configs.REGISTRY by name, then applies any
command-line overrides on top.  All override flags are optional; omitting them
keeps the value from the registered config.

List available configs:
    python scripts/benchmark_rtmpose.py --list

Example: (RTMPose-M on CrowdPose):
    source .env && python scripts/benchmark_rtmpose.py \\
        rtmpose-m_8xb64-210e_crowdpose-256x192

Override a single field (e.g. switch to oracle mode):
    source .env && python scripts/benchmark_rtmpose.py \\
        rtmpose-m_8xb64-210e_crowdpose-256x192 \\
        --bbox-file none
"""

import argparse
import sys
from pathlib import Path

from src import defaults
from src.benchmark import run_benchmark
from src.configs import REGISTRY

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "name",
        metavar="NAME",
        help=f"Registered config name (default.  Use --list to see all.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print all registered config names and exit.",
    )
    # --- optional overrides (all default to None = keep config value) ---
    p.add_argument("--config", type=Path, default=None, help="Override: MMPose config (.py).")
    p.add_argument("--checkpoint", type=Path, default=None, help="Override: checkpoint path.")
    p.add_argument("--checkpoint-url", default=None, help="Override: checkpoint download URL.")
    p.add_argument("--data-root", type=Path, default=None, help="Override: dataset root directory.")
    p.add_argument("--ann-file", default=None, help="Override: annotation file (relative to data-root).")
    p.add_argument("--img-prefix", default=None, help="Override: image directory (relative to data-root).")
    p.add_argument("--metainfo", type=Path, default=None, help="Override: dataset metainfo config.")
    p.add_argument(
        "--bbox-file",
        default=None,
        metavar="PATH",
        help="Override: detector-box file (relative to data-root).  Pass 'none' for oracle mode.",
    )
    p.add_argument("--name", default=None, help="Results sub-directory name (default: model name).")
    p.add_argument("--out", type=Path, default=None, help="Metrics output file (default: results/<name>/metrics.yaml).")
    p.add_argument("--show-dir", default=None, help="Directory to save visualisations.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print("Registered configurations:")
        for name in REGISTRY:
            print(f"  {name}")
        return

    name = args.name.lower()
    if name not in REGISTRY:
        names = "\n  ".join(REGISTRY)
        sys.exit(f"[error] Unknown model '{args.name}'.\nAvailable:\n  {names}")

    # Start from the registered config, then layer CLI overrides on top.
    cfg = dict(REGISTRY[name])

    if args.config is not None:
        cfg["config_path"] = args.config
    if args.checkpoint is not None:
        cfg["checkpoint"] = args.checkpoint
    if args.checkpoint_url is not None:
        cfg["checkpoint_url"] = args.checkpoint_url
    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    if args.ann_file is not None:
        cfg["ann_file"] = args.ann_file
    if args.img_prefix is not None:
        cfg["img_prefix"] = args.img_prefix
    if args.metainfo is not None:
        cfg["metainfo_config"] = args.metainfo
    if args.bbox_file is not None:
        cfg["bbox_file"] = (
            None if args.bbox_file.lower() in ("none", "null", "") else args.bbox_file
        )

    name = args.name or args.model
    out = args.out or (defaults.RESULTS_DIR / name / "metrics.yaml")

    run_benchmark(
        cfg,
        work_dir=defaults.RESULTS_DIR / name / "work_dir",
        out=out,
        show_dir=args.show_dir,
    )


if __name__ == "__main__":
    main()
