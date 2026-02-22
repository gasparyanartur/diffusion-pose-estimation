#!/usr/bin/env python3
"""
Benchmark RTMPose-M on the local CrowdPose test set using MMPose.

--- Evaluation modes ---

  GT-box mode (default, --bbox-file not set):
      Feeds ground-truth bounding boxes directly to the pose estimator.
      This is an ORACLE / UPPER-BOUND evaluation – results will be higher
      than the numbers reported in the MMPose model zoo.

  Detector-box mode (--bbox-file <path>):
      Feeds pre-computed detector boxes, reproducing the official two-stage
      pipeline (detector → pose estimator) and matching the reported numbers.
      Run scripts/prepare_crowdpose.py to download the detection file, then:
          source .env && python scripts/benchmark_rtmpose.py \
              --bbox-file crowdpose/annotations/det_for_crowd_test_0.1_0.5.json

Usage:
    source .env && python scripts/benchmark_rtmpose.py
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# Make the project root importable regardless of cwd or how the script is invoked
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import defaults  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress(count: int, block: int, total: int) -> None:
    pct = min(100, int(count * block * 100 / total))
    print(f"\r  {pct:3d}%", end="", flush=True)


def download_checkpoint(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[checkpoint] Using existing file: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint] Downloading from:\n  {url}")
    urllib.request.urlretrieve(url, dest, _progress)
    print(f"\n[checkpoint] Saved to: {dest}")


def patch_config(cfg, checkpoint: Path, bbox_file: str | None, benchmark_name: str) -> None:
    """
    Override config paths to match the local data layout.

    bbox_file: path to pre-computed detector boxes (relative to DATA_ROOT).
        If None, ground-truth boxes are used (oracle / upper-bound evaluation).
        Pass defaults.CROWDPOSE_BBOX_FILE to reproduce the official MMPose
        model-zoo numbers.
    """
    # mmengine expects data_root as a string with a trailing slash
    data_root = str(defaults.DATA_ROOT) + "/"
    results_root = defaults.RESULTS_DIR
    workdir = results_root / benchmark_name
    workdir.mkdir(parents=True, exist_ok=True)

    cfg.work_dir = str(workdir)
    cfg.load_from = str(checkpoint)
    cfg.launcher = "none"

    cfg.results_path = str(workdir / "results.json")

    # Inject absolute metainfo path so cwd never matters
    metainfo = dict(from_file=str(defaults.CROWDPOSE_METAINFO_CONFIG))

    for dl in (cfg.test_dataloader, cfg.val_dataloader):
        dl.dataset.data_root = data_root
        dl.dataset.ann_file = defaults.CROWDPOSE_ANN_FILE
        dl.dataset.data_prefix = dict(img=defaults.CROWDPOSE_IMG_PREFIX)
        dl.dataset.metainfo = metainfo
        if bbox_file is not None:
            dl.dataset.bbox_file = data_root + bbox_file
        elif "bbox_file" in dl.dataset:
            del dl.dataset["bbox_file"]

    abs_ann = data_root + defaults.CROWDPOSE_ANN_FILE
    cfg.test_evaluator.ann_file = abs_ann
    cfg.val_evaluator.ann_file = abs_ann


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=defaults.RTMPOSE_CROWDPOSE_CHECKPOINT,
        help="Local checkpoint path (downloaded automatically if absent).",
    )
    p.add_argument(
        "--checkpoint-url",
        default=defaults.RTMPOSE_CROWDPOSE_CHECKPOINT_URL,
        help="URL to download the checkpoint from.",
    )
    p.add_argument(
        "--bbox-file",
        type=str,
        default=defaults.CROWDPOSE_BBOX_FILE,
        metavar="PATH",
        help=(
            "Path (relative to data/) to pre-computed detector bounding boxes. "
            "Required to reproduce the official MMPose model-zoo numbers. "
            f"Default file: {defaults.CROWDPOSE_BBOX_FILE}"
        ),
    )
    p.add_argument(
        "--show-dir",
        type=str,
        default=None,
        help="Optional directory to save visualised predictions.",
    )
    p.add_argument(
        "--benchmark_name",
        type=str,
        default="rtmpose_crowdpose_benchmark",
        help="Subdirectory in results/ to save outputs to.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure the mmpose source clone is importable
    mmpose_src = str(defaults.MMPOSE_DIR)
    if mmpose_src not in sys.path:
        sys.path.insert(0, mmpose_src)

    import mmengine
    from mmengine.config import Config
    from mmengine.hooks import Hook
    from mmengine.runner import Runner

    download_checkpoint(args.checkpoint_url, args.checkpoint)

    ann_path = defaults.DATA_ROOT / defaults.CROWDPOSE_ANN_FILE
    if not ann_path.exists():
        sys.exit(f"[error] Annotation file not found: {ann_path}")

    img_dir = defaults.DATA_ROOT / defaults.CROWDPOSE_IMG_PREFIX
    if not img_dir.is_dir():
        sys.exit(f"[error] Image directory not found: {img_dir}")

    if args.bbox_file is None or args.bbox_file.lower() in ("none", "null"):
        print(
            "[warning] No --bbox-file supplied: using GT bounding boxes (oracle mode).\n"
            "          Results will be HIGHER than the official model-zoo numbers.\n"
            "          To reproduce official numbers run scripts/prepare_crowdpose.py, then:\n"
            f"          --bbox-file {defaults.CROWDPOSE_BBOX_FILE}\n"
        )
    else:
        bbox_path = defaults.DATA_ROOT / args.bbox_file
        if not bbox_path.exists():
            sys.exit(f"[error] bbox-file not found: {bbox_path}")
        print(f"[bbox-file] {args.bbox_file} (detector-box mode – comparable to model zoo)\n")

    cfg = Config.fromfile(str(defaults.RTMPOSE_CROWDPOSE_CONFIG))
    patch_config(cfg, args.checkpoint, args.bbox_file, args.benchmark_name)

    if args.show_dir:
        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.out_dir = args.show_dir

    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    runner = Runner.from_cfg(cfg)

    if cfg.results_path is not None:
        class _SaveMetrics(Hook):
            def after_test_epoch(self, _, metrics=None):
                if metrics:
                    mmengine.dump(metrics, cfg.results_path)
                    print(f"\n[results] Metrics written to: {cfg.results_path}")

        runner.register_hook(_SaveMetrics(), "LOWEST")

    eval_mode = f"detector boxes ({args.bbox_file})" if args.bbox_file else "GT boxes (oracle)"
    print(f"\n[config]     {defaults.RTMPOSE_CROWDPOSE_CONFIG.relative_to(defaults.PROJECT_DIR)}")
    print(f"[checkpoint] {Path(args.checkpoint).relative_to(defaults.PROJECT_DIR)}")
    print(f"[eval mode]  {eval_mode}")
    print(f"[data_root]  {defaults.DATA_ROOT}")
    print(f"[ann_file]   {defaults.CROWDPOSE_ANN_FILE}")
    print(f"[img_prefix] {defaults.CROWDPOSE_IMG_PREFIX}")
    print(f"[work_dir]   {cfg.work_dir}\n")

    runner.test()


if __name__ == "__main__":
    main()
