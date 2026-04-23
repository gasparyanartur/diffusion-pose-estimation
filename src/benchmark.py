"""
Core benchmarking logic for MMPose models.

The public entry point is run_benchmark(), which accepts a config dict
(as defined in src/configs.py) plus per-run overrides (work_dir, out, show_dir).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


def _progress(count: int, block: int, total: int) -> None:
    pct = min(100, int(count * block * 100 / total))
    print(f"\r  {pct:3d}%", end="", flush=True)


def download_checkpoint(url: str, dest: Path) -> None:
    """Download *url* to *dest*, skipping if the file already exists."""
    if dest.exists():
        print(f"[checkpoint] Using existing file: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint] Downloading from:\n  {url}")
    urllib.request.urlretrieve(url, dest, _progress)
    print(f"\n[checkpoint] Saved to: {dest}")


def _patch_config(
    cfg,
    *,
    checkpoint: Path,
    data_root: Path,
    ann_file: str,
    img_prefix: str,
    metainfo_config: Path,
    bbox_file: str | None,
    work_dir: Path,
) -> None:
    # metainfo_config must be absolute for parse_pose_metainfo to resolve correctly.
    data_root_str = str(data_root) + "/"

    cfg.work_dir = str(work_dir)
    cfg.load_from = str(checkpoint)
    cfg.launcher = "none"

    metainfo = dict(from_file=str(metainfo_config))

    for dl in (cfg.test_dataloader, cfg.val_dataloader):
        dl.dataset.data_root = data_root_str
        dl.dataset.ann_file = ann_file
        dl.dataset.data_prefix = dict(img=img_prefix)
        dl.dataset.metainfo = metainfo
        if bbox_file is not None:
            dl.dataset.bbox_file = data_root_str + bbox_file
        elif "bbox_file" in dl.dataset:
            del dl.dataset["bbox_file"]

    abs_ann = data_root_str + ann_file
    cfg.test_evaluator.ann_file = abs_ann
    cfg.val_evaluator.ann_file = abs_ann


def run_benchmark(
    config: dict,
    *,
    work_dir: Path,
    out: Path,
    show_dir: str | None = None,
) -> None:
    """Validate, patch config, build the mmengine Runner, and run evaluation."""
    mmpose_dir: Path = config["mmpose_dir"]
    config_path: Path = config["config_path"]
    checkpoint: Path = config["checkpoint"]
    checkpoint_url: str | None = config.get("checkpoint_url")
    data_root: Path = config["data_root"]
    ann_file: str = config["ann_file"]
    img_prefix: str = config["img_prefix"]
    metainfo_config: Path = config["metainfo_config"]
    bbox_file: str | None = config.get("bbox_file")

    mmpose_src = str(mmpose_dir)
    if mmpose_src not in sys.path:
        sys.path.insert(0, mmpose_src)

    import mmengine
    from mmengine.config import Config
    from mmengine.hooks import Hook
    from mmengine.runner import Runner

    if checkpoint_url:
        download_checkpoint(checkpoint_url, checkpoint)

    ann_path = data_root / ann_file
    if not ann_path.exists():
        sys.exit(f"[error] Annotation file not found: {ann_path}")

    img_dir = data_root / img_prefix
    if not img_dir.is_dir():
        sys.exit(f"[error] Image directory not found: {img_dir}")

    if bbox_file is None:
        print(
            "[warning] No bbox_file set: using GT bounding boxes (oracle mode).\n"
            "          Results will be HIGHER than the official model-zoo numbers.\n"
        )
    else:
        bbox_path = data_root / bbox_file
        if not bbox_path.exists():
            sys.exit(f"[error] bbox_file not found: {bbox_path}")
        print(f"[bbox-file] {bbox_file} (detector-box mode)\n")

    cfg = Config.fromfile(str(config_path))
    work_dir.mkdir(parents=True, exist_ok=True)
    _patch_config(
        cfg,
        checkpoint=checkpoint,
        data_root=data_root,
        ann_file=ann_file,
        img_prefix=img_prefix,
        metainfo_config=metainfo_config,
        bbox_file=bbox_file,
        work_dir=work_dir,
    )

    if show_dir:
        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.out_dir = show_dir

    runner = Runner.from_cfg(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)

    class _SaveMetrics(Hook):
        def after_test_epoch(self, _, metrics=None):
            if metrics:
                # Convert numpy scalars to plain Python floats so the YAML
                # output is human-readable rather than binary-encoded blobs.
                clean = {k: float(v) for k, v in metrics.items()}
                mmengine.dump(clean, str(out))
                print(f"\n[results] Metrics written to: {out}")

    runner.register_hook(_SaveMetrics(), "LOWEST")

    eval_mode = f"detector boxes ({bbox_file})" if bbox_file else "GT boxes (oracle)"
    try:
        config_rel = config_path.relative_to(mmpose_dir.parent)
    except ValueError:
        config_rel = config_path
    try:
        ckpt_rel = checkpoint.relative_to(data_root.parent)
    except ValueError:
        ckpt_rel = checkpoint

    print(f"[config]     {config_rel}")
    print(f"[checkpoint] {ckpt_rel}")
    print(f"[eval mode]  {eval_mode}")
    print(f"[data_root]  {data_root}")
    print(f"[ann_file]   {ann_file}")
    print(f"[img_prefix] {img_prefix}")
    print(f"[work_dir]   {work_dir}")
    print(f"[out]        {out}\n")

    runner.test()
