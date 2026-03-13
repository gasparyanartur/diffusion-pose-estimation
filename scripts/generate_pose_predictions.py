from pathlib import Path
import sys
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mmengine.config import Config
from mmengine.runner import Runner
from src.configs_train import REGISTRY

def generate_predictions(config):

    cfg = Config.fromfile(config["config_path"])

    cfg.load_from = str(config["checkpoint"])
    cfg.work_dir = "./posefix_predictions"

    data_root = config["data_root"]
    ann_file = config["ann_file"]
    img_prefix = config["img_prefix"]
    metainfo_config = config["metainfo_config"]

    metainfo = dict(from_file=str(metainfo_config))

    for dl in (cfg.test_dataloader, cfg.val_dataloader):
        dl.dataset.data_root = str(data_root) + "/"
        dl.dataset.ann_file = ann_file
        dl.dataset.data_prefix = dict(img=img_prefix)
        dl.dataset.metainfo = metainfo

    cfg.test_evaluator = dict(
        type="CocoMetric",
        ann_file=str(config["data_root"] / config["ann_file"]),
        format_only=False,
        outfile_prefix="posefix_train_predictions"
    )

    runner = Runner.from_cfg(cfg)
    runner.test()

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



if __name__ == "__main__":
    args = parse_args()

    if args.list:
        print("Registered configurations:")
        for name in REGISTRY:
            print(f"  {name}")
        

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


    generate_predictions(cfg)