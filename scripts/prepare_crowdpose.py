#!/usr/bin/env python3
"""
Download the MMPose-formatted CrowdPose annotations (including the pre-generated
YOLOv3 detection boxes) from the OpenMMLab CDN.

The tar contains:
    crowdpose_annotations/det_for_crowd_test_0.1_0.5.json   ← detector boxes
    crowdpose_annotations/mmpose_crowdpose_train.json
    crowdpose_annotations/mmpose_crowdpose_val.json
    crowdpose_annotations/mmpose_crowdpose_test.json

All files are extracted to data/crowdpose/annotations/, skipping any that
already exist.

Usage:
    python scripts/prepare_crowdpose.py
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

from src.configs import defaults  # noqa: E402

_TAR_PREFIX = "crowdpose_annotations/"


def _progress(count: int, block: int, total: int) -> None:
    pct = min(100, int(count * block * 100 / total))
    bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
    print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)


def main() -> None:
    dest_dir = defaults.DATA_ROOT / "crowdpose" / "annotations"
    dest_dir.mkdir(parents=True, exist_ok=True)

    tar_path = dest_dir / "_crowdpose_annotations.tar"

    if not tar_path.exists():
        print(f"Downloading annotations tar (~125 MB) …\n  {defaults.CROWDPOSE_ANNOTATIONS_URL}")
        urllib.request.urlretrieve(defaults.CROWDPOSE_ANNOTATIONS_URL, tar_path, _progress)
        print()
    else:
        print(f"[skip] Tar already downloaded: {tar_path}")

    print("\nExtracting …")
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if not member.name.startswith(_TAR_PREFIX):
                continue
            rel = member.name[len(_TAR_PREFIX):]
            if not rel:
                continue
            dest = dest_dir / rel
            if dest.exists():
                print(f"  [skip] {rel}")
                continue
            member.name = rel
            tf.extract(member, path=dest_dir)
            print(f"  [ok]   {rel}")

    tar_path.unlink()
    print(f"\nDone. Files are in {dest_dir}")
    print("\nYou can now benchmark with detector boxes.")

if __name__ == "__main__":
    main()
