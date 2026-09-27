#!/usr/bin/env python3
"""Create a deterministic annotation-closed CrowdPose subset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.crowdpose_subset import write_annotation_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_annotation_subset(
        args.source,
        args.output,
        image_count=args.image_count,
        seed=args.seed,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
