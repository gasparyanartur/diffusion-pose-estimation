"""Image and annotation I/O helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.io import read_image as tv_read_image


@lru_cache(maxsize=32000)
def read_image_size(path: str) -> tuple[int, int]:
    """Return (height, width) for an image file. Cached by path."""
    with Image.open(path) as img:
        return img.height, img.width


def read_image_pil(path: str) -> Image.Image:
    """Load an image as an RGB PIL Image."""
    with Image.open(path) as img:
        return img.convert("RGB")


def read_image(path: str) -> torch.Tensor:
    """Load an image from disk as a uint8 RGB tensor of shape (3, H, W)."""
    return tv_read_image(path).to(torch.uint8)


def load_gt_annotations(ann_path: Path | str) -> dict[int, list[dict[str, Any]]]:
    """Load a COCO-style annotation JSON. Returns {image_id: [ann_dict, ...]}."""
    with open(ann_path) as fh:
        data = json.load(fh)

    gt: dict[int, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        gt.setdefault(ann["image_id"], []).append(ann)
    return gt
