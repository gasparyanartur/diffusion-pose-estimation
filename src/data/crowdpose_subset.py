"""Create deterministic, annotation-closed subsets of COCO-style datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping


def select_annotation_subset(
    source: Mapping[str, Any],
    *,
    image_count: int,
    seed: int,
) -> dict[str, Any]:
    """Select a seeded image subset while retaining all matching annotations."""
    images = source.get("images")
    annotations = source.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("source must contain list-valued images and annotations")
    if image_count < 1:
        raise ValueError("image_count must be positive")
    if image_count > len(images):
        raise ValueError("image_count must be at most the source image count")

    images_by_id: dict[int, dict[str, Any]] = {}
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("id"), int):
            raise ValueError("each image must be a dictionary with an integer id")
        image_id = image["id"]
        if image_id in images_by_id:
            raise ValueError(f"duplicate image id: {image_id}")
        images_by_id[image_id] = image

    source_ids = sorted(images_by_id)
    selected_ids = sorted(random.Random(seed).sample(source_ids, image_count))
    selected_id_set = set(selected_ids)
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("each annotation must be a dictionary")
        image_id = annotation.get("image_id")
        if image_id not in images_by_id:
            raise ValueError(f"annotation refers to unknown image id: {image_id}")

    subset = {
        key: value
        for key, value in source.items()
        if key not in {"images", "annotations", "poseflow_subset"}
    }
    subset["images"] = [images_by_id[image_id] for image_id in selected_ids]
    subset["annotations"] = [
        annotation for annotation in annotations if annotation["image_id"] in selected_id_set
    ]
    subset["poseflow_subset"] = {
        "source_image_count": len(source_ids),
        "selected_image_count": image_count,
        "seed": seed,
    }
    return subset


def write_annotation_subset(
    source_path: Path,
    output_path: Path,
    *,
    image_count: int,
    seed: int,
) -> None:
    """Write one fresh deterministic annotation subset without touching its source."""
    if source_path.resolve() == output_path.resolve():
        raise ValueError("output path must differ from source path")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing subset: {output_path}")
    source = json.loads(source_path.read_text())
    subset = select_annotation_subset(source, image_count=image_count, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(subset, indent=2) + "\n")
