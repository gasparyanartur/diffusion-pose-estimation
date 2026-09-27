"""Tests for deterministic COCO-style CrowdPose annotation subsets."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.data.crowdpose_subset import select_annotation_subset


def _source_annotations() -> dict:
    return {
        "info": {"description": "fixture"},
        "categories": [{"id": 1, "name": "person"}],
        "images": [
            {"id": 9, "file_name": "9.jpg"},
            {"id": 3, "file_name": "3.jpg"},
            {"id": 7, "file_name": "7.jpg"},
            {"id": 1, "file_name": "1.jpg"},
        ],
        "annotations": [
            {"id": 20, "image_id": 7, "category_id": 1},
            {"id": 10, "image_id": 3, "category_id": 1},
            {"id": 30, "image_id": 7, "category_id": 1},
        ],
    }


def test_select_annotation_subset_is_seeded_sorted_and_annotation_closed() -> None:
    source = _source_annotations()

    first = select_annotation_subset(source, image_count=2, seed=42)
    second = select_annotation_subset(source, image_count=2, seed=42)

    assert first == second
    selected_ids = [image["id"] for image in first["images"]]
    assert selected_ids == sorted(selected_ids)
    assert len(selected_ids) == 2
    assert all(annotation["image_id"] in selected_ids for annotation in first["annotations"])
    assert first["info"] == source["info"]
    assert first["categories"] == source["categories"]
    assert first["poseflow_subset"] == {
        "source_image_count": 4,
        "selected_image_count": 2,
        "seed": 42,
    }


@pytest.mark.parametrize(
    ("image_count", "message"),
    [
        (0, "positive"),
        (5, "at most"),
    ],
)
def test_select_annotation_subset_rejects_invalid_counts(image_count: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        select_annotation_subset(_source_annotations(), image_count=image_count, seed=42)


def test_select_annotation_subset_rejects_duplicate_images_and_dangling_annotations() -> None:
    duplicated = _source_annotations()
    duplicated["images"].append({"id": 3, "file_name": "duplicate.jpg"})
    with pytest.raises(ValueError, match="duplicate"):
        select_annotation_subset(duplicated, image_count=2, seed=42)

    dangling = _source_annotations()
    dangling["annotations"].append({"id": 99, "image_id": 99, "category_id": 1})
    with pytest.raises(ValueError, match="unknown image"):
        select_annotation_subset(dangling, image_count=2, seed=42)


def test_poseflow_v5_test_subset_preset_uses_the_fixed_monitoring_annotation() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="benchmark_poseflow_v5_testval_1000")

    assert config.data.train_ann == "crowdpose/annotations/crowdpose_trainval.json"
    assert config.data.val_ann == "crowdpose/annotations/crowdpose_test_val_seed42_1000.json"
    assert config.data.test_ann == "crowdpose/annotations/crowdpose_test.json"
    assert config.run_name == "poseflow_v5_testval_1000_training"
    assert config.pipeline.val_ap_every_n_epochs == 5
