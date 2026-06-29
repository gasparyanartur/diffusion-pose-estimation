"""Tests for CrowdPose dataset registration with Detectron2.

These tests require Detectron2 to be installed.  They are skipped otherwise
so the test suite continues to pass in environments without Detectron2.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CROWDPOSE_ROOT = Path("data/crowdpose")
DATA_AVAILABLE = (CROWDPOSE_ROOT / "annotations" / "crowdpose_val.json").exists()

detectron2 = pytest.importorskip("detectron2", reason="Detectron2 not installed")


# Re-import after the skip guard so the module is only loaded when available.
from detectron2.data import DatasetCatalog, MetadataCatalog  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "external" / "diffusiondet"))
from datasets.register_crowdpose import register_all_crowdpose  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_catalog():
    """Remove any previously registered crowdpose datasets between tests."""
    splits = ["crowdpose_train", "crowdpose_val", "crowdpose_test"]
    for name in splits:
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
    yield
    for name in splits:
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)


def test_register_all_crowdpose_runs():
    register_all_crowdpose(root=str(CROWDPOSE_ROOT))
    assert "crowdpose_train" in DatasetCatalog
    assert "crowdpose_val" in DatasetCatalog
    assert "crowdpose_test" in DatasetCatalog


def test_metadata_has_person_class():
    register_all_crowdpose(root=str(CROWDPOSE_ROOT))
    meta = MetadataCatalog.get("crowdpose_val")
    assert meta.thing_classes == ["person"]


@pytest.mark.skipif(not DATA_AVAILABLE, reason="CrowdPose data not present")
def test_dataset_entries_have_required_keys():
    register_all_crowdpose(root=str(CROWDPOSE_ROOT))
    entries = DatasetCatalog.get("crowdpose_val")
    assert len(entries) > 0
    entry = entries[0]
    for key in ("file_name", "image_id", "height", "width", "annotations"):
        assert key in entry, f"missing key: {key}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="CrowdPose data not present")
def test_annotations_have_bbox_and_category():
    register_all_crowdpose(root=str(CROWDPOSE_ROOT))
    entries = DatasetCatalog.get("crowdpose_val")
    # Find an entry that has at least one annotation.
    entry = next(e for e in entries if e["annotations"])
    ann = entry["annotations"][0]
    for key in ("bbox", "category_id", "iscrowd"):
        assert key in ann, f"annotation missing key: {key}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="CrowdPose data not present")
def test_category_id_is_zero_indexed():
    """Detectron2 remaps CrowdPose category_id=1 → class index 0."""
    register_all_crowdpose(root=str(CROWDPOSE_ROOT))
    entries = DatasetCatalog.get("crowdpose_val")
    entry = next(e for e in entries if e["annotations"])
    ann = entry["annotations"][0]
    assert ann["category_id"] == 0, (
        f"Expected 0-indexed class id, got {ann['category_id']}"
    )
