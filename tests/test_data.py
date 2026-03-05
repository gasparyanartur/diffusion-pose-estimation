"""Unit tests for src.data – core functionality and key edge cases."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import GroupedBatchSampler, PredictionDataset, simple_collate

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

NUM_KEYPOINTS = 14


def _make_predictions(
    people_counts: list[int],
    include_bboxes: bool = True,
) -> list[dict[str, Any]]:
    """Return synthetic prediction records with the given per-image people counts."""
    records = []
    for i, n in enumerate(people_counts):
        rng = np.random.default_rng(seed=i)
        record: dict[str, Any] = {
            "image_id": i,
            "img_path": f"/fake/img_{i}.jpg",
            "keypoints": rng.random((n, NUM_KEYPOINTS, 2)).astype(np.float32) * 640,
            "keypoint_scores": rng.random((n, NUM_KEYPOINTS)).astype(np.float32),
        }
        if include_bboxes:
            record["bboxes"] = rng.random((n, 4)).astype(np.float32) * 640
            record["bbox_scores"] = rng.random((n,)).astype(np.float32)
        records.append(record)
    return records


@pytest.fixture
def dataset() -> PredictionDataset:
    """12 images, 4 each for 1 / 2 / 3 people, with bboxes."""
    counts = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    return PredictionDataset(
        model_name="test-model",
        architecture_type="top-down",
        num_keypoints=NUM_KEYPOINTS,
        predictions=_make_predictions(counts),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dataset_getitem_shapes_and_keys(dataset):
    """__getitem__ returns correct tensor shapes and all expected keys."""
    sample = dataset[4]  # an image with 2 people
    assert set(sample.keys()) == {"image_id", "img_path", "keypoints", "keypoint_scores", "bboxes", "bbox_scores"}
    n = sample["keypoints"].shape[0]
    assert sample["keypoints"].shape == (n, NUM_KEYPOINTS, 2)
    assert sample["keypoint_scores"].shape == (n, NUM_KEYPOINTS)
    assert sample["bboxes"].shape == (n, 4)
    assert sample["bbox_scores"].shape == (n,)
    assert sample["keypoints"].dtype == torch.float32


def test_dataset_from_path_round_trip(tmp_path, dataset):
    """from_path reconstructs an equivalent dataset from a pickle file."""
    cache = tmp_path / "preds.pkl"
    preds = _make_predictions([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    payload = {"model_name": "m", "architecture_type": "top-down", "num_keypoints": NUM_KEYPOINTS, "predictions": preds}
    cache.write_bytes(pickle.dumps(payload))

    loaded = PredictionDataset.from_path(cache)
    assert len(loaded) == len(dataset)
    assert loaded.model_name == "m"
    assert torch.allclose(loaded[0]["keypoints"], dataset[0]["keypoints"])


def test_sampler_groups_and_covers_all_indices(dataset):
    """Sampler forms one group per unique N and yields every index exactly once."""
    sampler = GroupedBatchSampler(dataset, batch_size=2, drop_last=False, shuffle=False)
    assert set(sampler.groups.keys()) == {1, 2, 3}

    seen = [idx for batch in sampler for idx in batch]
    assert sorted(seen) == list(range(len(dataset)))


def test_sampler_drop_last(dataset):
    """drop_last=True discards partial batches; drop_last=False keeps them."""
    # 4 images per group, batch_size=3 → 1 full + 1 partial per group
    full = GroupedBatchSampler(dataset, batch_size=3, drop_last=False, shuffle=False)
    dropped = GroupedBatchSampler(dataset, batch_size=3, drop_last=True, shuffle=False)
    assert len(list(full)) == 6      # 3 groups × 2 batches
    assert len(list(dropped)) == 3   # 3 groups × 1 full batch


def test_collate_output_shapes_and_raises_on_heterogeneous_n(dataset):
    """simple_collate stacks into (B, N, K, 2) and rejects mixed-N batches."""
    # Normal case: uniform N
    batch = [dataset[i] for i in range(4, 8)]  # 4 images, each with 2 people
    out = simple_collate(batch)
    assert out["keypoints"].shape == (4, 2, NUM_KEYPOINTS, 2)
    assert out["keypoint_scores"].shape == (4, 2, NUM_KEYPOINTS)
    assert out["bboxes"].shape == (4, 2, 4)

    # Edge case: mixed N should raise
    with pytest.raises(RuntimeError, match="same number of people"):
        simple_collate([dataset[0], dataset[4]])  # 1 person vs 2 people
