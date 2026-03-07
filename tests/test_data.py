"""Unit tests for src.data – core functionality and key edge cases."""

from __future__ import annotations

import pickle
import random
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import (
    CROWDPOSE_EDGE_LIST,
    GroupedBatchSampler,
    PredictionDataset,
    create_graph_from_keypoints,
    load_gt_annotations,
    make_edge_index,
    simple_collate,
)

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


def _make_gt_ann_json(image_ids: list[int], gt_counts: list[int], tmp_path) -> str:
    """Write a minimal COCO-style annotation JSON and return its path."""
    import json

    annotations = []
    ann_id = 0
    for img_id, n_gt in zip(image_ids, gt_counts):
        rng = np.random.default_rng(seed=img_id)
        for _ in range(n_gt):
            # flat [x, y, v, x, y, v, ...] for NUM_KEYPOINTS joints
            kpts_xy = (rng.random(NUM_KEYPOINTS * 2) * 640).tolist()
            vis = [2] * NUM_KEYPOINTS
            flat = []
            for k in range(NUM_KEYPOINTS):
                flat += [kpts_xy[k * 2], kpts_xy[k * 2 + 1], vis[k]]
            annotations.append({"id": ann_id, "image_id": img_id, "keypoints": flat,
                                 "num_keypoints": NUM_KEYPOINTS, "category_id": 1})
            ann_id += 1

    data = {"images": [{"id": i} for i in image_ids], "annotations": annotations}
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(data))
    return path


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


def test_sampler_shuffle_interleaves_groups(dataset):
    """Shuffled sampler interleaves batches from different groups; unshuffled does not."""
    random.seed(0)
    shuffled = GroupedBatchSampler(dataset, batch_size=2, drop_last=False, shuffle=True)
    unshuffled = GroupedBatchSampler(dataset, batch_size=2, drop_last=False, shuffle=False)

    def batch_n_values(sampler):
        """Return the sequence of N-people values for each yielded batch."""
        return [dataset[b[0]]["keypoints"].shape[0] for b in sampler]

    # Unshuffled: all N=1 batches, then all N=2, then all N=3 — strictly non-decreasing
    ns_unshuffled = batch_n_values(unshuffled)
    assert ns_unshuffled == sorted(ns_unshuffled), "Unshuffled should be group-sequential"

    # Shuffled: the N sequence should NOT be perfectly sorted (groups are interleaved)
    # With seed 0 and 12 images/3 groups this is virtually guaranteed
    ns_shuffled = batch_n_values(shuffled)
    assert ns_shuffled != sorted(ns_shuffled), "Shuffled should interleave groups"

    # But all indices must still be covered exactly once
    seen = [idx for batch in GroupedBatchSampler(dataset, batch_size=2, drop_last=False, shuffle=True) for idx in batch]
    assert sorted(seen) == list(range(len(dataset)))


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


def test_gt_keypoints_shapes_and_collate_as_list(tmp_path):
    """GT keypoints land on each sample with correct shape; collated as a list."""
    preds = _make_predictions([2, 2, 2, 2])          # 4 images, 2 predicted people each
    gt_counts = [3, 1, 2, 0]                          # GT people count differs from predicted

    ann_path = _make_gt_ann_json(list(range(4)), gt_counts, tmp_path)
    gt = load_gt_annotations(ann_path)

    ds = PredictionDataset(
        model_name="m", architecture_type="top-down",
        num_keypoints=NUM_KEYPOINTS, predictions=preds,
        gt_by_image_id=gt,
    )
    assert ds.has_gt is True

    # Per-sample shapes
    for i, n_gt in enumerate(gt_counts):
        s = ds[i]
        assert "gt_keypoints" in s and "gt_visibility" in s
        assert s["gt_keypoints"].shape == (n_gt, NUM_KEYPOINTS, 2)
        assert s["gt_visibility"].shape == (n_gt, NUM_KEYPOINTS)

    # Collated: GT stays as a list (ragged N_gt)
    batch = [ds[i] for i in range(4)]
    out = simple_collate(batch)
    assert isinstance(out["gt_keypoints"], list) and len(out["gt_keypoints"]) == 4
    assert isinstance(out["gt_visibility"], list) and len(out["gt_visibility"]) == 4
    assert out["gt_keypoints"][0].shape == (gt_counts[0], NUM_KEYPOINTS, 2)


# ---------------------------------------------------------------------------
# Graph utility tests
# ---------------------------------------------------------------------------


def test_make_edge_index_shape_and_reverse_edges():
    """Directed variant has shape (2, E); undirected doubles it and adds reverses."""
    assert make_edge_index(CROWDPOSE_EDGE_LIST, make_undirected=False).shape == (2, len(CROWDPOSE_EDGE_LIST))

    ei = make_edge_index(CROWDPOSE_EDGE_LIST, make_undirected=True)
    assert ei.shape == (2, 2 * len(CROWDPOSE_EDGE_LIST))
    edges = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert all((v, u) in edges for u, v in CROWDPOSE_EDGE_LIST)


def test_create_graph_from_keypoints():
    """Output shapes, node_idx formula, batch vector, and edge index bounds are correct."""
    B, N, K = 3, 5, NUM_KEYPOINTS
    keypoints = torch.rand(B, N, K, 2)
    ei = make_edge_index(CROWDPOSE_EDGE_LIST)
    E = ei.shape[1]
    g = create_graph_from_keypoints(ei, keypoints)

    # Shapes
    assert g["edge_index"].shape == (2, B * N * E)
    assert g["node_pos"].shape   == (B * N * K, 2)
    assert g["batch"].shape      == (B * N * K,)
    assert g["batch_idx"].shape  == (B, N, K)
    assert g["pose_idx"].shape   == (B, N, K)

    # node_idx formula: node_idx[b, n, k] == b*N*K + n*K + k
    b_t = torch.arange(B).view(B, 1, 1)
    n_t = torch.arange(N).view(1, N, 1)
    k_t = torch.arange(K).view(1, 1, K)
    assert torch.equal(g["node_idx"], b_t * N * K + n_t * K + k_t)

    # batch vector and edge bounds
    assert torch.equal(g["batch"], torch.arange(B).repeat_interleave(N * K))
    assert g["edge_index"].min() >= 0 and g["edge_index"].max() < B * N * K
