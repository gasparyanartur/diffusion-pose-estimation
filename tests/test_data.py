"""Unit tests for src.data – core functionality and key edge cases."""

from __future__ import annotations

import json
import pickle
import random
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import (
    CROWDPOSE_EDGE_LIST,
    SceneGraphSampler,
    PoseDataset,
    PredictionDataset,
    create_graph_from_keypoints,
    load_gt_annotations,
    make_edge_index,
    prepare_pose_graph_batch,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

NUM_KEYPOINTS = 14


def _make_pose_samples(people_counts: list[int]) -> list[dict[str, Any]]:
    """Return synthetic PoseDataset samples with the given per-image people counts."""
    samples = []
    for i, n in enumerate(people_counts):
        rng = np.random.default_rng(seed=i)
        samples.append({
            "image_id":   i,
            "img_path":   f"/fake/img_{i}.jpg",
            "keypoints":  torch.from_numpy(rng.random((n, NUM_KEYPOINTS, 2)).astype(np.float32) * 640),
            "visibility": torch.from_numpy(rng.integers(0, 3, (n, NUM_KEYPOINTS)).astype(np.float32)),
        })
    return samples


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
def dataset() -> PoseDataset:
    """12 images, 4 each for 1 / 2 / 3 people."""
    counts = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    return PoseDataset(samples=_make_pose_samples(counts))


def _make_gt_ann_json(image_ids: list[int], gt_counts: list[int], tmp_path) -> str:
    """Write a minimal COCO-style annotation JSON and return its path."""
    annotations = []
    ann_id = 0
    for img_id, n_gt in zip(image_ids, gt_counts):
        rng = np.random.default_rng(seed=img_id)
        for _ in range(n_gt):
            kpts_xy = (rng.random(NUM_KEYPOINTS * 2) * 640).tolist()
            vis = [2] * NUM_KEYPOINTS
            flat = []
            for k in range(NUM_KEYPOINTS):
                flat += [kpts_xy[k * 2], kpts_xy[k * 2 + 1], vis[k]]
            annotations.append({
                "id": ann_id, "image_id": img_id, "keypoints": flat,
                "num_keypoints": NUM_KEYPOINTS, "category_id": 1,
            })
            ann_id += 1

    data = {"images": [{"id": i} for i in image_ids], "annotations": annotations}
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# PoseDataset tests
# ---------------------------------------------------------------------------


def test_pose_dataset_getitem_shapes_and_keys(dataset):
    """__getitem__ returns correct tensor shapes and all expected keys."""
    sample = dataset[4]  # an image with 2 people
    assert set(sample.keys()) == {"image_id", "img_path", "keypoints", "visibility"}
    n = sample["keypoints"].shape[0]
    assert sample["keypoints"].shape == (n, NUM_KEYPOINTS, 2)
    assert sample["visibility"].shape == (n, NUM_KEYPOINTS)
    assert sample["keypoints"].dtype == torch.float32


def test_pose_dataset_from_path(tmp_path):
    """from_path loads keypoints and visibility from a COCO annotation JSON."""
    counts = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    ann_path = _make_gt_ann_json(list(range(12)), counts, tmp_path)

    ds = PoseDataset.from_path(ann_path)
    assert len(ds) == 12
    by_id = {ds[i]["image_id"]: ds[i] for i in range(len(ds))}
    for img_id, n in enumerate(counts):
        s = by_id[img_id]
        assert s["keypoints"].shape == (n, NUM_KEYPOINTS, 2)
        assert s["visibility"].shape == (n, NUM_KEYPOINTS)
        assert s["keypoints"].dtype == torch.float32


def test_sampler_groups_and_covers_all_indices(dataset):
    """Sampler forms one group per unique N and yields every index exactly once."""
    sampler = SceneGraphSampler(dataset, batch_size=2, drop_last=False, shuffle=False)
    assert set(sampler.groups.keys()) == {1, 2, 3}

    seen = [idx for batch in sampler for idx in batch]
    assert sorted(seen) == list(range(len(dataset)))


def test_sampler_drop_last(dataset):
    """drop_last=True discards partial batches; drop_last=False keeps them."""
    # 4 images per group, batch_size=3 → 1 full + 1 partial per group
    full    = SceneGraphSampler(dataset, batch_size=3, drop_last=False, shuffle=False)
    dropped = SceneGraphSampler(dataset, batch_size=3, drop_last=True,  shuffle=False)
    assert len(list(full))    == 6   # 3 groups × 2 batches
    assert len(list(dropped)) == 3   # 3 groups × 1 full batch


def test_sampler_shuffle_interleaves_groups(dataset):
    """Shuffled sampler interleaves batches from different groups; unshuffled does not."""
    random.seed(0)
    shuffled   = SceneGraphSampler(dataset, batch_size=2, drop_last=False, shuffle=True)
    unshuffled = SceneGraphSampler(dataset, batch_size=2, drop_last=False, shuffle=False)

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
    seen = [idx for batch in SceneGraphSampler(dataset, batch_size=2, drop_last=False, shuffle=True) for idx in batch]
    assert sorted(seen) == list(range(len(dataset)))


def test_dataloader_collate_shapes(dataset):
    """default_collate stacks uniform-N batches correctly; SceneGraphSampler ensures homogeneous N."""
    import torch.utils.data
    batch = [dataset[i] for i in range(4, 8)]  # 4 images, each with 2 people
    out = torch.utils.data.default_collate(batch)
    assert out["keypoints"].shape == (4, 2, NUM_KEYPOINTS, 2)
    assert out["visibility"].shape == (4, 2, NUM_KEYPOINTS)
    assert isinstance(out["image_id"], torch.Tensor)  # int scalars get stacked


def test_create_dataloader_yields_homogeneous_batches(dataset):
    """create_dataloader yields batches where every sample has the same N; all indices covered."""
    loader = dataset.create_dataloader(batch_size=2, shuffle=False, drop_last=False)

    all_image_ids = []
    for batch in loader:
        # Every batch must have shape (B, N, K, 2) with a consistent N across the batch
        kpts = batch["keypoints"]          # (B, N, K, 2)
        vis  = batch["visibility"]         # (B, N, K)
        B = kpts.shape[0]
        assert kpts.ndim == 4 and kpts.shape[2] == NUM_KEYPOINTS
        assert vis.shape == (B, kpts.shape[1], NUM_KEYPOINTS)
        all_image_ids.extend(batch["image_id"].tolist())

    # Every sample in the dataset must appear exactly once
    assert sorted(all_image_ids) == list(range(len(dataset)))


# ---------------------------------------------------------------------------
# PredictionDataset tests
# ---------------------------------------------------------------------------


def test_prediction_dataset_getitem():
    """__getitem__ returns pred_* keys; GT keys absent when no annotations are loaded."""
    preds = _make_predictions([2, 2])
    ds = PredictionDataset(
        model_name="m", architecture_type="top-down",
        num_keypoints=NUM_KEYPOINTS, predictions=preds,
    )
    s = ds[0]
    assert set(s.keys()) == {"image_id", "img_path", "pred_keypoints", "pred_keypoint_scores", "bboxes", "bbox_scores"}
    assert s["pred_keypoints"].shape == (2, NUM_KEYPOINTS, 2)
    assert s["pred_keypoint_scores"].shape == (2, NUM_KEYPOINTS)
    assert "keypoints" not in s and "visibility" not in s

def test_prediction_dataset_from_path_with_gt(tmp_path):
    """from_path loads predictions from pickle; GT accessible via get_gt() when ann_path given."""
    gt_counts = [3, 1, 2, 0]
    preds = _make_predictions([2, 2, 2, 2])
    payload = {
        "model_name": "resnet", "architecture_type": "top-down",
        "num_keypoints": NUM_KEYPOINTS, "predictions": preds,
    }
    pkl_path = tmp_path / "preds.pkl"
    pkl_path.write_bytes(pickle.dumps(payload))
    ann_path = _make_gt_ann_json(list(range(4)), gt_counts, tmp_path)

    ds = PredictionDataset.from_path(pkl_path, ann_path=ann_path)
    assert ds.model_name == "resnet" and ds.has_gt

    # GT is not in __getitem__ – accessed separately via get_gt()
    for i, n_gt in enumerate(gt_counts):
        s = ds[i]
        assert "keypoints" not in s and "visibility" not in s
        gt = ds.get_gt(s["image_id"])
        assert gt is not None
        assert gt["keypoints"].shape  == (n_gt, NUM_KEYPOINTS, 2)
        assert gt["visibility"].shape == (n_gt, NUM_KEYPOINTS)

    # get_gt returns None when no annotations were loaded
    ds_no_gt = PredictionDataset(
        model_name="m", architecture_type="top-down",
        num_keypoints=NUM_KEYPOINTS, predictions=preds,
    )
    assert ds_no_gt.get_gt(0) is None

    # Batches are collatable with default_collate because all pred tensors are uniform
    import torch.utils.data
    sampler = SceneGraphSampler(ds, batch_size=4, group_key="pred_keypoints")
    batch = [ds[i] for i in range(4)]
    out = torch.utils.data.default_collate(batch)
    assert out["pred_keypoints"].shape == (4, 2, NUM_KEYPOINTS, 2)


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


def test_create_graph_batch_samples_are_disjoint():
    """Node sets for different batch samples are disjoint, and edges never cross batch boundaries."""
    B, N, K = 3, 4, NUM_KEYPOINTS
    keypoints = torch.rand(B, N, K, 2)
    ei = make_edge_index(CROWDPOSE_EDGE_LIST)
    g = create_graph_from_keypoints(ei, keypoints)

    node_idx = g["node_idx"]   # (B, N, K)
    edge_index = g["edge_index"]  # (2, B*N*E)

    # Node sets per batch sample must be pairwise disjoint
    node_sets = [set(node_idx[b].reshape(-1).tolist()) for b in range(B)]
    for b1 in range(B):
        for b2 in range(b1 + 1, B):
            assert node_sets[b1].isdisjoint(node_sets[b2]), (
                f"Batch samples {b1} and {b2} share node indices"
            )

    # Every edge must connect two nodes that belong to the same batch sample
    all_nodes = node_idx.reshape(-1)          # (B*N*K,)
    node_to_batch = g["batch"]                # (B*N*K,) maps flat node index -> b
    src_batch = node_to_batch[edge_index[0]]
    dst_batch = node_to_batch[edge_index[1]]
    assert torch.all(src_batch == dst_batch), "Some edge crosses batch-sample boundaries"


def test_create_graph_poses_are_disjoint():
    """Node sets for different poses within a batch sample are disjoint, and edges never cross pose boundaries."""
    B, N, K = 3, 4, NUM_KEYPOINTS
    keypoints = torch.rand(B, N, K, 2)
    ei = make_edge_index(CROWDPOSE_EDGE_LIST)
    g = create_graph_from_keypoints(ei, keypoints)

    node_idx = g["node_idx"]   # (B, N, K)
    pose_idx  = g["pose_idx"]  # (B, N, K)
    edge_index = g["edge_index"]  # (2, B*N*E)

    # Within each batch sample, pose node sets must be pairwise disjoint
    for b in range(B):
        pose_sets = [set(node_idx[b, n, :].tolist()) for n in range(N)]
        for n1 in range(N):
            for n2 in range(n1 + 1, N):
                assert pose_sets[n1].isdisjoint(pose_sets[n2]), (
                    f"Poses {n1} and {n2} in batch sample {b} share node indices"
                )

    # Every edge must connect two nodes that belong to the same pose
    flat_node_idx = node_idx.reshape(-1)      # (B*N*K,)
    flat_pose_idx = pose_idx.reshape(-1)      # (B*N*K,)
    src_pose = flat_pose_idx[edge_index[0]]
    dst_pose = flat_pose_idx[edge_index[1]]
    assert torch.all(src_pose == dst_pose), "Some edge crosses pose boundaries"


def test_prepare_pose_graph_batch():
    """Root node is at origin; coordinates are normalised; visibility shape is correct."""
    B, N, K = 2, 3, 14
    torch.manual_seed(0)
    kpts = torch.rand(B, N, K, 2) * 640
    vis  = torch.randint(0, 3, (B, N, K)).float()

    # --- without img_size: scale factor is 100 ---
    g = prepare_pose_graph_batch({"keypoints": kpts, "visibility": vis})

    # Output shapes
    assert g["node_pos"].shape              == (B * N * 15, 2)
    assert g["canonical_visibility"].shape  == (B, N, 15)

    # Root node (index 0 per pose) must be exactly (0, 0) after root-relative shift
    node_pos = g["node_pos"].view(B, N, 15, 2)
    assert torch.all(node_pos[:, :, 0, :] == 0.0), "Root nodes should be at origin"

    # All coordinates should be in a sensible range when divided by 100
    # (raw offsets up to ~640 px → up to ~6.4)
    assert g["node_pos"].abs().max().item() < 10.0

    # --- with img_size: scale factor is max(W, H) per image ---
    img_size = torch.tensor([[640.0, 480.0], [320.0, 240.0]])
    g2 = prepare_pose_graph_batch({"keypoints": kpts, "visibility": vis}, img_size=img_size)

    # Root still at origin
    node_pos2 = g2["node_pos"].view(B, N, 15, 2)
    assert torch.all(node_pos2[:, :, 0, :] == 0.0)

    # Coordinates for image b=0 are divided by 640; offsets within frame are ≤ 1
    coords_b0 = node_pos2[0]  # (N, 15, 2)
    assert coords_b0.abs().max().item() <= 1.0 + 1e-5, (
        "Root-relative offsets within a 640-px image should be at most 1.0 when scaled by max dim"
    )
