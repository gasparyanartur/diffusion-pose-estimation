"""Pose graph data structures and helpers.

Builds a block-diagonal PyG graph from a batch of B images, each containing N
people with K keypoints. The flattened node index is ``b*N*K + n*K + k``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from src.data.skeleton import CANONICAL_EDGE_INDEX


def create_graph_from_keypoints(
    edge_index: Tensor,
    keypoints: Tensor,
) -> dict[str, Tensor]:
    """Pack B pose graphs into one block-diagonal PyG graph.

    Args:
        edge_index: ``LongTensor[2, E]`` — per-keypoint edges (indices in [0, K)).
        keypoints:  ``FloatTensor[B, N, K, 2]``.

    Returns:
        Dict with keys: ``edge_index``, ``node_pos``, ``batch``, ``batch_idx``,
        ``pose_idx``, ``kpt_type_idx``, ``node_idx``.
    """
    B, N, K, _ = keypoints.shape
    device = keypoints.device

    batch_offset = torch.arange(B, device=device) * N * K
    pose_offset  = torch.arange(N, device=device) * K
    kpt_offset   = torch.arange(K, device=device)

    node_idx = (
        batch_offset.view(B, 1, 1)
        + pose_offset.view(1, N, 1)
        + kpt_offset.view(1, 1, K)
    )  # (B, N, K)

    batch_idx    = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, K)
    pose_idx     = torch.arange(N, device=device).view(1, N, 1).expand(B, N, K)
    kpt_type_idx = kpt_offset             .view(1, 1, K).expand(B, N, K)

    global_edge_from = node_idx[..., edge_index[0]].reshape(-1)
    global_edge_to   = node_idx[..., edge_index[1]].reshape(-1)
    flat_edge_index  = torch.stack([global_edge_from, global_edge_to], dim=0)

    batch_vec = torch.arange(B, device=device).repeat_interleave(N * K)
    node_pos = keypoints.reshape(B * N * K, 2)

    return {
        "edge_index":   flat_edge_index,
        "node_pos":     node_pos,
        "batch":        batch_vec,
        "batch_idx":    batch_idx,
        "pose_idx":     pose_idx,
        "kpt_type_idx": kpt_type_idx,
        "node_idx":     node_idx,
    }


@dataclass
class PoseSample:
    """Raw data from the DataLoader, owned before canonical graph construction."""

    keypoints: Tensor       # (B, N, K, 2) pixel-space
    visibility: Tensor      # (B, N, K)    0=absent 1=occluded 2=visible
    gt_keypoints: Tensor    # (B, N, K, 2) clean original

    image_id: list[int] = field(default_factory=list)
    img_path: list[str] = field(default_factory=list)
    image: list[Tensor] | None = None

    @property
    def B(self) -> int:
        return self.keypoints.shape[0]

    @property
    def N(self) -> int:
        return self.keypoints.shape[1]


@dataclass
class PoseGraphBatch:
    """Canonical pose graph batch derived from a :class:`PoseSample`.

    ``node_pos`` is in canonical space ``pixel / [W, H] * 2 - 1`` so in-frame
    joints lie in ``(-1, 1)``.
    """

    node_pos: Tensor                # (B*N*K, 2)
    edge_index: Tensor              # (2, B*N*E)
    node_batch: Tensor              # (B*N*K,)
    batch_idx: Tensor               # (B, N, K)
    pose_idx: Tensor                # (B, N, K)
    kpt_type_idx: Tensor            # (B, N, K)
    node_idx: Tensor                # (B, N, K)
    canonical_visibility: Tensor    # (B, N, K)
    img_wh: Tensor                  # (B, 2)  [W, H] in pixels

    @property
    def B(self) -> int:
        return self.img_wh.shape[0]

    @property
    def N(self) -> int:
        return self.batch_idx.shape[1]

    @property
    def K(self) -> int:
        return self.canonical_visibility.shape[2]

    @property
    def num_nodes(self) -> int:
        return self.B * self.N * self.K

    @property
    def flat_kpt_type_idx(self) -> Tensor:
        return self.kpt_type_idx.reshape(-1)


def prepare_sample(batch: dict) -> PoseSample:
    """Build a :class:`PoseSample` from a raw DataLoader batch dict."""
    keypoints = batch["keypoints"]
    visibility = batch["visibility"]
    image_id_raw = batch.get("image_id", [])
    image_id = [int(i) for i in image_id_raw] if len(image_id_raw) else []
    return PoseSample(
        keypoints=keypoints,
        visibility=visibility,
        gt_keypoints=keypoints,
        image_id=image_id,
        img_path=list(batch.get("img_path", [])),
        image=batch.get("image"),
    )


def prepare_pose_graph_batch(
    sample: PoseSample,
    edge_index: Tensor | None = None,
    img_size: Tensor | None = None,
) -> PoseGraphBatch:
    """Build a canonical pose graph from a :class:`PoseSample`.

    Normalises keypoints to ``(-1, 1)`` via ``pixel / [W, H] * 2 - 1``.
    Inverse: ``pixel = (canonical + 1) / 2 * [W, H]``.
    """
    keypoints = sample.keypoints
    visibility = sample.visibility
    B = keypoints.shape[0]

    if edge_index is None:
        edge_index = CANONICAL_EDGE_INDEX
    ei = edge_index.to(keypoints.device)

    if img_size is not None:
        img_wh = img_size.to(keypoints.device)
    else:
        img_wh = torch.full((B, 2), 800.0, device=keypoints.device)

    canonical_kpts = keypoints / img_wh.view(B, 1, 1, 2) * 2 - 1

    graph = create_graph_from_keypoints(ei, canonical_kpts)

    return PoseGraphBatch(
        node_pos=graph["node_pos"],
        edge_index=graph["edge_index"],
        node_batch=graph["batch"],
        batch_idx=graph["batch_idx"],
        pose_idx=graph["pose_idx"],
        kpt_type_idx=graph["kpt_type_idx"],
        node_idx=graph["node_idx"],
        canonical_visibility=visibility,
        img_wh=img_wh,
    )
