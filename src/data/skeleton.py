"""CrowdPose joint metadata and edge connectivity."""

from __future__ import annotations

import torch
from torch import Tensor


# CrowdPose skeleton: 14 keypoints (0-indexed)
CROWDPOSE_JOINT_NAMES: dict[str, int] = {
    "left_shoulder":  0,
    "right_shoulder": 1,
    "left_elbow":     2,
    "right_elbow":    3,
    "left_wrist":     4,
    "right_wrist":    5,
    "left_hip":       6,
    "right_hip":      7,
    "left_knee":      8,
    "right_knee":     9,
    "left_ankle":     10,
    "right_ankle":    11,
    "head":           12,
    "neck":           13,
}

CROWDPOSE_EDGE_LIST: list[tuple[int, int]] = [
    (13, 0), (0, 2), (2, 4),
    (13, 1), (1, 3), (3, 5),
    (13, 12),
    (13, 6), (6, 8), (8, 10),
    (13, 7), (7, 9), (9, 11),
    (6, 7),
]


def make_edge_index(
    edge_list: list[tuple[int, int]],
    make_undirected: bool = True,
    add_self_loops: bool = False,
    num_nodes: int | None = None,
) -> Tensor:
    """Convert ``[(src, dst), ...]`` to ``LongTensor[2, E]``.

    Args:
        edge_list: List of directed edges.
        make_undirected: Duplicate every edge in both directions.
        add_self_loops: Append (i, i) for i in range(num_nodes).
        num_nodes: Required when ``add_self_loops=True``.
    """
    if make_undirected:
        edge_list = edge_list + [(dst, src) for src, dst in edge_list]
    if add_self_loops:
        if num_nodes is None:
            raise ValueError("num_nodes must be provided when add_self_loops=True")
        edge_list = edge_list + [(i, i) for i in range(num_nodes)]
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


CANONICAL_EDGE_INDEX: Tensor = make_edge_index(
    CROWDPOSE_EDGE_LIST, add_self_loops=True, num_nodes=14
)
