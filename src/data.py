"""Dataset, sampler, collation, and graph utilities for pose-estimation predictions."""

from __future__ import annotations

import json
import pickle
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.io import read_image as tv_read_image
from torch.utils.data import Dataset


@lru_cache(maxsize=8192)
def read_image_size(path: str) -> tuple[int, int]:
    """Return (height, width) for an image file. Results are cached by path."""
    with Image.open(path) as img:
        return img.height, img.width


def read_image_pil(path: str) -> Image.Image:
    """Load an image as an RGB PIL Image."""
    with Image.open(path) as img:
        return img.convert("RGB")


def read_image(path: str) -> torch.Tensor:
    """Load an image from disk as a uint8 RGB tensor of shape (3, H, W)."""
    return tv_read_image(path).to(torch.uint8)


def load_gt_annotations(
    ann_path: Path | str,
) -> dict[int, list[dict[str, Any]]]:
    """Load a COCO-style annotation JSON. Returns {image_id: [ann_dict, ...]}."""
    with open(ann_path) as fh:
        data = json.load(fh)

    gt: dict[int, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        gt.setdefault(ann["image_id"], []).append(ann)
    return gt


def _collate_with_images(batch: list[dict]) -> dict:
    """Like default_collate but keeps 'image' as a list (images have variable spatial size)."""
    import torch.utils.data
    images = [s["image"] for s in batch] if "image" in batch[0] else None
    other = [{k: v for k, v in s.items() if k != "image"} for s in batch]
    result = torch.utils.data.default_collate(other)
    if images is not None:
        result["image"] = images
    return result


class PoseDataset(Dataset):
    """One sample per image; N people, K=14 keypoints each.
    Pairs with SceneGraphSampler so every batch has uniform N.

    __getitem__ returns:
        image_id    int
        img_path    str
        keypoints   Tensor[N, 14, 2]  – raw (x, y) CrowdPose joints
        visibility  Tensor[N, 14]     – 0=absent 1=occluded 2=visible
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        num_keypoints: int = 14,
        load_images: bool = False,
    ) -> None:
        self.samples = samples
        self.num_keypoints = num_keypoints
        self._load_images = load_images

    @classmethod
    def from_path(
        cls,
        ann_path: Path | str,
        img_root: Path | str | None = None,
        num_keypoints: int = 14,
    ) -> "PoseDataset":
        """Load from a COCO-style annotation JSON.

        img_root: root directory prepended to each image filename from the JSON.
        Images where every annotation has no labeled keypoints are excluded.
        """
        with open(ann_path) as fh:
            data = json.load(fh)

        img_root_path = Path(img_root) if img_root is not None else None
        image_paths: dict[int, str] = {
            img["id"]: (
                str(img_root_path / img["file_name"])
                if img_root_path is not None
                else img.get("file_name", "")
            )
            for img in data.get("images", [])
        }

        # Group annotations by image_id, skipping unlabeled people
        by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            if ann.get("num_keypoints", 0) > 0:
                by_image[ann["image_id"]].append(ann)

        samples = []
        for image_id, anns in by_image.items():
            kpts = np.array(
                [a["keypoints"] for a in anns], dtype=np.float32
            ).reshape(len(anns), num_keypoints, 3)  # (N, K, 3)
            samples.append({
                "image_id":   image_id,
                "img_path":   image_paths.get(image_id, ""),
                "keypoints":  torch.from_numpy(kpts[:, :, :2]),  # (N, K, 2)
                "visibility": torch.from_numpy(kpts[:, :, 2]),   # (N, K)
            })

        return cls(samples=samples, num_keypoints=num_keypoints)

    def create_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        load_images: bool = False,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> "torch.utils.data.DataLoader":
        """DataLoader backed by SceneGraphSampler (homogeneous N per batch).

        Args:
            load_images: If True, each worker loads the full image tensor in
                __getitem__ and stores it in batch["image"] as a list.  This
                moves disk I/O off the GPU-process training step and into the
                CPU worker threads, overlapping with GPU compute.
            pin_memory: Use pinned (page-locked) memory for faster host→GPU
                transfers.  Recommended when training on a GPU.
        """
        import torch.utils.data
        self._load_images = load_images
        sampler = SceneGraphSampler(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return torch.utils.data.DataLoader(
            self, batch_sampler=sampler,
            collate_fn=_collate_with_images,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        if self._load_images:
            sample = dict(sample)  # shallow copy to avoid mutating the cache
            sample["image"] = read_image(sample["img_path"]).float().div(255.0)
        return sample


class PredictionDataset(PoseDataset):
    """PoseDataset extended with model predictions and optional GT annotations."""

    def __init__(
        self,
        model_name: str,
        architecture_type: Literal["top-down", "bottom-up"],
        num_keypoints: int,
        predictions: list[dict[str, Any]],
        gt_by_image_id: dict[int, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.architecture_type = architecture_type
        self.num_keypoints = num_keypoints
        self.predictions = predictions
        self.has_bboxes: bool = bool(predictions) and "bboxes" in predictions[0]
        self._gt_by_image_id = gt_by_image_id
        self.has_gt: bool = gt_by_image_id is not None

    @classmethod
    def from_path(cls, path: Path | str, ann_path: Path | str | None = None, **kwargs: Any) -> "PredictionDataset":
        """Load from a pickle file with keys: model_name, architecture_type, num_keypoints, predictions."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)

        kwargs.update({
            "model_name": data["model_name"],
            "architecture_type": data["architecture_type"],
            "num_keypoints": data["num_keypoints"],
            "predictions": data["predictions"],
        })
        if ann_path is not None:
            kwargs["gt_by_image_id"] = load_gt_annotations(ann_path)

        return cls(**kwargs)

    def create_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> "torch.utils.data.DataLoader":
        """DataLoader backed by SceneGraphSampler (homogeneous N per batch)."""
        import torch.utils.data

        sampler = SceneGraphSampler(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            group_key="pred_keypoints",
        )
        return torch.utils.data.DataLoader(
            self,
            batch_sampler=sampler,
            collate_fn=torch.utils.data.default_collate,
            num_workers=num_workers,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.predictions)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pred = self.predictions[idx]
        sample: dict[str, Any] = {
            "image_id":             pred["image_id"],
            "img_path":             pred.get("img_path", ""),
            "pred_keypoints":       torch.tensor(pred["keypoints"],       dtype=torch.float32),  # (N, K, 2)
            "pred_keypoint_scores": torch.tensor(pred["keypoint_scores"], dtype=torch.float32),  # (N, K)
        }
        if self.has_bboxes:
            sample["bboxes"]      = torch.tensor(pred["bboxes"],       dtype=torch.float32)  # (N, 4)
            sample["bbox_scores"] = torch.tensor(pred["bbox_scores"],  dtype=torch.float32)  # (N,)
        return sample

    def get_gt(self, image_id: int) -> dict[str, Any] | None:
        """Return GT keypoints and visibility for an image, or None if no annotations loaded."""
        if not self.has_gt or self._gt_by_image_id is None:
            return None
        gt_anns = self._gt_by_image_id.get(image_id, [])
        if not gt_anns:
            return {
                "keypoints":  torch.zeros(0, self.num_keypoints, 2, dtype=torch.float32),
                "visibility": torch.zeros(0, self.num_keypoints,    dtype=torch.float32),
            }
        kpts_flat = np.array([a["keypoints"] for a in gt_anns], dtype=np.float32)
        kpts = kpts_flat.reshape(len(gt_anns), self.num_keypoints, 3)
        return {
            "keypoints":  torch.from_numpy(kpts[:, :, :2]),  # (N_gt, K, 2)
            "visibility": torch.from_numpy(kpts[:, :, 2]),   # (N_gt, K)
        }


class SceneGraphSampler:
    """Groups dataset indices by number of people per image so every batch has uniform N."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        group_key: str = "keypoints",
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.group_key = group_key

        # Group indices by number of people (first dim of group_key tensor)
        self.groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):  # type: ignore[arg-type]
            sample = dataset[idx]
            n_people: int = sample[group_key].shape[0]
            self.groups[n_people].append(idx)

        # Pre-compute the full batch list once so __iter__ is a trivial walk.
        #
        # Non-shuffled: batches are ordered by ascending group size, indices
        #   within each group are in dataset order.
        # Shuffled: indices within each group are shuffled first, then the
        #   resulting batch list is shuffled globally so batches from different
        #   groups are interleaved randomly.
        all_batches: list[list[int]] = []
        for n_people in sorted(self.groups):
            indices = list(self.groups[n_people])
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        self._batches: list[list[int]] = all_batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


# CrowdPose skeleton: 14 keypoints (0-indexed)
# 0=left_shoulder  1=right_shoulder  2=left_elbow   3=right_elbow
# 4=left_wrist     5=right_wrist     6=left_hip     7=right_hip
# 8=left_knee      9=right_knee      10=left_ankle  11=right_ankle
# 12=head          13=neck
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
    (13, 0), (0, 2), (2, 4),   # left arm:   neck → shoulder → elbow → wrist
    (13, 1), (1, 3), (3, 5),   # right arm:  neck → shoulder → elbow → wrist
    (13, 12),                  # neck → head
    (13, 6), (6, 8), (8, 10),  # left leg:   neck → hip → knee → ankle
    (13, 7), (7, 9), (9, 11),  # right leg:  neck → hip → knee → ankle
    (6, 7),                    # hips
]

# Canonical pose graph: CrowdPose + virtual root node at index 0 (all original indices +1).
# root = mean(left_hip, right_hip, neck); connected to those three joints.
# 0=root  1=left_shoulder  2=right_shoulder  3=left_elbow   4=right_elbow
# 5=left_wrist  6=right_wrist  7=left_hip  8=right_hip  9=left_knee
# 10=right_knee  11=left_ankle  12=right_ankle  13=head  14=neck

CANONICAL_POSE_GRAPH_JOINT_NAMES: dict[str, int] = {
    "root":           0,   # virtual — mean of left_hip, right_hip, neck
    "left_shoulder":  1,
    "right_shoulder": 2,
    "left_elbow":     3,
    "right_elbow":    4,
    "left_wrist":     5,
    "right_wrist":    6,
    "left_hip":       7,
    "right_hip":      8,
    "left_knee":      9,
    "right_knee":     10,
    "left_ankle":     11,
    "right_ankle":    12,
    "head":           13,
    "neck":           14,
}

CANONICAL_POSE_GRAPH_EDGE_LIST: list[tuple[int, int]] = (
    # All original CrowdPose edges, indices shifted by +1
    [(u + 1, v + 1) for u, v in CROWDPOSE_EDGE_LIST]
    + [
        # Root connects to left_hip, right_hip, and neck
        (CANONICAL_POSE_GRAPH_JOINT_NAMES["root"], CANONICAL_POSE_GRAPH_JOINT_NAMES["left_hip"]),
        (CANONICAL_POSE_GRAPH_JOINT_NAMES["root"], CANONICAL_POSE_GRAPH_JOINT_NAMES["right_hip"]),
        (CANONICAL_POSE_GRAPH_JOINT_NAMES["root"], CANONICAL_POSE_GRAPH_JOINT_NAMES["neck"]),
    ]
)


def add_canonical_root_node(keypoints: Tensor) -> Tensor:
    """Prepend a virtual root node (mean of left_hip, right_hip, neck) to a CrowdPose tensor.

    (..., 14, 2) -> (..., 15, 2)
    """
    j = CROWDPOSE_JOINT_NAMES
    root_pos = (
        keypoints[..., j["left_hip"], :]
        + keypoints[..., j["right_hip"], :]
        + keypoints[..., j["neck"], :]
    ) / 3.0                          # (..., 2)
    root_pos = root_pos.unsqueeze(-2)  # (..., 1, 2)
    return torch.cat([root_pos, keypoints], dim=-2)  # (..., 15, 2)


def make_edge_index(
    edge_list: list[tuple[int, int]],
    make_undirected: bool = True,
) -> Tensor:
    """Convert [(src, dst), ...] to LongTensor[2, E]. Doubles edges if make_undirected=True."""
    if make_undirected:
        edge_list = edge_list + [(dst, src) for src, dst in edge_list]
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def create_graph_from_keypoints(
    edge_index: Tensor,
    keypoints: Tensor,
) -> dict[str, Tensor]:
    """Pack B pose graphs into a single block-diagonal PyG graph.

    node_idx[b, n, k] = b*N*K + n*K + k

    Args:
        edge_index  LongTensor[2, E]   – per-keypoint edges (indices in [0, K))
        keypoints   FloatTensor[B, N, K, 2]

    Returns:
        edge_index    (2, B*N*E)   – global block-diagonal edge index
        node_pos      (B*N*K, 2)   – flat node positions
        batch         (B*N*K,)     – PyG batch vector
        batch_idx     (B, N, K)    – which sample each node belongs to
        pose_idx      (B, N, K)    – which pose (person) each node belongs to (0..N-1)
        kpt_type_idx  (B, N, K)    – keypoint type per node (0..K-1); use this for type embeddings
        node_idx      (B, N, K)    – global flat node index
    """
    B, N, K, _ = keypoints.shape
    device = keypoints.device

    batch_offset = torch.arange(B, device=device) * N * K  # (B,)
    pose_offset  = torch.arange(N, device=device) * K      # (N,)
    kpt_offset   = torch.arange(K, device=device)          # (K,)

    # node_idx[b, n, k] = b*N*K + n*K + k
    node_idx = (
        batch_offset.view(B, 1, 1)
        + pose_offset.view(1, N, 1)
        + kpt_offset.view(1, 1, K)
    )  # (B, N, K)

    # Structured index tensors for masked selection
    batch_idx    = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, K)  # (B, N, K)
    pose_idx     = torch.arange(N, device=device).view(1, N, 1).expand(B, N, K)  # (B, N, K)
    kpt_type_idx = kpt_offset             .view(1, 1, K).expand(B, N, K)  # (B, N, K) values in [0, K-1]

    # Build flat block-diagonal edge index
    global_edge_from = node_idx[..., edge_index[0]].reshape(-1)  # (B*N*E,)
    global_edge_to   = node_idx[..., edge_index[1]].reshape(-1)  # (B*N*E,)
    flat_edge_index  = torch.stack([global_edge_from, global_edge_to], dim=0)  # (2, B*N*E)

    # PyG batch vector
    batch_vec = torch.arange(B, device=device).repeat_interleave(N * K)  # (B*N*K,)

    node_pos = keypoints.reshape(B * N * K, 2)  # (B*N*K, 2)

    return {
        "edge_index":   flat_edge_index,  # (2, B*N*E)
        "node_pos":     node_pos,         # (B*N*K, 2)
        "batch":        batch_vec,        # (B*N*K,)
        "batch_idx":    batch_idx,        # (B, N, K)
        "pose_idx":     pose_idx,         # (B, N, K)
        "kpt_type_idx": kpt_type_idx,     # (B, N, K)
        "node_idx":     node_idx,         # (B, N, K)
    }


def prepare_pose_graph_batch(
    batch: dict,
    edge_index: Tensor | None = None,
    img_size: Tensor | None = None,
) -> dict:
    """Build a canonical pose graph batch from raw CrowdPose keypoints.

    Reads ``batch["keypoints"]`` and ``batch["visibility"]``, computes the
    canonical graph representation, and updates *batch* in-place with all
    graph fields.  Returns *batch* for convenience.

    Computes the root per (batch, person) using visible anchor joints
    (left_hip, right_hip, neck); falls back to centroid of all visible joints.
    Keypoints are then made root-relative and normalised by max(W, H)
    (or 800 if img_size is not provided).

    Args:
        batch       dict with at least:
                      keypoints   FloatTensor[B, N, 14, 2]
                      visibility  FloatTensor[B, N, 14]  – 0=absent 1=occluded 2=visible
        edge_index  LongTensor[2, E] | None  – defaults to CANONICAL_POSE_GRAPH_EDGE_LIST
        img_size    FloatTensor[B, 2] | None – (W, H) per image for normalisation

    Returns: *batch* updated in-place with the output of create_graph_from_keypoints, plus:
        visibility  FloatTensor[B, N, 15]  – canonical visibility with root prepended
        node_pos coordinates are root-relative and divided by max(W, H)
    """
    keypoints  = batch["keypoints"]
    visibility = batch["visibility"]
    if edge_index is None:
        edge_index = make_edge_index(CANONICAL_POSE_GRAPH_EDGE_LIST)
    ei = edge_index.to(keypoints.device)

    j = CROWDPOSE_JOINT_NAMES
    anchor_idx = torch.tensor(
        [j["left_hip"], j["right_hip"], j["neck"]], device=keypoints.device
    )  # (3,)

    anchor_vis  = visibility[:, :, anchor_idx]          # (B, N, 3)
    anchor_mask = anchor_vis > 0                        # (B, N, 3)
    has_anchor  = anchor_mask.any(dim=2, keepdim=True)  # (B, N, 1)

    # Mean of visible anchors; fallback to centroid of all visible joints
    anchor_pos = keypoints[:, :, anchor_idx, :]                          # (B, N, 3, 2)
    anchor_w   = anchor_mask.float().unsqueeze(-1)                       # (B, N, 3, 1)
    root_from_anchors = (anchor_pos * anchor_w).sum(dim=2) / anchor_w.sum(dim=2).clamp(min=1)  # (B, N, 2)

    all_mask      = (visibility > 0).float().unsqueeze(-1)               # (B, N, 14, 1)
    root_from_all = (keypoints * all_mask).sum(dim=2) / all_mask.sum(dim=2).clamp(min=1)  # (B, N, 2)

    root_pos = torch.where(has_anchor, root_from_anchors, root_from_all) # (B, N, 2)

    # Root visibility: min of visible anchors, or 0 if none
    root_vis = torch.where(
        has_anchor.squeeze(2),                                            # (B, N)
        anchor_vis.masked_fill(~anchor_mask, 2.0).min(dim=2).values,     # (B, N)
        torch.zeros(keypoints.shape[:2], device=keypoints.device),       # (B, N)
    )

    canonical_kpts = torch.cat([root_pos.unsqueeze(2), keypoints], dim=2)  # (B, N, 15, 2)
    canonical_vis  = torch.cat([root_vis.unsqueeze(2), visibility], dim=2) # (B, N, 15)

    # Make all keypoints relative to the root node (root becomes [0, 0])
    root = canonical_kpts[:, :, 0:1, :]   # (B, N, 1, 2)
    canonical_kpts = canonical_kpts - root # (B, N, 15, 2)

    # Normalise by max(W, H) per image so coordinates are in [-1, 1] roughly
    B = keypoints.shape[0]
    if img_size is not None:
        scale = img_size.max(dim=1).values.to(keypoints.device)  # (B,)
    else:
        scale = torch.full((B,), 800.0, device=keypoints.device)
    canonical_kpts = canonical_kpts / scale.view(B, 1, 1, 1)

    graph = create_graph_from_keypoints(ei, canonical_kpts)  # (B, N, 15, 2)
    graph["canonical_visibility"] = canonical_vis             # (B, N, 15)
    graph["root_pos"]             = root_pos                  # (B, N, 2)  pixel-space root anchor
    graph["scale"]                = scale                     # (B,)        normalisation divisor
    batch.update(graph)
    return batch
