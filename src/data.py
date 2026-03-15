"""Dataset, sampler, collation, and graph utilities for pose-estimation predictions.

NOTE: UPDATE THIS WHEN UPDATING THE CODE.

PoseDataset
-----------
GT-only dataset; one sample per image.  Images where every annotation has no
labeled keypoints are excluded.

    from src.data import PoseDataset

    ds = PoseDataset.from_path("data/crowdpose/annotations/crowdpose_train.json")
    sample = ds[0]
    # sample["image_id"]   -> int
    # sample["img_path"]   -> str
    # sample["keypoints"]  -> Tensor[N, 14, 2]   raw (x, y)
    # sample["visibility"] -> Tensor[N, 14]      0=absent 1=occluded 2=visible

    loader = ds.create_dataloader(batch_size=32, shuffle=True, drop_last=True)

PredictionDataset
-----------------
Extends PoseDataset with model predictions loaded from a .pkl cache.
GT annotations (which have a ragged number of people per image) are deliberately
not returned by __getitem__; use get_gt(image_id) instead so default_collate
can stack predictions uniformly.

    from src.data import PredictionDataset

    ds = PredictionDataset.from_path(
        "results/cache/rtmpose-m_crowdpose_predictions.pkl",
        ann_path="data/crowdpose/annotations/crowdpose_test.json",  # optional GT
    )
    sample = ds[0]
    # sample["image_id"]             -> int
    # sample["img_path"]             -> str
    # sample["pred_keypoints"]       -> Tensor[N, K, 2]
    # sample["pred_keypoint_scores"] -> Tensor[N, K]
    # sample["bboxes"]               -> Tensor[N, 4]   (top-down only)
    # sample["bbox_scores"]          -> Tensor[N]       (top-down only)

    gt = ds.get_gt(sample["image_id"])
    # gt["keypoints"]  -> Tensor[N_gt, K, 2]
    # gt["visibility"] -> Tensor[N_gt, K]
    # Returns None if no GT was loaded.

SceneGraphSampler
-----------------
Batch sampler used internally by both create_dataloader methods.  Groups
images by the number of people they contain so every batch has a uniform N,
enabling default_collate to stack tensors without padding.

- Shuffled mode interleaves batches from different group sizes globally.
- drop_last=True drops the final partial batch within each group.

prepare_pose_graph_batch
------------------------
Main entry point for the model.  Takes raw CrowdPose keypoints and returns a
block-diagonal PyG graph ready for GNN processing.

    from src.data import PoseDataset, prepare_pose_graph_batch

    ds = PoseDataset.from_path("data/crowdpose/annotations/crowdpose_train.json", "data/crowdpose/images")
    loader = ds.create_dataloader(batch_size=4)
    batch = next(iter(loader))
    prepare_pose_graph_batch(batch)  # optional: img_size=FloatTensor[B, 2]

Internally it:
  1. Computes a virtual root as the mean of visible left_hip, right_hip, and
     neck anchors per person (falls back to centroid of all visible joints).
  2. Prepends the root as node 0, shifting CrowdPose indices by +1 (14 -> 15
     keypoints per person).
  3. Subtracts the root position so the root node is at (0, 0).
  4. Divides all coordinates by max(W, H) per image (or 800 if not provided).
  5. Packs the B scene graphs into a single block-diagonal PyG structure.
"""

from __future__ import annotations

import json
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.io import read_image as tv_read_image
from torch.utils.data import Dataset


@lru_cache(maxsize=32000)
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

def make_edge_index(
    edge_list: list[tuple[int, int]],
    make_undirected: bool = True,
    add_self_loops: bool = False,
    num_nodes: int | None = None,
) -> Tensor:
    """Convert [(src, dst), ...] to LongTensor[2, E].

    make_undirected: duplicate each edge in both directions.
    add_self_loops:  append (i, i) for i in range(num_nodes) after undirected expansion.
    """
    if make_undirected:
        edge_list = edge_list + [(dst, src) for src, dst in edge_list]
    if add_self_loops:
        if num_nodes is None:
            raise ValueError("num_nodes must be provided when add_self_loops=True")
        edge_list = edge_list + [(i, i) for i in range(num_nodes)]
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


# Pre-built edge index for the CrowdPose graph (undirected + self-loops, 14 nodes).
CANONICAL_EDGE_INDEX: Tensor = make_edge_index(
    CROWDPOSE_EDGE_LIST, add_self_loops=True, num_nodes=14
)


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


@dataclass
class PoseSample:
    """Raw data from the DataLoader, owned before canonical graph construction."""

    keypoints: Tensor       # (B, N, 14, 2)  pixel-space CrowdPose joints; may be jittered
    visibility: Tensor      # (B, N, 14)      0=absent 1=occluded 2=visible
    gt_keypoints: Tensor    # (B, N, 14, 2)  always the clean original

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
    """Canonical pose graph batch, derived from a PoseSample."""

    node_pos: Tensor                # (B*N*K, 2)  image-normalised: pixel/[W,H]*2-1 ∈ (-1,1)
    edge_index: Tensor              # (2, B*N*E)
    node_batch: Tensor              # (B*N*K,)    PyG batch vector
    batch_idx: Tensor               # (B, N, K)
    pose_idx: Tensor                # (B, N, K)
    kpt_type_idx: Tensor            # (B, N, K)
    node_idx: Tensor                # (B, N, K)
    canonical_visibility: Tensor    # (B, N, 14)
    img_wh: Tensor                  # (B, 2)      [W, H] in pixels

    # Optional — populated during prepare_batch
    image_cond: torch.Tensor | None = None
    global_image_cond: torch.Tensor | None = None

    @property
    def B(self) -> int:
        return self.img_wh.shape[0]

    @property
    def N(self) -> int:
        return self.batch_idx.shape[1]

    @property
    def K(self) -> int:
        return self.canonical_visibility.shape[2]  # 14

    @property
    def num_nodes(self) -> int:
        return self.B * self.N * self.K

    @property
    def flat_kpt_type_idx(self) -> Tensor:
        return self.kpt_type_idx.reshape(-1)


def prepare_sample(batch: dict) -> PoseSample:
    """Build a PoseSample from a raw DataLoader batch dict.

    Sets ``gt_keypoints`` to the same tensor as ``keypoints``.  If
    augmentation later replaces ``sample.keypoints`` via ``keypoint_jitter``
    (which returns a new tensor), ``gt_keypoints`` keeps pointing to the
    original — no clone needed.
    """
    keypoints  = batch["keypoints"]
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
    """Build a canonical pose graph batch from a PoseSample.

    Normalises keypoints to (-1, 1) via: canonical = pixel / [W, H] * 2 - 1.
    The inverse is: pixel = (canonical + 1) / 2 * [W, H].

    Args:
        sample      PoseSample with keypoints (B, N, 14, 2) and visibility (B, N, 14)
        edge_index  LongTensor[2, E] | None  – defaults to CANONICAL_EDGE_INDEX
        img_size    FloatTensor[B, 2] | None – (W, H) per image; fallback is (800, 800)
    """
    keypoints  = sample.keypoints
    visibility = sample.visibility
    B = keypoints.shape[0]
    if edge_index is None:
        edge_index = CANONICAL_EDGE_INDEX
    ei = edge_index.to(keypoints.device)

    if img_size is not None:
        img_wh = img_size.to(keypoints.device)  # (B, 2)
    else:
        img_wh = torch.full((B, 2), 800.0, device=keypoints.device)

    # canonical = pixel / [W, H] * 2 - 1  →  coords in (-1, 1) for in-frame joints
    canonical_kpts = keypoints / img_wh.view(B, 1, 1, 2) * 2 - 1  # (B, N, 14, 2)

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
