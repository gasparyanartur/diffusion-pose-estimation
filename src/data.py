"""
Pose-prediction dataset and dataloader utilities.

This module provides:

* :class:`PredictionDataset` – a ``torch.utils.data.Dataset`` that wraps
  pre-computed pose-estimation predictions stored as a cached ``.pkl`` file.
  Each sample corresponds to one image and contains the predicted keypoints,
  keypoint confidence scores, and (optionally) bounding-boxes together with
  their scores.

* :class:`GroupedBatchSampler` – a custom ``BatchSampler`` that groups dataset
  indices by the number of detected people per image and yields batches that
  are *homogeneous* in that count.  This removes the need for padding and
  makes downstream graph-based processing straightforward.

* :func:`simple_collate` – a collation function compatible with
  :class:`GroupedBatchSampler` that stacks samples into tensors of shape
  ``(B, N, …)`` where *B* is the batch size and *N* is the (uniform) number
  of people.

* :func:`load_gt_annotations` – loads a COCO-style JSON annotation file and
  returns a mapping from ``image_id`` to a list of per-person GT records,
  each containing ``gt_keypoints`` ``(N_gt, K, 2)`` and ``gt_visibility``
  ``(N_gt, K)`` tensors.  Because the number of GT people per image (*N_gt*)
  is independent of the number of predictions (*N_pred*), GT tensors are kept
  as lists in collated batches rather than being stacked.

* :func:`make_edge_index` – converts a list of ``(src, dst)`` keypoint-index
  pairs into a ``LongTensor[2, E]`` edge index, optionally making the graph
  undirected.

* :func:`create_graph_from_keypoints` – converts a batched keypoints tensor
  ``(B, N, K, 2)`` into a PyG-compatible flat graph with a block-diagonal
  edge index ``(2, B·N·E)``, a ``node_pos`` tensor ``(B·N·K, 2)``, and a
  ``batch`` vector ``(B·N·K,)`` for pooling.

Typical usage
-------------
>>> from pathlib import Path
>>> from torch.utils.data import DataLoader
>>> from src.data import PredictionDataset, GroupedBatchSampler, simple_collate
>>>
>>> # Predictions only
>>> dataset = PredictionDataset.from_path(Path("results/cache/predictions.pkl"))
>>>
>>> # Predictions + ground truth
>>> dataset = PredictionDataset.from_path_with_gt(
...     Path("results/cache/predictions.pkl"),
...     Path("data/crowdpose/annotations/crowdpose_test.json"),
... )
>>> sampler = GroupedBatchSampler(dataset, batch_size=16)
>>> loader  = DataLoader(dataset, batch_sampler=sampler, collate_fn=simple_collate)
>>> for batch in loader:
...     kpts    = batch["keypoints"]     # (B, N_pred, K, 2)
...     gt_kpts = batch["gt_keypoints"]  # list[Tensor[N_gt_i, K, 2]], length B
"""

from __future__ import annotations

import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PredictionDataset(Dataset):
    """Dataset wrapping pre-computed pose-estimation predictions.

    The dataset is format-agnostic and works with both *top-down* (one crop
    per person) and *bottom-up* (full-image) models.  Each sample represents
    **one image** and contains a variable number of detected people; samples
    are *not* padded to a fixed length.

    N denotes the number of detected people in the image
    K denotes the number of keypoints per person (e.g. 14 for CrowdPose)
    Bounding boxes are given as (x_min, y_min, x_max, y_max) 
    Keypoints are given as (x, y) coordinates in the original image space.

    Parameters
    ----------
    model_name:
        Human-readable identifier for the model that produced the predictions
        (e.g. ``"rtmpose-m_8xb64-210e_crowdpose-256x192"``).
    architecture_type:
        Either ``"top-down"`` or ``"bottom-up"``.
    num_keypoints:
        Number of keypoints per person (e.g. 14 for CrowdPose).
    predictions:
        Raw prediction records.  Each record is a :class:`dict` with at
        minimum the following keys:

        * ``"image_id"``         – ``int``
        * ``"keypoints"``        – array-like of shape ``(N, K, 2)``
        * ``"keypoint_scores"``  – array-like of shape ``(N, K)``

        Top-down models additionally provide:

        * ``"bboxes"``       – array-like of shape ``(N, 4)`` or ``(N, 5)``
        * ``"bbox_scores"``  – array-like of shape ``(N,)``

    Attributes
    ----------
    has_bboxes : bool
        ``True`` when bounding-box predictions are available.

    Sample structure
    ----------------
    Each item returned by :meth:`__getitem__` is a :class:`dict`:

    ================== =====================================================================
    Key                Description
    ================== =====================================================================
    ``image_id``       ``int`` – dataset image identifier
    ``img_path``       ``str`` – path to the raw image (may be empty)
    ``keypoints``      ``Tensor[N_pred, K, 2]`` – ``(x, y)`` predicted keypoint coords
    ``keypoint_scores``  ``Tensor[N_pred, K]`` – per-keypoint confidence scores
    ``bboxes``         ``Tensor[N_pred, 4]`` – ``xyxy`` boxes *(top-down only)*
    ``bbox_scores``    ``Tensor[N_pred]`` – per-person detector scores *(top-down only)*
    ``gt_keypoints``   ``Tensor[N_gt, K, 2]`` – ground-truth ``(x, y)`` coords *(optional)*
    ``gt_visibility``  ``Tensor[N_gt, K]`` – visibility flags: 0=absent, 1=occluded,
                       2=visible *(optional)*
    ================== =====================================================================
    """

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
        # Optional ground-truth annotations keyed by image_id
        self._gt_by_image_id: dict[int, list[dict[str, Any]]] | None = gt_by_image_id
        self.has_gt: bool = gt_by_image_id is not None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, path: Path | str,         ann_path: Path | str | None = None, **kwargs: Any) -> "PredictionDataset":
        """Load a :class:`PredictionDataset` from a cached predictions file.

        The file is expected to be a ``pickle``-serialised :class:`dict` with
        the following top-level keys:

        * ``"model_name"``         – ``str``
        * ``"architecture_type"``  – ``str``
        * ``"num_keypoints"``      – ``int``
        * ``"predictions"``        – ``list[dict]``

        Parameters
        ----------
        path:
            Filesystem path to the ``.pkl`` cache file.
        ann_path:
            Path to a COCO-style annotation JSON file (e.g.
            ``crowdpose_test.json``).  This is required to set the
            :attr:`has_gt` flag correctly, but the actual GT annotations are
            not loaded until :meth:`from_path_with_gt` is used.
            
        Returns
        -------
        PredictionDataset
        """
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
        """Create a :class:`~torch.utils.data.DataLoader` backed by :class:`GroupedBatchSampler`.

        Batches are homogeneous in the number of people *N*, so the collated
        ``keypoints`` tensor always has shape ``(B, N, K, 2)`` with no
        padding required.

        Parameters
        ----------
        batch_size:
            Number of images per batch.
        shuffle:
            When ``True`` (default), indices within each group are shuffled and
            the resulting batches are globally interleaved across groups.
        drop_last:
            When ``True``, discard the final partial batch in each group.
        num_workers:
            Number of worker processes for data loading (passed directly to
            :class:`~torch.utils.data.DataLoader`).

        Returns
        -------
        torch.utils.data.DataLoader
        """
        import torch.utils.data

        sampler = GroupedBatchSampler(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
        return torch.utils.data.DataLoader(
            self,
            batch_sampler=sampler,
            collate_fn=simple_collate,
            num_workers=num_workers,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.predictions)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the sample for image at position *idx*.

        Parameters
        ----------
        idx:
            Zero-based index into :attr:`predictions`.

        Returns
        -------
        dict
            See the *Sample structure* table in the class docstring.
        """
        pred = self.predictions[idx]
        sample: dict[str, Any] = {
            "image_id": pred["image_id"],
            "img_path": pred.get("img_path", ""),
            # (N_pred, K, 2)  –  (x, y) coordinates
            "keypoints": torch.tensor(pred["keypoints"], dtype=torch.float32),
            # (N_pred, K)
            "keypoint_scores": torch.tensor(pred["keypoint_scores"], dtype=torch.float32),
        }
        if self.has_bboxes:
            # (N_pred, 4) or (N_pred, 5)
            sample["bboxes"] = torch.tensor(pred["bboxes"], dtype=torch.float32)
            # (N_pred,)
            sample["bbox_scores"] = torch.tensor(pred["bbox_scores"], dtype=torch.float32)
        if self.has_gt and self._gt_by_image_id is not None:
            gt_anns = self._gt_by_image_id.get(pred["image_id"], [])
            if gt_anns:
                # Each annotation has a flat keypoints list: [x, y, v, x, y, v, …]
                kpts_flat = np.array(
                    [a["keypoints"] for a in gt_anns], dtype=np.float32
                )  # (N_gt, K*3)
                kpts = kpts_flat.reshape(len(gt_anns), self.num_keypoints, 3)
                # (N_gt, K, 2)
                sample["gt_keypoints"] = torch.from_numpy(kpts[:, :, :2])
                # (N_gt, K)  – 0: absent, 1: occluded, 2: visible
                sample["gt_visibility"] = torch.from_numpy(kpts[:, :, 2])
            else:
                # Image exists in predictions but has no GT annotations
                sample["gt_keypoints"] = torch.zeros(
                    0, self.num_keypoints, 2, dtype=torch.float32
                )
                sample["gt_visibility"] = torch.zeros(
                    0, self.num_keypoints, dtype=torch.float32
                )
        return sample


# ---------------------------------------------------------------------------
# Ground-truth loader
# ---------------------------------------------------------------------------


def load_gt_annotations(
    ann_path: Path | str,
) -> dict[int, list[dict[str, Any]]]:
    """Load a COCO-style annotation JSON and index annotations by ``image_id``.

    Supports any dataset whose annotation file follows the COCO convention:

    .. code-block:: json

        {
            "images": [{"id": 1, ...}, ...],
            "annotations": [
                {
                    "image_id": 1,
                    "keypoints": [x0, y0, v0, x1, y1, v1, ...],
                    ...
                },
                ...
            ]
        }

    Keypoint visibility flags follow the COCO convention:
    ``0`` = not labeled, ``1`` = labeled but occluded, ``2`` = labeled and
    visible.

    Parameters
    ----------
    ann_path:
        Path to the annotation JSON file.

    Returns
    -------
    dict[int, list[dict]]
        Mapping from ``image_id`` to the list of annotation dicts for that
        image.  Images with no annotations are absent from the mapping.
    """
    with open(ann_path) as fh:
        data = json.load(fh)

    gt: dict[int, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        gt.setdefault(ann["image_id"], []).append(ann)
    return gt


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class GroupedBatchSampler:
    """Batch sampler that groups images by their number of detected people.

    Yielding batches that are *homogeneous* in the number of people (*N*)
    avoids any need for padding: all ``keypoints`` tensors within a batch have
    identical shape ``(K, 2)`` per person, so the collated batch has a clean
    shape of ``(B, N, K, 2)``.

    Parameters
    ----------
    dataset:
        A :class:`PredictionDataset` (or any dataset whose samples contain a
        ``"keypoints"`` tensor whose first dimension is the number of people).
    batch_size:
        Number of images per batch.
    drop_last:
        When ``True``, discard the final (potentially smaller) batch for each
        group.  When ``False`` (default), include it.
    shuffle:
        When ``True`` (default), shuffle the indices within every group at
        construction time.

    Notes
    -----
    Groups are built eagerly at construction time by iterating over the entire
    dataset once.  For very large datasets consider caching the per-image
    people counts separately.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Group indices by number of people (first dim of keypoints tensor)
        self.groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):  # type: ignore[arg-type]
            sample = dataset[idx]
            n_people: int = sample["keypoints"].shape[0]
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

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        """Yield pre-computed batches.

        When *shuffle* is ``True`` the batch order is globally randomised
        across groups, so images with different numbers of people are
        interleaved throughout training rather than being presented group by
        group.
        """
        yield from self._batches

    def __len__(self) -> int:
        """Return the total number of batches across all groups."""
        return len(self._batches)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def simple_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of same-*N* samples into batched tensors.

    Designed to pair with :class:`GroupedBatchSampler`, which guarantees that
    all samples in ``batch`` have the same number of people *N*.  Tensors are
    stacked along a new leading batch dimension.

    Parameters
    ----------
    batch:
        A list of sample dicts as returned by
        :meth:`PredictionDataset.__getitem__`.  All samples must have the same
        value of ``keypoints.shape[0]`` (number of people).

    Returns
    -------
    dict
        A collated batch with the following keys:

        ================== =================================================================
        Key                Shape / type
        ================== =================================================================
        ``image_ids``      ``list[int]``
        ``img_paths``      ``list[str]``
        ``keypoints``      ``Tensor[B, N_pred, K, 2]``
        ``keypoint_scores``  ``Tensor[B, N_pred, K]``
        ``bboxes``         ``Tensor[B, N_pred, 4]`` or ``None``
        ``bbox_scores``    ``Tensor[B, N_pred]`` or ``None``
        ``gt_keypoints``   ``list[Tensor[N_gt_i, K, 2]]`` length *B* or ``None``.
                           Kept as a list because *N_gt* may differ across images
                           even within a homogeneous-*N_pred* batch.
        ``gt_visibility``  ``list[Tensor[N_gt_i, K]]`` length *B* or ``None``
        ================== =================================================================

    Raises
    ------
    RuntimeError
        If the samples have inconsistent keypoint shapes (different *N*).
    """
    # Validate homogeneous N across samples
    n_people_set = {s["keypoints"].shape[0] for s in batch}
    if len(n_people_set) > 1:
        raise RuntimeError(
            "simple_collate expects all samples to have the same number of "
            f"people, but found counts: {sorted(n_people_set)}.  "
            "Use GroupedBatchSampler to ensure homogeneous batches."
        )

    has_bboxes = "bboxes" in batch[0]
    has_gt = "gt_keypoints" in batch[0]
    return {
        "image_ids": [s["image_id"] for s in batch],
        "img_paths": [s.get("img_path", "") for s in batch],
        # (B, N_pred, K, 2)
        "keypoints": torch.stack([s["keypoints"] for s in batch]),
        # (B, N_pred, K)
        "keypoint_scores": torch.stack([s["keypoint_scores"] for s in batch]),
        # (B, N_pred, 4/5) or None
        "bboxes": torch.stack([s["bboxes"] for s in batch]) if has_bboxes else None,
        # (B, N_pred) or None
        "bbox_scores": (
            torch.stack([s["bbox_scores"] for s in batch]) if has_bboxes else None
        ),
        # list[Tensor[N_gt_i, K, 2]] or None  –  N_gt may differ per image
        "gt_keypoints": [s["gt_keypoints"] for s in batch] if has_gt else None,
        # list[Tensor[N_gt_i, K]] or None
        "gt_visibility": [s["gt_visibility"] for s in batch] if has_gt else None,
    }


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

# CrowdPose skeleton: 14 keypoints (0-indexed)
# 0=left_shoulder  1=right_shoulder  2=left_elbow   3=right_elbow
# 4=left_wrist     5=right_wrist     6=left_hip     7=right_hip
# 8=left_knee      9=right_knee      10=left_ankle  11=right_ankle
# 12=head          13=neck
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
) -> Tensor:
    """Convert a list of ``(src, dst)`` keypoint-index pairs to a PyG edge index.

    Parameters
    ----------
    edge_list:
        List of ``(src, dst)`` tuples using 0-based keypoint indices.
    make_undirected:
        When ``True`` (default), each directed edge ``(u, v)`` is mirrored
        with ``(v, u)`` so the resulting graph is undirected.

    Returns
    -------
    Tensor
        ``LongTensor[2, E]`` where ``E`` equals ``len(edge_list)`` when
        ``make_undirected=False``, or ``2 * len(edge_list)`` otherwise.
    """
    if make_undirected:
        edge_list = edge_list + [(dst, src) for src, dst in edge_list]
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def create_graph_from_keypoints(
    edge_index: Tensor,
    keypoints: Tensor,
) -> dict[str, Tensor]:
    """Build a PyG-compatible batched graph from a keypoints tensor.

    All *B* pose graphs are packed into a single block-diagonal graph,
    following the PyG mini-batch convention.  Node indices are laid out as::

        node_idx[b, n, k] = b*N*K + n*K + k

    Parameters
    ----------
    edge_index:
        ``LongTensor[2, E]`` – keypoint-level edge index (indices in
        ``[0, K)``) as returned by :func:`make_edge_index`.
    keypoints:
        ``FloatTensor[B, N, K, 2]`` – ``(x, y)`` keypoint coordinates where
        *B* is the batch size, *N* the number of people per image, and *K*
        the number of keypoints per person.

    Returns
    -------
    dict with the following keys:

    ============= ============================================================
    Key           Shape / description
    ============= ============================================================
    ``edge_index``  ``(2, B·N·E)`` – global edge index, pass to PyG conv layers
    ``node_pos``    ``(B·N·K, 2)`` – flat node feature / position tensor
    ``batch``       ``(B·N·K,)`` – PyG batch vector for pooling
    ``batch_idx``   ``(B, N, K)`` – structured batch index; mask with
                    ``batch_idx == b`` to select sample *b*
    ``pose_idx``    ``(B, N, K)`` – structured pose index; mask with
                    ``pose_idx == n`` to select person *n*
    ``node_idx``    ``(B, N, K)`` – global flat node index for each ``(b,n,k)``
    ============= ============================================================
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
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, K)  # (B, N, K)
    pose_idx  = torch.arange(N, device=device).view(1, N, 1).expand(B, N, K)  # (B, N, K)

    # Build flat block-diagonal edge index
    global_edge_from = node_idx[..., edge_index[0]].reshape(-1)  # (B*N*E,)
    global_edge_to   = node_idx[..., edge_index[1]].reshape(-1)  # (B*N*E,)
    flat_edge_index  = torch.stack([global_edge_from, global_edge_to], dim=0)  # (2, B*N*E)

    # PyG batch vector
    batch_vec = torch.arange(B, device=device).repeat_interleave(N * K)  # (B*N*K,)

    node_pos = keypoints.reshape(B * N * K, 2)  # (B*N*K, 2)

    return {
        "edge_index": flat_edge_index,  # (2, B*N*E)
        "node_pos":   node_pos,         # (B*N*K, 2)
        "batch":      batch_vec,        # (B*N*K,)
        "batch_idx":  batch_idx,        # (B, N, K)
        "pose_idx":   pose_idx,         # (B, N, K)
        "node_idx":   node_idx,         # (B, N, K)
    }
