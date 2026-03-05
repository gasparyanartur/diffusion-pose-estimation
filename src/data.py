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

Typical usage
-------------
>>> from pathlib import Path
>>> from torch.utils.data import DataLoader
>>> from src.data import PredictionDataset, GroupedBatchSampler, simple_collate
>>>
>>> dataset = PredictionDataset.from_path(Path("results/cache/predictions.pkl"))
>>> sampler = GroupedBatchSampler(dataset, batch_size=16)
>>> loader  = DataLoader(dataset, batch_sampler=sampler, collate_fn=simple_collate)
>>> for batch in loader:
...     kpts = batch["keypoints"]  # (B, N, K, 2)
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Literal

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

    ================== =====================================================
    Key                Description
    ================== =====================================================
    ``image_id``       ``int`` – dataset image identifier
    ``img_path``       ``str`` – path to the raw image (may be empty)
    ``keypoints``      ``Tensor[N, K, 2]`` – ``(x, y)`` keypoint coords
    ``keypoint_scores``  ``Tensor[N, K]`` – per-keypoint confidence scores
    ``bboxes``         ``Tensor[N, 4]`` – ``xyxy`` boxes *(top-down only)*
    ``bbox_scores``    ``Tensor[N]`` – per-person detector scores
                       *(top-down only)*
    ================== =====================================================
    """

    def __init__(
        self,
        model_name: str,
        architecture_type: Literal["top-down", "bottom-up"],
        num_keypoints: int,
        predictions: list[dict[str, Any]],
    ) -> None:
        self.model_name = model_name
        self.architecture_type = architecture_type
        self.num_keypoints = num_keypoints
        self.predictions = predictions
        self.has_bboxes: bool = bool(predictions) and "bboxes" in predictions[0]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, path: Path | str) -> "PredictionDataset":
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

        Returns
        -------
        PredictionDataset
        """
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return cls(
            model_name=data["model_name"],
            architecture_type=data["architecture_type"],
            num_keypoints=data["num_keypoints"],
            predictions=data["predictions"],
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
            # (N, K, 2)  –  (x, y) coordinates
            "keypoints": torch.tensor(pred["keypoints"], dtype=torch.float32),
            # (N, K)
            "keypoint_scores": torch.tensor(pred["keypoint_scores"], dtype=torch.float32),
        }
        if self.has_bboxes:
            # (N, 4) or (N, 5)
            sample["bboxes"] = torch.tensor(pred["bboxes"], dtype=torch.float32)
            # (N,)
            sample["bbox_scores"] = torch.tensor(pred["bbox_scores"], dtype=torch.float32)
        return sample


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

        if self.shuffle:
            for group in self.groups.values():
                random.shuffle(group)

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        """Yield lists of indices, one list per batch.

        Batches are produced group-by-group (ascending number of people).
        Within each group, indices are chunked into sub-lists of length
        :attr:`batch_size`.
        """
        for n_people in sorted(self.groups):
            indices = self.groups[n_people]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        """Return the total number of batches across all groups."""
        total = 0
        for indices in self.groups.values():
            total += len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                total += 1
        return total


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

        ================== =====================================================
        Key                Shape / type
        ================== =====================================================
        ``image_ids``      ``list[int]``
        ``img_paths``      ``list[str]``
        ``keypoints``      ``Tensor[B, N, K, 2]``
        ``keypoint_scores``  ``Tensor[B, N, K]``
        ``bboxes``         ``Tensor[B, N, 4]`` or ``None``
        ``bbox_scores``    ``Tensor[B, N]`` or ``None``
        ================== =====================================================

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
    return {
        "image_ids": [s["image_id"] for s in batch],
        "img_paths": [s.get("img_path", "") for s in batch],
        # (B, N, K, 2)
        "keypoints": torch.stack([s["keypoints"] for s in batch]),
        # (B, N, K)
        "keypoint_scores": torch.stack([s["keypoint_scores"] for s in batch]),
        # (B, N, 4/5) or None
        "bboxes": torch.stack([s["bboxes"] for s in batch]) if has_bboxes else None,
        # (B, N) or None
        "bbox_scores": (
            torch.stack([s["bbox_scores"] for s in batch]) if has_bboxes else None
        ),
    }
