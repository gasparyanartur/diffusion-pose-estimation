"""Batch sampler that groups dataset indices by number of people per image."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator

from torch.utils.data import Dataset


class SceneGraphSampler:
    """Group dataset indices by number of people per image so every batch has uniform N.

    This lets ``default_collate`` stack tensors without padding.

    Args:
        dataset: Source dataset; each sample must expose a ``[group_key]``
            tensor whose first dimension equals the number of people.
        batch_size: Target batch size.
        drop_last: Drop the final partial batch within each group.
        shuffle: Shuffle indices within groups and the batch ordering.
        group_key: Sample dict key whose first dim is the group size.
    """

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

        self.groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):  # type: ignore[arg-type]
            sample = dataset[idx]
            n_people: int = sample[group_key].shape[0]
            self.groups[n_people].append(idx)

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
