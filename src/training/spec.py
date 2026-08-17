"""Pipeline-specific extensions for the shared training runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pytorch_lightning as pl

from src.training.config import CheckpointConfig


@dataclass(frozen=True)
class TrainingSpec:
    """Pipeline-specific lifecycle policy supplied to ``TrainingRunner``."""

    checkpoint: CheckpointConfig | None = None
    callbacks: Sequence[pl.Callback] = field(default_factory=tuple)
    plugins: Sequence[Any] = field(default_factory=tuple)
    result_filename: str = "test_results.json"

