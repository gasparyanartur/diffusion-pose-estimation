"""Shared training primitives."""

from src.training.optim import build_adamw_with_schedule, iter_trainable
from src.training.config import CheckpointConfig, LoggingConfig, TrainingConfig
from src.training.runner import TrainingResult, TrainingRunner
from src.training.spec import TrainingSpec

__all__ = [
    "build_adamw_with_schedule",
    "iter_trainable",
    "CheckpointConfig",
    "LoggingConfig",
    "TrainingConfig",
    "TrainingResult",
    "TrainingRunner",
    "TrainingSpec",
]
