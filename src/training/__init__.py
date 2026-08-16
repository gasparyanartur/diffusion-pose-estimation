"""Shared training primitives."""

from src.training.optim import build_adamw_with_schedule, iter_trainable

__all__ = ["build_adamw_with_schedule", "iter_trainable"]
