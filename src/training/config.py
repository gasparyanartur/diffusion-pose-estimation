"""Typed runtime configuration for the shared training runner."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from src.configs.base_config import BaseConfig


class CheckpointConfig(BaseConfig):
    """Checkpoint policy consumed by :class:`TrainingRunner`."""

    monitor: str | None = None
    mode: Literal["min", "max"] = "min"
    filename: str = "epoch{epoch:03d}"
    save_top_k: int = 1
    save_last: bool = True
    save_last_every_epoch: bool = False
    every_n_epochs: int | None = None
    final_epoch: bool = False

    @field_validator("save_top_k")
    @classmethod
    def _validate_save_top_k(cls, value: int) -> int:
        if value < 0:
            raise ValueError("save_top_k must be non-negative")
        return value

    @field_validator("every_n_epochs")
    @classmethod
    def _validate_every_n_epochs(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("every_n_epochs must be positive when provided")
        return value


class LoggingConfig(BaseConfig):
    """File and experiment logger settings."""

    logger_name: str = "training"
    tensorboard: bool = True
    csv: bool = True
    tensorboard_name: str = "tb_logs"
    csv_name: str = "csv_logs"


class TrainingConfig(BaseConfig):
    """Hydra-facing runtime policy for a training run.

    Model-specific optimization fields remain on pipeline configs until each
    pipeline is migrated to consume this policy explicitly.
    """

    accelerator: str = "auto"
    devices: int | str | list[int] = "auto"
    precision: int | str = "32"
    max_epochs: int | None = None
    grad_clip: float | None = None
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 50
    seed: int | None = None
    resume: str | None = None
    skip_training: bool = False
    test_after_fit: bool = True
    limit_val_batches: int | float = 1.0
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("log_every_n_steps")
    @classmethod
    def _validate_log_every_n_steps(cls, value: int) -> int:
        if value < 1:
            raise ValueError("log_every_n_steps must be positive")
        return value

    @field_validator("accumulate_grad_batches")
    @classmethod
    def _validate_accumulate_grad_batches(cls, value: int) -> int:
        if value < 1:
            raise ValueError("accumulate_grad_batches must be positive")
        return value
