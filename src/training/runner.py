"""Hydra-independent training lifecycle orchestration."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.training.callbacks import build_checkpoint_callback
from src.training.config import TrainingConfig
from src.training.spec import TrainingSpec


@dataclass(frozen=True)
class TrainingResult:
    """Artifacts returned after a training or evaluation-only run."""

    checkpoint: str | None
    metrics: dict[str, Any] | None
    result_path: Path | None


class TrainingRunner:
    """Run a Lightning module according to a typed runtime policy.

    The runner has no Hydra dependency. Callers provide the resolved run
    directory, module, data module, and any pipeline-specific callbacks or
    checkpoint I/O plugins.
    """

    def __init__(self, config: TrainingConfig, run_directory: Path) -> None:
        self.config = config
        self.run_directory = run_directory
        self.checkpoint_directory = run_directory / "checkpoints"

    @staticmethod
    def devices_arg(raw: int | str | list[int]) -> int | str | list[int]:
        """Convert Hydra device values to the type expected by Lightning."""
        if raw == "auto":
            return "auto"
        if isinstance(raw, list):
            return raw
        value = str(raw)
        if "," in value:
            return [int(device) for device in value.split(",")]
        return int(value)

    def logger(self) -> logging.Logger:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(self.config.logging.logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            stream = logging.StreamHandler(sys.stdout)
            stream.setFormatter(logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(stream)
        log_path = str(self.run_directory / "train.log")
        if not any(
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename == log_path
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(log_path, mode="a")
            file_handler.setFormatter(logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(file_handler)
        return logger

    def seed_everything(self) -> None:
        if self.config.seed is not None:
            pl.seed_everything(self.config.seed, workers=True)

    def _loggers(self) -> list[Any]:
        loggers: list[Any] = []
        if self.config.logging.tensorboard:
            tensorboard_version = self._logger_version(
                self.config.logging.tensorboard_name
            )
            loggers.append(TensorBoardLogger(
                save_dir=str(self.run_directory),
                name=self.config.logging.tensorboard_name,
                version=tensorboard_version,
            ))
        if self.config.logging.csv:
            csv_version = self._logger_version(self.config.logging.csv_name)
            loggers.append(CSVLogger(
                save_dir=str(self.run_directory),
                name=self.config.logging.csv_name,
                version=csv_version,
            ))
        return loggers

    def _logger_version(self, logger_name: str) -> str:
        """Choose a non-destructive logger directory for resumed runs."""
        if not self.config.resume:
            return ""

        logger_directory = self.run_directory / logger_name
        version = "resume"
        suffix = 0
        while (logger_directory / version).exists():
            suffix += 1
            version = f"resume_{suffix}"
        return version

    def build_trainer(
        self,
        *,
        max_epochs: int,
        spec: TrainingSpec | None = None,
        callbacks: Sequence[pl.Callback] = (),
        plugins: Sequence[Any] = (),
    ) -> tuple[pl.Trainer, ModelCheckpoint]:
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        spec = spec or TrainingSpec()
        checkpoint_config = spec.checkpoint or self.config.checkpoint
        checkpoint = build_checkpoint_callback(
            self.checkpoint_directory, checkpoint_config
        )
        progress_callbacks = (
            [TQDMProgressBar(refresh_rate=1)]
            if self.config.enable_progress_bar else []
        )
        all_callbacks = [
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
            *progress_callbacks,
            *spec.callbacks,
            *callbacks,
        ]
        gradient_clip = self.config.grad_clip if self.config.grad_clip and self.config.grad_clip > 0 else None
        trainer = pl.Trainer(
            accelerator=self.config.accelerator,
            devices=self.devices_arg(self.config.devices),
            precision=self.config.precision,
            max_epochs=max_epochs,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            gradient_clip_val=gradient_clip,
            limit_val_batches=self.config.limit_val_batches,
            check_val_every_n_epoch=self.config.check_val_every_n_epoch,
            num_sanity_val_steps=self.config.num_sanity_val_steps,
            enable_progress_bar=self.config.enable_progress_bar,
            callbacks=all_callbacks,
            logger=self._loggers(),
            log_every_n_steps=self.config.log_every_n_steps,
            deterministic=False,
            plugins=[*spec.plugins, *plugins],
        )
        return trainer, checkpoint

    def run(
        self,
        module: pl.LightningModule,
        data_module: pl.LightningDataModule,
        *,
        max_epochs: int,
        spec: TrainingSpec | None = None,
        callbacks: Sequence[pl.Callback] = (),
        plugins: Sequence[Any] = (),
        result_filename: str = "test_results.json",
    ) -> TrainingResult:
        log = self.logger()
        spec = spec or TrainingSpec(result_filename=result_filename)
        trainer, checkpoint_callback = self.build_trainer(
            max_epochs=max_epochs, spec=spec, callbacks=callbacks, plugins=plugins
        )
        resume = self.config.resume
        if self.config.skip_training:
            if not resume:
                raise ValueError("skip_training=true requires training.resume")
            log.info("Skipping training; loading checkpoint: %s", resume)
            output = trainer.test(module, datamodule=data_module, ckpt_path=resume)
            checkpoint_path = resume
        else:
            trainer.fit(module, datamodule=data_module, ckpt_path=resume)
            checkpoint_path = (
                checkpoint_callback.best_model_path
                or checkpoint_callback.last_model_path
                or None
            )
            if not checkpoint_path or not self.config.test_after_fit:
                return TrainingResult(checkpoint_path, None, None)
            output = trainer.test(module, datamodule=data_module, ckpt_path=checkpoint_path)

        metrics = dict(output[0]) if output else None
        result_path = None
        if metrics is not None:
            result_path = self.run_directory / spec.result_filename
            result_path.write_text(
                json.dumps({"checkpoint": checkpoint_path, "metrics": metrics}, indent=4)
            )
        return TrainingResult(checkpoint_path, metrics, result_path)
