"""Reusable Lightning callbacks and checkpoint I/O policies."""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from lightning_fabric.plugins.io import TorchCheckpointIO
from pytorch_lightning.callbacks import ModelCheckpoint

from src.training.config import CheckpointConfig


class FinalEpochModelCheckpoint(ModelCheckpoint):
    """Run a periodic checkpoint policy on the final epoch as well."""

    def _save_if_monitor_available(self, trainer: "pl.Trainer") -> None:
        if self._should_skip_saving_checkpoint(trainer):
            return
        monitor_candidates = self._monitor_candidates(trainer)
        if self.monitor is not None and self.monitor not in monitor_candidates:
            return
        self._save_topk_checkpoint(trainer, monitor_candidates)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not self._should_save_on_train_epoch_end(trainer):
            return
        is_last = (trainer.current_epoch + 1) == trainer.max_epochs
        on_period = (
            self._every_n_epochs >= 1
            and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        )
        if is_last and not on_period:
            if self._should_save_on_train_epoch_end(trainer):
                self._save_if_monitor_available(trainer)
        else:
            if self.monitor is not None and self.monitor not in self._monitor_candidates(trainer):
                return
            super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        is_last = (trainer.current_epoch + 1) == trainer.max_epochs
        on_period = (
            self._every_n_epochs >= 1
            and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        )
        if is_last and not on_period:
            self._save_if_monitor_available(trainer)
        else:
            if self.monitor is not None and self.monitor not in self._monitor_candidates(trainer):
                return
            super().on_validation_end(trainer, pl_module)


class SaveLastEveryEpochModelCheckpoint(ModelCheckpoint):
    """Save the latest completed validation epoch independently of top-k."""

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # The external DiffusionRegPose trainer has no validation loop. Keep
        # its latest-checkpoint behavior when validation is disabled.
        val_batches = getattr(trainer, "num_val_batches", [1])
        if any(val_batches):
            return
        if self._should_skip_saving_checkpoint(trainer):
            return
        self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._should_skip_saving_checkpoint(trainer) or self._should_save_on_train_epoch_end(trainer):
            return
        monitor_candidates = self._monitor_candidates(trainer)
        if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
            self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)


class TrustedCheckpointIO(TorchCheckpointIO):
    """Load trusted local checkpoints without weights-only restrictions."""

    def load_checkpoint(self, path, map_location=None, weights_only=None):
        return super().load_checkpoint(path, map_location=map_location, weights_only=False)


def build_checkpoint_callback(
    directory: Path, config: CheckpointConfig
) -> ModelCheckpoint:
    if config.final_epoch:
        callback_type = FinalEpochModelCheckpoint
    elif config.save_last_every_epoch:
        callback_type = SaveLastEveryEpochModelCheckpoint
    else:
        callback_type = ModelCheckpoint
    return callback_type(
        dirpath=directory,
        filename=config.filename,
        monitor=config.monitor,
        mode=config.mode,
        save_top_k=config.save_top_k,
        save_last=config.save_last,
        every_n_epochs=config.every_n_epochs,
        save_on_train_epoch_end=False if (config.final_epoch or config.save_last_every_epoch) else None,
        auto_insert_metric_name=False,
    )
