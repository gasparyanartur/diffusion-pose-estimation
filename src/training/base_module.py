"""Shared Lightning base for all training modules in this project.

Provides lifecycle hooks that every module needs:
- strip frozen parameters from saved checkpoints
- restore frozen parameters from live weights on load
- log ``train/lr`` at the start of every training epoch
- print a model summary at the start of fit
"""

from __future__ import annotations

import logging

import pytorch_lightning as pl


class BaseTrainingModule(pl.LightningModule):
    """Base class for all project Lightning modules.

    Subclasses inherit four lifecycle hooks without reimplementing them:

    * ``on_save_checkpoint`` — removes frozen (``requires_grad=False``)
      parameters from the saved state dict, keeping checkpoints compact.
    * ``on_load_checkpoint`` — re-inserts any missing keys from the current
      live weights (restores the frozen parameters dropped on save).
    * ``on_fit_start`` — logs a ``ModelSummary`` at depth 3 via the root
      logger.
    * ``on_train_epoch_start`` — logs ``train/lr`` from the first param group
      of the first optimizer.
    """

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        frozen_keys = {name for name, p in self.named_parameters() if not p.requires_grad}
        state = checkpoint["state_dict"]
        for key in frozen_keys:
            state.pop(key, None)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state = checkpoint["state_dict"]
        for key, value in self.state_dict().items():
            if key not in state:
                state[key] = value

    # ------------------------------------------------------------------
    # Fit / epoch start
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        from pytorch_lightning.utilities.model_summary import ModelSummary
        log = logging.getLogger(__name__)
        summary = ModelSummary(self, max_depth=3)
        log.info("Model summary:\n%s", summary)

    def on_train_epoch_start(self) -> None:
        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        lr = opt.param_groups[0]["lr"]
        self.log("train/lr", lr, prog_bar=False, on_step=False, on_epoch=True)
