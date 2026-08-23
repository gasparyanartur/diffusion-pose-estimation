from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, Dataset

from src.training import CheckpointConfig, TrainingConfig, TrainingRunner
from src.training.callbacks import SaveLastEveryEpochModelCheckpoint, build_checkpoint_callback


def test_devices_arg_accepts_lightning_device_forms(tmp_path):
    runner = TrainingRunner(TrainingConfig(), tmp_path)

    assert runner.devices_arg("auto") == "auto"
    assert runner.devices_arg("2") == 2
    assert runner.devices_arg("0,1") == [0, 1]
    assert runner.devices_arg([1, 3]) == [1, 3]


def test_training_config_validates_checkpoint_policy():
    config = CheckpointConfig(monitor="val/loss", mode="min", save_top_k=2)

    assert config.monitor == "val/loss"
    assert config.save_top_k == 2

    with pytest.raises(ValueError, match="save_top_k"):
        CheckpointConfig(save_top_k=-1)

    with pytest.raises(ValueError, match="accumulate_grad_batches"):
        TrainingConfig(accumulate_grad_batches=0)


def test_diffusionregpose_style_checkpoint_policy_uses_latest_callback(tmp_path):
    callback = build_checkpoint_callback(
        tmp_path,
        CheckpointConfig(
            monitor="val/loss", save_last=True, save_last_every_epoch=True,
        ),
    )

    assert isinstance(callback, SaveLastEveryEpochModelCheckpoint)


def test_latest_checkpoint_advances_when_validation_loss_worsens(tmp_path):
    class OneSampleDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            del index
            return torch.ones(1)

    class WorseningValidationModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def training_step(self, batch, batch_idx):
            del batch_idx
            return self.weight * batch.mean()

        def validation_step(self, batch, batch_idx):
            del batch, batch_idx
            self.log("val/loss", float(self.current_epoch + 1), on_epoch=True)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.0)

    callback = build_checkpoint_callback(
        tmp_path,
        CheckpointConfig(
            monitor="val/loss", filename="epoch{epoch:03d}",
            save_top_k=1, save_last=True, save_last_every_epoch=True,
        ),
    )
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=2, logger=False,
        enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, callbacks=[callback],
    )
    loader = DataLoader(OneSampleDataset(), batch_size=1)
    trainer.fit(WorseningValidationModule(), loader, loader)

    assert torch.load(callback.last_model_path, map_location="cpu", weights_only=False)["epoch"] == 1
    assert torch.load(callback.best_model_path, map_location="cpu", weights_only=False)["epoch"] == 0


def test_runner_passes_gradient_accumulation_to_lightning(tmp_path):
    config = TrainingConfig(
        accelerator="cpu",
        devices=1,
        precision=32,
        accumulate_grad_batches=8,
        logging={"tensorboard": False, "csv": False},
    )
    trainer, _ = TrainingRunner(config, tmp_path).build_trainer(max_epochs=1)

    assert trainer.accumulate_grad_batches == 8


def test_runner_saves_last_checkpoint_without_validation(tmp_path):
    class TinyDataModule(pl.LightningDataModule):
        def train_dataloader(self):
            samples = [(torch.ones(1), torch.ones(1)) for _ in range(2)]
            return DataLoader(samples, batch_size=1)

        def val_dataloader(self):
            return DataLoader([(torch.ones(1), torch.ones(1))], batch_size=1)

    class TinyModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            del batch_idx
            return (self.layer(batch[0]) - batch[1]).square().mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    config = TrainingConfig(
        accelerator="cpu", devices=1, precision=32, max_epochs=2,
        limit_val_batches=0, test_after_fit=False,
        checkpoint={
            "monitor": None, "filename": "epoch{epoch:03d}",
            "save_top_k": 0, "save_last": True, "save_last_every_epoch": True,
        },
        logging={"tensorboard": True, "csv": False},
    )
    result = TrainingRunner(config, tmp_path).run(
        TinyModule(), TinyDataModule(), max_epochs=2,
    )

    assert result.checkpoint is not None
    assert Path(result.checkpoint).name == "last.ckpt"
    assert Path(result.checkpoint).exists()


def test_training_config_is_composed_from_hydra_root():
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path("conf").resolve())
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "hardware.accelerator=cpu",
                "hardware.devices=1",
                "hardware.precision=32",
                "training.max_epochs=3",
            ],
        )

    config = TrainingConfig.from_hydra_config(cfg.training)
    assert config.accelerator == "cpu"
    assert config.devices == 1
    assert config.precision == 32
    assert config.max_epochs == 3


def test_logger_does_not_duplicate_file_handlers(tmp_path):
    config = TrainingConfig(logging={"logger_name": "runner-test"})
    runner = TrainingRunner(config, tmp_path)

    logger = runner.logger()
    runner.logger()

    file_handlers = [
        handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)
    ]
    assert len(file_handlers) == 1
