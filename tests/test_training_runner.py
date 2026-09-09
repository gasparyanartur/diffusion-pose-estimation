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


def test_runner_configures_validation_interval(tmp_path):
    config = TrainingConfig(
        accelerator="cpu", devices=1, check_val_every_n_epoch=5,
        logging={"tensorboard": False, "csv": False},
    )

    trainer, _ = TrainingRunner(config, tmp_path).build_trainer(max_epochs=10)

    assert trainer.check_val_every_n_epoch == 5


def test_runner_disables_sanity_validation_when_configured(tmp_path):
    config = TrainingConfig(
        accelerator="cpu", devices=1, num_sanity_val_steps=0,
        logging={"tensorboard": False, "csv": False},
    )

    trainer, _ = TrainingRunner(config, tmp_path).build_trainer(max_epochs=1)

    assert trainer.num_sanity_val_steps == 0


def test_runner_configures_progress_bar(tmp_path):
    enabled, _ = TrainingRunner(
        TrainingConfig(
            accelerator="cpu", devices=1, enable_progress_bar=True,
            logging={"tensorboard": False, "csv": False},
        ), tmp_path / "enabled",
    ).build_trainer(max_epochs=1)
    disabled, _ = TrainingRunner(
        TrainingConfig(
            accelerator="cpu", devices=1, enable_progress_bar=False,
            logging={"tensorboard": False, "csv": False},
        ), tmp_path / "disabled",
    ).build_trainer(max_epochs=1)

    assert enabled.progress_bar_callback is not None
    assert disabled.progress_bar_callback is None
    assert any(callback.__class__.__name__ == "TQDMProgressBar" for callback in enabled.callbacks)
    assert not any(callback.__class__.__name__ == "TQDMProgressBar" for callback in disabled.callbacks)


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


def test_resumed_loggers_preserve_existing_history(tmp_path):
    csv_directory = tmp_path / "csv_logs"
    tensorboard_directory = tmp_path / "tb_logs"
    csv_directory.mkdir()
    tensorboard_directory.mkdir()
    (csv_directory / "metrics.csv").write_text("epoch,loss\n1,2\n")
    (tensorboard_directory / "events.out.tfevents.previous").write_text("old")

    config = TrainingConfig(
        resume="checkpoints/last.ckpt",
        logging={"tensorboard": True, "csv": True},
    )
    loggers = TrainingRunner(config, tmp_path)._loggers()

    assert {logger.log_dir for logger in loggers} == {
        str(csv_directory / "resume"),
        str(tensorboard_directory / "resume"),
    }
    assert (csv_directory / "metrics.csv").read_text() == "epoch,loss\n1,2\n"
    assert (tensorboard_directory / "events.out.tfevents.previous").read_text() == "old"


def test_resumed_loggers_increment_existing_continuation_directory(tmp_path):
    (tmp_path / "csv_logs" / "resume").mkdir(parents=True)
    (tmp_path / "tb_logs" / "resume").mkdir(parents=True)

    config = TrainingConfig(
        resume="checkpoints/last.ckpt",
        logging={"tensorboard": True, "csv": True},
    )
    loggers = TrainingRunner(config, tmp_path)._loggers()

    assert {logger.log_dir for logger in loggers} == {
        str(tmp_path / "csv_logs" / "resume_1"),
        str(tmp_path / "tb_logs" / "resume_1"),
    }
