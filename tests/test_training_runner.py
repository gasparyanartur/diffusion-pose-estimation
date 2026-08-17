from __future__ import annotations

import logging
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.training import CheckpointConfig, TrainingConfig, TrainingRunner


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
