from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


EXTERNAL_ROOT = Path(__file__).resolve().parents[1] / "external" / "DiffusionRegPose"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

from engine import train_one_epoch


class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, samples: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "prediction": self.linear(samples),
            "target": 2 * samples + 1,
        }


class SquaredErrorCriterion(nn.Module):
    weight_dict = {"loss_regression": 1.0}

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return {
            "loss_regression": (
                outputs["prediction"] - outputs["target"]
            ).square().mean()
        }


def make_training_args(accumulation_steps: int) -> SimpleNamespace:
    return SimpleNamespace(
        amp=False,
        debug=False,
        gradient_accumulation_steps=accumulation_steps,
        onecyclelr=False,
        rank=1,
        use_dn=False,
    )


def make_microbatch_loader() -> list[tuple[torch.Tensor, list[dict[str, torch.Tensor]], None]]:
    return [
        (torch.tensor([[value]], dtype=torch.float32), [{}], None)
        for value in (1.0, 2.0, 3.0, 4.0)
    ]


def test_accumulated_microbatches_match_single_full_batch_update() -> None:
    torch.manual_seed(0)
    accumulated_model = LinearRegressionModel()
    full_batch_model = LinearRegressionModel()
    full_batch_model.load_state_dict(accumulated_model.state_dict())

    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)
    full_batch_optimizer = torch.optim.SGD(full_batch_model.parameters(), lr=0.1)
    criterion = SquaredErrorCriterion()

    train_one_epoch(
        accumulated_model,
        criterion,
        make_microbatch_loader(),
        accumulated_optimizer,
        torch.device("cpu"),
        epoch=0,
        args=make_training_args(accumulation_steps=4),
    )

    full_batch_samples = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    full_batch_loss = criterion(
        full_batch_model(full_batch_samples),
        [{}],
    )["loss_regression"]
    full_batch_loss.backward()
    full_batch_optimizer.step()

    torch.testing.assert_close(
        accumulated_model.linear.weight,
        full_batch_model.linear.weight,
    )