"""Shared optimizer/scheduler builder for Lightning pipelines.

Splits parameters into decay / no-decay groups, optionally adds linear warmup
and cosine annealing.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


class TwoCycleCosineAnnealingLR(torch.optim.lr_scheduler.LRScheduler):
    """Run one cosine decay, restart at a fixed absolute peak, then decay again."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_epochs: int,
        restart_epoch: int,
        first_minimum: float,
        restart_peak: float,
        final_minimum: float,
    ) -> None:
        if total_epochs < 2:
            raise ValueError("two-cycle cosine scheduling requires at least two epochs")
        if not 1 <= restart_epoch < total_epochs:
            raise ValueError("restart_epoch must be in [1, total_epochs)")
        if min(first_minimum, restart_peak, final_minimum) < 0.0:
            raise ValueError("two-cycle cosine learning rates must be non-negative")
        self.total_epochs = total_epochs
        self.restart_epoch = restart_epoch
        self.first_minimum = first_minimum
        self.restart_peak = restart_peak
        self.final_minimum = final_minimum
        super().__init__(optimizer)
        if any(base_lr < first_minimum for base_lr in self.base_lrs):
            raise ValueError("first_minimum cannot exceed an initial learning rate")
        if restart_peak < final_minimum:
            raise ValueError("restart_peak cannot be smaller than final_minimum")

    @staticmethod
    def _cosine_value(start: float, end: float, progress: float) -> float:
        return end + (start - end) * (1.0 + math.cos(math.pi * progress)) / 2.0

    def get_lr(self) -> list[float]:
        epoch = self.last_epoch
        if epoch < self.restart_epoch:
            denominator = max(self.restart_epoch - 1, 1)
            progress = epoch / denominator
            return [
                self._cosine_value(base_lr, self.first_minimum, progress)
                for base_lr in self.base_lrs
            ]
        denominator = max(self.total_epochs - self.restart_epoch - 1, 1)
        progress = min((epoch - self.restart_epoch) / denominator, 1.0)
        value = self._cosine_value(self.restart_peak, self.final_minimum, progress)
        return [value for _ in self.base_lrs]


def build_adamw_with_schedule(
    module: nn.Module,
    *,
    lr: float,
    lr_backbone: float | None = None,
    weight_decay: float,
    lr_T_max: int | None = None,
    lr_warmup_epochs: int = 0,
    lr_min_frac: float = 1e-2,
    scheduler_type: str = "cosine",
    lr_decay_milestones: tuple[int, ...] = (),
    lr_cycle_restart_epoch: int | None = None,
    lr_cycle_first_min: float | None = None,
    lr_cycle_restart_peak: float | None = None,
    lr_cycle_final_min: float | None = None,
    split_weight_decay: bool = True,
) -> dict | torch.optim.Optimizer:
    """Build AdamW with optional decay splitting and a learning-rate schedule.

    Returns either the optimiser alone, or a Lightning ``configure_optimizers``
    dict containing ``optimizer`` and ``lr_scheduler``.
    """
    decay: dict[str, list[torch.Tensor]] = {"main": [], "backbone": []}
    no_decay: dict[str, list[torch.Tensor]] = {"main": [], "backbone": []}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        group_name = "backbone" if name.startswith("model.backbone.") else "main"
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay[group_name].append(param)
        else:
            decay[group_name].append(param)
    param_groups = []
    if split_weight_decay:
        groups = (
            ("main", decay["main"], weight_decay),
            ("main", no_decay["main"], 0.0),
            ("backbone", decay["backbone"], weight_decay),
            ("backbone", no_decay["backbone"], 0.0),
        )
    else:
        groups = (
            ("main", decay["main"] + no_decay["main"], weight_decay),
            ("backbone", decay["backbone"] + no_decay["backbone"], weight_decay),
        )
    for group_name, parameters, group_weight_decay in groups:
        if parameters:
            param_groups.append({
                "params": parameters,
                "lr": lr if group_name == "main" else (lr_backbone or lr),
                "weight_decay": group_weight_decay,
            })
    optimizer = torch.optim.AdamW(param_groups)

    if scheduler_type not in {"cosine", "multistep", "two_cycle_cosine"}:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")
    if scheduler_type == "multistep" and not lr_decay_milestones:
        raise ValueError("multistep scheduling requires lr_decay_milestones")

    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
    milestones: list[int] = []
    if lr_warmup_epochs > 0:
        schedulers.append(
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=lr_warmup_epochs,
            )
        )
        milestones.append(lr_warmup_epochs)
    if scheduler_type == "multistep":
        schedulers.append(torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(lr_decay_milestones), gamma=0.1,
        ))
    elif lr_T_max is not None:
        if scheduler_type == "two_cycle_cosine":
            if any(value is None for value in (
                lr_cycle_restart_epoch,
                lr_cycle_first_min,
                lr_cycle_restart_peak,
                lr_cycle_final_min,
            )):
                raise ValueError("two_cycle_cosine requires all cycle parameters")
            schedulers.append(
                TwoCycleCosineAnnealingLR(
                    optimizer,
                    total_epochs=lr_T_max,
                    restart_epoch=lr_cycle_restart_epoch,
                    first_minimum=lr_cycle_first_min,
                    restart_peak=lr_cycle_restart_peak,
                    final_minimum=lr_cycle_final_min,
                )
            )
        else:
            cosine_epochs = max(1, lr_T_max - lr_warmup_epochs)
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cosine_epochs, eta_min=lr * lr_min_frac,
                )
            )
    if not schedulers:
        return optimizer
    if len(schedulers) == 1:
        scheduler: torch.optim.lr_scheduler.LRScheduler = schedulers[0]
    else:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=schedulers, milestones=milestones,
        )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
    }


def iter_trainable(module: nn.Module) -> Iterable[torch.Tensor]:
    """All trainable parameters of ``module``."""
    return (p for p in module.parameters() if p.requires_grad)
