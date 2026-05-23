"""Shared optimiser/scheduler builder for training modules.

Splits parameters into decay / no-decay groups, optionally adds linear warmup
and cosine annealing.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def build_adamw_with_schedule(
    module: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    lr_T_max: int | None = None,
    lr_warmup_epochs: int = 0,
    lr_min_frac: float = 1e-2,
) -> dict | torch.optim.Optimizer:
    """Build AdamW with split decay groups + optional warmup+cosine schedule.

    Returns either the optimiser alone, or a Lightning ``configure_optimizers``
    dict containing ``optimizer`` and ``lr_scheduler``.
    """
    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr)

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
    if lr_T_max is not None:
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
