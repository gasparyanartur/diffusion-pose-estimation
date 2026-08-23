"""Shared optimizer/scheduler builder for Lightning pipelines.

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
    lr_backbone: float | None = None,
    weight_decay: float,
    lr_T_max: int | None = None,
    lr_warmup_epochs: int = 0,
    lr_min_frac: float = 1e-2,
    scheduler_type: str = "cosine",
    lr_decay_milestones: tuple[int, ...] = (),
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

    if scheduler_type not in {"cosine", "multistep"}:
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
