#!/usr/bin/env python3
"""Compare one deterministic criterion/update step with DiffusionRegPose."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "external" / "DiffusionRegPose"


def _targets() -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(17)
    keypoints = torch.rand((2, 28), generator=generator) * 0.7 + 0.15
    visibility = torch.ones((2, 14))
    return [{
        "labels": torch.tensor([1, 1], dtype=torch.long),
        "boxes": torch.tensor(
            [[0.30, 0.35, 0.20, 0.30], [0.70, 0.65, 0.25, 0.20]],
            dtype=torch.float32,
        ),
        "keypoints": torch.cat((keypoints, visibility), dim=-1),
        "area": torch.tensor([0.06, 0.05], dtype=torch.float32),
    }]


def _outputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(23)
    return {
        "pred_logits": torch.randn((1, 6, 2), generator=generator),
        "pred_boxes": torch.rand((1, 6, 4), generator=generator) * 0.7 + 0.15,
        "pred_keypoints": torch.rand((1, 6, 42), generator=generator) * 0.7 + 0.15,
    }


def _native_criterion():
    from src.models.pipelines.pose_estimators.diffusionregpose.training import (
        DiffusionRegPoseCriterion,
        DiffusionRegPoseLossWeights,
    )

    weights = DiffusionRegPoseLossWeights(
        cls=2.0,
        bbox=5.0,
        giou=2.0,
        keypoints=10.0,
        oks=4.0,
        set_cost_class=2.0,
        set_cost_bbox=5.0,
        set_cost_giou=2.0,
        set_cost_keypoints=10.0,
        set_cost_oks=4.0,
        focal_alpha=0.25,
    )
    return DiffusionRegPoseCriterion(
        num_classes=2,
        num_body_points=14,
        num_box_decoder_layers=2,
        weights=weights,
    )


def _external_criterion():
    sys.path.insert(0, str(EXTERNAL))
    from models.diffusionregpose.criterion import SetCriterion
    from models.diffusionregpose.matcher import HungarianMatcher

    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        focal_alpha=0.25,
        cost_keypoints=10.0,
        cost_kpvis=0.0,
        cost_oks=4.0,
        num_body_points=14,
    )
    return SetCriterion(
        2,
        matcher=matcher,
        weight_dict={
            "loss_ce": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_keypoints": 10.0,
            "loss_oks": 4.0,
        },
        focal_alpha=0.25,
        losses=["labels", "boxes", "keypoints"],
        num_box_decoder_layers=2,
        num_body_points=14,
    )


def _weighted_external_loss(
    losses: dict[str, torch.Tensor], criterion,
) -> torch.Tensor:
    return sum(
        losses[name] * weight
        for name, weight in criterion.weight_dict.items()
        if name in losses
    )


def _run_native_model_step(device: torch.device) -> None:
    from src.models.pipelines.pose_estimators.diffusionregpose.training import (
        DiffusionRegPosePipelineConfig,
        build_diffusionregpose_targets,
    )

    torch.manual_seed(101)
    pipeline = DiffusionRegPosePipelineConfig(
        backbone_pretrained=False,
        pretrained_checkpoint=str(
            EXTERNAL / "pretrain_models" / "edpose_r50_crowdpose.pth"
        ),
        sampling_timesteps=1,
    ).build(max_epochs=1)
    pipeline.to(device)
    pipeline.train()
    batch = {
        "image": torch.randn(1, 3, 256, 256, device=device),
        "gt_keypoints": torch.rand(1, 1, 14, 2, device=device) * 1.2 - 0.6,
        "gt_labeled": torch.ones(1, 1, 14, device=device),
        "gt_visible": torch.ones(1, 1, 14, device=device),
        "gt_boxes_canon": torch.tensor(
            [[[0.0, 0.0, 0.5, 0.6]]], device=device
        ),
    }
    diffusion, timesteps, targets = build_diffusionregpose_targets(
        batch,
        pipeline.model,
        num_proposals=pipeline.model.num_proposals,
        num_body_points=14,
    )
    outputs = pipeline(batch["image"], diffusion, timesteps, targets)
    losses = pipeline.criterion(outputs, targets)
    if not torch.isfinite(losses["total"]):
        raise AssertionError(f"native one-step loss is not finite: {losses['total']}")
    losses["total"].backward()
    gradients = [
        parameter.grad
        for parameter in pipeline.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    trainable_count = sum(parameter.requires_grad for parameter in pipeline.parameters())
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("native one-step produced no gradients or a non-finite gradient")
    optimizer_spec = pipeline.configure_optimizers()
    optimizer = optimizer_spec["optimizer"] if isinstance(optimizer_spec, dict) else optimizer_spec
    before = next(parameter.detach().clone() for parameter in pipeline.parameters() if parameter.requires_grad)
    optimizer.step()
    after = next(parameter.detach() for parameter in pipeline.parameters() if parameter.requires_grad)
    if torch.equal(before, after):
        raise AssertionError("native one-step did not update a trainable parameter")
    print(f"native_model_loss: {losses['total'].item():.6f}")
    print(f"native_gradient_tensors: {len(gradients)}/{trainable_count}")
    print("native EDPose model one-step: PASS")


def _matches(criterion, outputs, targets) -> list[tuple[list[int], list[int]]]:
    matches = criterion.matcher(outputs, targets)
    if isinstance(matches, tuple):
        matches = matches[0]
    return [(source.tolist(), target.tolist()) for source, target in matches]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    targets = _targets()
    initial = _outputs()
    native_criterion = _native_criterion()
    external_criterion = _external_criterion()

    native_outputs = {name: value.clone().requires_grad_() for name, value in initial.items()}
    external_outputs = {name: value.clone().requires_grad_() for name, value in initial.items()}
    native_losses = native_criterion(native_outputs, targets)
    external_losses = external_criterion(external_outputs, targets)
    external_total = _weighted_external_loss(external_losses, external_criterion)

    native_matches = _matches(native_criterion, native_outputs, targets)
    external_matches = _matches(external_criterion, external_outputs, targets)
    if native_matches != external_matches:
        raise AssertionError(
            f"Hungarian matches differ: native={native_matches}, external={external_matches}"
        )

    names = ("loss_ce", "loss_bbox", "loss_giou", "loss_keypoints", "loss_oks")
    loss_errors = {}
    for name in names:
        error = (native_losses[name] - external_losses[name]).abs().item()
        loss_errors[name] = error
        if error > 5e-6:
            raise AssertionError(f"{name} differs by {error:.6g}")
    total_error = (native_losses["total"] - external_total).abs().item()
    if total_error > 5e-6:
        raise AssertionError(f"total differs by {total_error:.6g}")

    native_losses["total"].backward()
    external_total.backward()
    gradient_errors = {}
    for name in initial:
        native_gradient = native_outputs[name].grad
        external_gradient = external_outputs[name].grad
        if native_gradient is None or external_gradient is None:
            raise AssertionError(f"missing gradient for {name}")
        error = (native_gradient - external_gradient).abs().max().item()
        gradient_errors[name] = error
        if error > 5e-6:
            raise AssertionError(f"gradient for {name} differs by {error:.6g}")

    native_parameters = [torch.nn.Parameter(value.detach().clone()) for value in initial.values()]
    external_parameters = [torch.nn.Parameter(value.detach().clone()) for value in initial.values()]
    native_step_outputs = dict(zip(initial, native_parameters))
    external_step_outputs = dict(zip(initial, external_parameters))
    native_step_loss = native_criterion(native_step_outputs, targets)["total"]
    external_step_loss = _weighted_external_loss(
        external_criterion(external_step_outputs, targets), external_criterion
    )
    native_optimizer = torch.optim.SGD(native_parameters, lr=1e-3)
    external_optimizer = torch.optim.SGD(external_parameters, lr=1e-3)
    native_optimizer.zero_grad()
    external_optimizer.zero_grad()
    native_step_loss.backward()
    external_step_loss.backward()
    native_optimizer.step()
    external_optimizer.step()
    update_errors = {
        name: (native - external).abs().max().item()
        for name, native, external in zip(initial, native_parameters, external_parameters)
    }
    if max(update_errors.values()) > 5e-6:
        raise AssertionError(f"parameter updates differ: {update_errors}")

    print(f"matches: {native_matches}")
    print(f"loss_errors: {loss_errors}")
    print(f"total_error: {total_error:.6g}")
    print(f"gradient_errors: {gradient_errors}")
    print(f"update_errors: {update_errors}")
    print("one-step criterion/update parity: PASS")
    _run_native_model_step(torch.device(args.device))


if __name__ == "__main__":
    main()
