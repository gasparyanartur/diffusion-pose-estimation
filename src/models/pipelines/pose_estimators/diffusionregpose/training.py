"""Training targets, matching, and losses for DiffusionRegPose."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.optimize import linear_sum_assignment
from src.configs.base_config import BaseConfig
from src.training.optim import build_adamw_with_schedule
from torch import Tensor, nn
from torchvision.ops import generalized_box_iou

from .evaluation import box_cxcywh_to_xyxy
from .model import DiffusionRegPoseModel


def _keypoints_to_xyxyzz(keypoints: Tensor) -> Tensor:
    coordinates = keypoints[..., :2].flatten(-2)
    visibility = keypoints[..., 2]
    return torch.cat((coordinates, visibility), dim=-1)


def build_diffusionregpose_targets(
    batch: dict[str, Tensor],
    diffusion: nn.Module,
    *,
    num_proposals: int,
    num_body_points: int,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor, list[dict[str, Tensor]]]:
    """Build reference-format targets and timestep-zero diffusion inputs."""
    keypoints = batch["gt_keypoints"]
    labeled = batch["gt_labeled"].bool()
    visible = batch["gt_visible"].float()
    boxes = batch["gt_boxes_canon"]
    batch_size, person_count, point_count, _ = keypoints.shape
    if point_count != num_body_points:
        raise ValueError("batch keypoint count does not match the model")
    if boxes.shape[:2] != (batch_size, person_count):
        raise ValueError("box and keypoint target counts must agree")
    if num_proposals < 1:
        raise ValueError("num_proposals must be positive")

    targets: list[dict[str, Tensor]] = []
    diffusion_inputs = keypoints.new_empty(batch_size, num_proposals, num_body_points * 2)
    for batch_index in range(batch_size):
        valid_count = int(labeled[batch_index].any(dim=-1).sum().item())
        valid_count = min(valid_count, person_count)
        target_keypoints = (keypoints[batch_index, :valid_count] + 1.0) / 2.0
        target_visible = visible[batch_index, :valid_count]
        target_boxes = boxes[batch_index, :valid_count]
        targets.append({
            "labels": torch.zeros(valid_count, dtype=torch.long, device=keypoints.device),
            "boxes": target_boxes,
            "keypoints": _keypoints_to_xyxyzz(
                torch.cat((target_keypoints, target_visible.unsqueeze(-1)), dim=-1)
            ),
            "area": (target_boxes[:, 2] * target_boxes[:, 3]).clamp_min(1e-6),
        })

        if valid_count:
            clean = target_keypoints[..., :2].flatten(1)
            count = min(valid_count, num_proposals)
            diffusion_inputs[batch_index, :count] = (clean[:count] * 2.0 - 1.0) * diffusion.scale
        else:
            count = 0
        if count < num_proposals:
            diffusion_inputs[batch_index, count:] = torch.randn(
                num_proposals - count,
                num_body_points * 2,
                device=keypoints.device,
                dtype=keypoints.dtype,
            ) * diffusion.scale

    timesteps = torch.zeros(batch_size, dtype=torch.long, device=keypoints.device)
    diffusion_inputs = diffusion.q_sample(diffusion_inputs, timesteps, noise=noise).to(keypoints.dtype)
    return diffusion_inputs.clamp(-diffusion.scale, diffusion.scale), timesteps, targets


@dataclass(frozen=True)
class DiffusionRegPoseLossWeights:
    cls: float = 2.0
    bbox: float = 5.0
    giou: float = 2.0
    keypoints: float = 1.0
    oks: float = 4.0
    set_cost_class: float = 2.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0
    set_cost_keypoints: float = 10.0
    set_cost_oks: float = 4.0
    focal_alpha: float = 0.25


class DiffusionRegPoseMatcher(nn.Module):
    """Hungarian matcher matching the reference classification/box/pose costs."""

    def __init__(
        self,
        *,
        num_body_points: int = 14,
        weights: DiffusionRegPoseLossWeights | None = None,
    ) -> None:
        super().__init__()
        self.num_body_points = num_body_points
        self.weights = weights or DiffusionRegPoseLossWeights()
        sigmas = [0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89, 0.79, 0.79]
        if num_body_points > len(sigmas):
            raise ValueError("DiffusionRegPose supports at most 14 keypoints")
        self.register_buffer("sigmas", torch.tensor(sigmas[:num_body_points], dtype=torch.float32) / 10.0)

    @torch.no_grad()
    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> list[tuple[Tensor, Tensor]]:
        matches: list[tuple[Tensor, Tensor]] = []
        for logits, boxes, keypoints, target in zip(
            outputs["pred_logits"], outputs["pred_boxes"], outputs["pred_keypoints"], targets
        ):
            target_labels = target["labels"]
            if target_labels.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=logits.device)
                matches.append((empty, empty))
                continue
            probabilities = logits.sigmoid()
            alpha, gamma = self.weights.focal_alpha, 2.0
            neg = (1 - alpha) * probabilities.pow(gamma) * -(1 - probabilities + 1e-8).log()
            pos = alpha * (1 - probabilities).pow(gamma) * -(probabilities + 1e-8).log()
            class_cost = pos[:, target_labels] - neg[:, target_labels]
            box_cost = torch.cdist(boxes, target["boxes"], p=1)
            giou_cost = -generalized_box_iou(
                box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target["boxes"])
            )
            pred_xy = keypoints[:, : self.num_body_points * 2].reshape(-1, self.num_body_points, 2)
            target_xy = target["keypoints"][:, : self.num_body_points * 2].reshape(-1, self.num_body_points, 2)
            target_visible = target["keypoints"][:, self.num_body_points * 2 :]
            distance = (pred_xy[:, None] - target_xy[None]).pow(2).sum(-1)
            variances = (self.sigmas.to(pred_xy) * 2).pow(2)
            oks = torch.exp(-distance / (target["area"][None, :, None] * variances[None, None] * 2))
            oks = (oks * target_visible[None]).sum(-1) / (target_visible.sum(-1)[None] + 1e-6)
            keypoint_cost = (
                (keypoints[:, None, : self.num_body_points * 2] - target["keypoints"][None, :, : self.num_body_points * 2]).abs()
                * target_visible[None].repeat_interleave(2, dim=-1)
            ).sum(-1)
            cost = (
                self.weights.set_cost_class * class_cost
                + self.weights.set_cost_bbox * box_cost
                + self.weights.set_cost_giou * giou_cost
                + self.weights.set_cost_keypoints * keypoint_cost
                + self.weights.set_cost_oks * (1 - oks.clamp_min(1e-6))
            )
            row, column = linear_sum_assignment(cost.detach().cpu())
            matches.append((torch.as_tensor(row, device=logits.device), torch.as_tensor(column, device=logits.device)))
        return matches


class DiffusionRegPoseCriterion(nn.Module):
    """Reference-compatible final and auxiliary set-prediction losses."""

    def __init__(
        self,
        *,
        num_classes: int = 2,
        matcher: DiffusionRegPoseMatcher | None = None,
        num_body_points: int = 14,
        num_box_decoder_layers: int = 2,
        weights: DiffusionRegPoseLossWeights | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_body_points = num_body_points
        self.num_box_decoder_layers = num_box_decoder_layers
        self.weights = weights or DiffusionRegPoseLossWeights()
        self.matcher = matcher or DiffusionRegPoseMatcher(
            num_body_points=num_body_points, weights=self.weights
        )

    def _loss_for_outputs(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]], *, include_keypoints: bool = True) -> dict[str, Tensor]:
        matches = self.matcher(outputs, targets)
        logits = outputs["pred_logits"]
        target_classes = torch.zeros_like(logits)
        for batch_index, (source, target) in enumerate(matches):
            if source.numel():
                target_classes[batch_index, source, targets[batch_index]["labels"][target]] = 1.0
        probabilities = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, target_classes, reduction="none")
        p_t = probabilities * target_classes + (1 - probabilities) * (1 - target_classes)
        alpha_t = self.weights.focal_alpha * target_classes + (1 - self.weights.focal_alpha) * (1 - target_classes)
        loss_ce = (alpha_t * ce * (1 - p_t).pow(2)).mean(2).sum(1).mean() / max(logits.shape[1], 1)
        loss_bbox = logits.new_zeros(())
        loss_giou = logits.new_zeros(())
        loss_keypoints = logits.new_zeros(())
        loss_oks = logits.new_zeros(())
        matched_count = sum(source.numel() for source, _ in matches)
        normalizer = max(matched_count, 1)
        for batch_index, (source, target_indices) in enumerate(matches):
            if source.numel() == 0:
                continue
            target = targets[batch_index]
            pred_boxes = outputs["pred_boxes"][batch_index, source]
            target_boxes = target["boxes"][target_indices]
            loss_bbox = loss_bbox + F.l1_loss(pred_boxes, target_boxes, reduction="sum")
            loss_giou = loss_giou + (1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes)
            ))).sum()
            if include_keypoints:
                pred_keypoints = outputs["pred_keypoints"][batch_index, source]
                target_keypoints = target["keypoints"][target_indices]
                valid = target_keypoints[:, self.num_body_points * 2 :]
                coordinate_error = (pred_keypoints[:, : self.num_body_points * 2] - target_keypoints[:, : self.num_body_points * 2]).abs()
                loss_keypoints = loss_keypoints + (coordinate_error * valid.repeat_interleave(2, dim=-1)).sum()
                pred_xy = pred_keypoints[:, : self.num_body_points * 2].reshape(-1, self.num_body_points, 2)
                target_xy = target_keypoints[:, : self.num_body_points * 2].reshape(-1, self.num_body_points, 2)
                variance = (self.matcher.sigmas.to(pred_xy) * 2).pow(2)
                oks = torch.exp(-(pred_xy - target_xy).pow(2).sum(-1) / (target["area"][target_indices, None] * variance[None] * 2))
                loss_oks = loss_oks + (1 - (oks * valid).sum(-1) / (valid.sum(-1) + 1e-6)).sum()
        values = {
            "loss_ce": loss_ce * logits.shape[1],
            "loss_bbox": loss_bbox / normalizer,
            "loss_giou": loss_giou / normalizer,
            "loss_keypoints": loss_keypoints / normalizer,
            "loss_oks": loss_oks / normalizer,
        }
        values["total"] = (
            self.weights.cls * values["loss_ce"]
            + self.weights.bbox * values["loss_bbox"]
            + self.weights.giou * values["loss_giou"]
            + self.weights.keypoints * values["loss_keypoints"]
            + self.weights.oks * values["loss_oks"]
        )
        return values

    def forward(self, outputs: dict, targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        losses = self._loss_for_outputs(outputs, targets)
        for index, auxiliary in enumerate(outputs.get("aux_outputs", [])):
            auxiliary_losses = self._loss_for_outputs(
                auxiliary, targets, include_keypoints=index >= self.num_box_decoder_layers
            )
            for name, value in auxiliary_losses.items():
                if name != "total":
                    losses[f"{name}_{index}"] = value
                losses["total"] = losses["total"] + value
        return losses


class DiffusionRegPosePipeline(pl.LightningModule):
    """Lightning training wrapper around the checkpoint-compatible model."""

    def __init__(
        self,
        model: DiffusionRegPoseModel,
        criterion: DiffusionRegPoseCriterion,
        *,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        max_epochs: int | None = None,
        lr_warmup_epochs: int = 1,
        lr_min_frac: float = 1e-2,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_frac = lr_min_frac
        self.save_hyperparameters(ignore=["model", "criterion"])

    def forward(self, images: Tensor, diffusion_keypoints: Tensor, timesteps: Tensor) -> dict:
        return self.model(images, diffusion_keypoints, timesteps)

    def _losses_for_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        diffusion_keypoints, timesteps, targets = build_diffusionregpose_targets(
            batch,
            self.model,
            num_proposals=self.model.num_proposals,
            num_body_points=self.model.transformer.decoder.num_body_points,
        )
        outputs = self(batch["image"], diffusion_keypoints, timesteps)
        return self.criterion(outputs, targets)

    def _log_losses(self, prefix: str, losses: dict[str, Tensor], batch_size: int) -> None:
        self.log(
            f"{prefix}/loss",
            losses["total"],
            prog_bar=True,
            batch_size=batch_size,
            on_step=prefix == "train",
            on_epoch=True,
        )
        for name, value in losses.items():
            if name != "total":
                self.log(
                    f"{prefix}/{name}", value.detach(), batch_size=batch_size,
                    on_step=False, on_epoch=True,
                )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        losses = self._losses_for_batch(batch)
        self._log_losses("train", losses, batch["image"].shape[0])
        return losses["total"]

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        del batch_idx
        losses = self._losses_for_batch(batch)
        self._log_losses("val", losses, batch["image"].shape[0])

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        del batch_idx
        losses = self._losses_for_batch(batch)
        self._log_losses("test", losses, batch["image"].shape[0])

    def configure_optimizers(self):
        return build_adamw_with_schedule(
            self,
            lr=self.lr,
            weight_decay=self.weight_decay,
            lr_T_max=self.max_epochs,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
        )


class DiffusionRegPoseLossWeightsConfig(BaseConfig):
    cls: float = 2.0
    bbox: float = 5.0
    giou: float = 2.0
    keypoints: float = 1.0
    oks: float = 4.0
    set_cost_class: float = 2.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0
    set_cost_keypoints: float = 10.0
    set_cost_oks: float = 4.0
    focal_alpha: float = 0.25

    def to_dataclass(self) -> DiffusionRegPoseLossWeights:
        return DiffusionRegPoseLossWeights(**self.model_dump())


class DiffusionRegPosePipelineConfig(BaseConfig):
    """Hydra configuration for the native training wrapper."""

    backbone_pretrained: bool = True
    timesteps: int = 1000
    sampling_timesteps: int = 1
    num_queries: int = 900
    num_proposals: int = 100
    num_classes: int = 2
    num_feature_levels: int = 4
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    hidden_dim: int = 256
    num_heads: int = 8
    num_body_points: int = 14
    num_box_decoder_layers: int = 2
    num_group: int = 100
    lr: float = 2e-4
    weight_decay: float = 1e-4
    lr_warmup_epochs: int = 1
    lr_min_frac: float = 1e-2
    weights: DiffusionRegPoseLossWeightsConfig = DiffusionRegPoseLossWeightsConfig()

    def build(self, *, max_epochs: int | None = None) -> DiffusionRegPosePipeline:
        model = DiffusionRegPoseModel(
            backbone_pretrained=self.backbone_pretrained,
            timesteps=self.timesteps,
            sampling_timesteps=self.sampling_timesteps,
            num_queries=self.num_queries,
            num_proposals=self.num_proposals,
            num_classes=self.num_classes,
            num_feature_levels=self.num_feature_levels,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_body_points=self.num_body_points,
            num_box_decoder_layers=self.num_box_decoder_layers,
            num_group=self.num_group,
        )
        weights = self.weights.to_dataclass()
        return DiffusionRegPosePipeline(
            model,
            DiffusionRegPoseCriterion(
                num_classes=self.num_classes,
                num_body_points=self.num_body_points,
                num_box_decoder_layers=self.num_box_decoder_layers,
                weights=weights,
            ),
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_epochs=max_epochs,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
        )
