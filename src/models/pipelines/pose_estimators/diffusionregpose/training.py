"""Training targets, matching, and losses for DiffusionRegPose."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
import warnings

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pydantic import field_validator
from scipy.optimize import linear_sum_assignment
from src.configs.base_config import BaseConfig
from src.training.optim import build_adamw_with_schedule
from torch import Tensor, nn

from .evaluation import (
    DiffusionRegPoseAPMetric,
    DiffusionRegPoseBBoxAPMetric,
    box_cxcywh_to_xyxy,
    postprocess_crowdpose,
)
from .model import DiffusionRegPoseModel


CROWDPOSE_PERSON_LABEL = 1


def _generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute the reference GIoU formula, including its stabilizing epsilons."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    intersection_size = (right_bottom - left_top).clamp_min(0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    union = area1[:, None] + area2 - intersection
    iou = intersection / (union + 1e-6)

    enclosing_left_top = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    enclosing_right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    enclosing_size = (enclosing_right_bottom - enclosing_left_top).clamp_min(0)
    enclosing_area = enclosing_size[..., 0] * enclosing_size[..., 1]
    return iou - (enclosing_area - union) / (enclosing_area + 1e-6)


def diffusionregpose_ap_epoch_active(
    current_epoch: int,
    max_epochs: int | None,
    cadence: int,
) -> bool:
    """Return whether AP should be calculated after this zero-based epoch."""
    if cadence < 1:
        raise ValueError("AP cadence must be positive")
    completed_epoch = current_epoch + 1
    return completed_epoch % cadence == 0 or (
        max_epochs is not None and completed_epoch == max_epochs
    )


@torch.no_grad()
def _postprocess_diffusionregpose_ap(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    *,
    num_body_points: int = 14,
) -> list[dict[str, Tensor]]:
    """Match the sampled-model candidate path used by both AP evaluators."""
    images = batch["image"]
    target_sizes = batch.get("original_img_hw", batch["img_hw"]).to(
        images.device, dtype=torch.float32,
    )
    nms_target_sizes = batch.get("img_hw")
    if nms_target_sizes is None or nms_target_sizes.shape != target_sizes.shape:
        nms_target_sizes = torch.tensor(
            [images.shape[-2], images.shape[-1]],
            device=images.device,
            dtype=torch.float32,
        ).expand(images.shape[0], -1)
    else:
        nms_target_sizes = nms_target_sizes.to(images.device, dtype=torch.float32)
    return postprocess_crowdpose(
        outputs,
        target_sizes,
        num_body_points=num_body_points,
        nms_target_sizes=nms_target_sizes,
    )


@torch.no_grad()
def update_diffusionregpose_ap_metric(
    metric: DiffusionRegPoseAPMetric,
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    *,
    num_body_points: int = 14,
    results: list[dict[str, Tensor]] | None = None,
) -> None:
    """Add post-processed person predictions from one validation batch to AP."""
    if results is None:
        results = _postprocess_diffusionregpose_ap(
            outputs, batch, num_body_points=num_body_points,
        )
    for batch_index, result in enumerate(results):
        person = result["labels"] == CROWDPOSE_PERSON_LABEL
        serialized = result["keypoints"][person].reshape(-1, num_body_points, 3)
        scores = result["scores"][person]
        metric.update(
            serialized.unsqueeze(0),
            scores[:, None].expand(-1, num_body_points).unsqueeze(0),
            batch["image_id"][batch_index : batch_index + 1],
            bbox_scores=scores.unsqueeze(0),
        )


@torch.no_grad()
def update_diffusionregpose_bbox_ap_metric(
    metric: DiffusionRegPoseBBoxAPMetric,
    results: list[dict[str, Tensor]],
    batch: dict[str, Tensor],
) -> None:
    """Add person boxes from the shared pose-NMS result set to bbox AP."""
    for batch_index, result in enumerate(results):
        person = result["labels"] == CROWDPOSE_PERSON_LABEL
        metric.update(
            result["boxes"][person].unsqueeze(0),
            result["scores"][person].unsqueeze(0),
            result["labels"][person].unsqueeze(0),
            batch["image_id"][batch_index : batch_index + 1],
        )


def _keypoints_to_xyxyzz(keypoints: Tensor) -> Tensor:
    coordinates = keypoints[..., :2]
    visibility = keypoints[..., 2]
    # The model and external criterion use xyxy...vvv: coordinate pairs stay
    # interleaved, while visibility is stored in the final block.
    return torch.cat((coordinates.flatten(-2), visibility), dim=-1)


def build_diffusionregpose_targets(
    batch: dict[str, Tensor],
    diffusion: nn.Module,
    *,
    num_proposals: int,
    num_body_points: int,
    noise: Tensor | None = None,
    noise_seed: int | None = None,
) -> tuple[Tensor, Tensor, list[dict[str, Tensor]]]:
    """Build reference-format targets and timestep-zero diffusion inputs."""
    keypoints = batch["gt_keypoints"]
    diffusion_keypoints = batch.get("gt_diffusion_keypoints", keypoints)
    labeled = batch["gt_labeled"].bool()
    # The external CrowdPose converter maps both visible and occluded labels
    # to the binary supervision mask used by its criterion.
    target_visibility = labeled.float()
    boxes = batch["gt_boxes_canon"]
    batch_size, person_count, point_count, _ = keypoints.shape
    if point_count != num_body_points:
        raise ValueError("batch keypoint count does not match the model")
    if boxes.shape[:2] != (batch_size, person_count):
        raise ValueError("box and keypoint target counts must agree")
    if diffusion_keypoints.shape[:1] != (batch_size,) or diffusion_keypoints.shape[2:] != (
        point_count,
        2,
    ):
        raise ValueError("diffusion keypoints must match batch and keypoint dimensions")
    if num_proposals < 1:
        raise ValueError("num_proposals must be positive")

    generator = None
    if noise_seed is not None:
        generator = torch.Generator(device=keypoints.device)
        generator.manual_seed(noise_seed)

    def random_tensor(shape: tuple[int, ...]) -> Tensor:
        return torch.randn(
            shape,
            device=keypoints.device,
            dtype=keypoints.dtype,
            generator=generator,
        )

    targets: list[dict[str, Tensor]] = []
    diffusion_inputs = keypoints.new_empty(batch_size, num_proposals, num_body_points * 2)
    for batch_index in range(batch_size):
        valid_indices = labeled[batch_index].any(dim=-1).nonzero(as_tuple=False).flatten()
        valid_count = valid_indices.numel()
        target_keypoints = (
            keypoints[batch_index].index_select(0, valid_indices) + 1.0
        ) / 2.0
        target_visible = target_visibility[batch_index].index_select(0, valid_indices)
        target_boxes = boxes[batch_index].index_select(0, valid_indices)
        targets.append({
            "labels": torch.full(
                (valid_count,), CROWDPOSE_PERSON_LABEL,
                dtype=torch.long, device=keypoints.device,
            ),
            "boxes": target_boxes,
            "keypoints": _keypoints_to_xyxyzz(
                torch.cat((target_keypoints, target_visible.unsqueeze(-1)), dim=-1)
            ),
            "area": (target_boxes[:, 2] * target_boxes[:, 3]).clamp_min(1e-6),
        })

        if "gt_diffusion_keypoints" in batch:
            person_mask = batch.get("person_mask")
            if person_mask is None:
                diffusion_indices = torch.arange(
                    diffusion_keypoints.shape[1], device=keypoints.device
                )
            else:
                diffusion_indices = person_mask[batch_index].bool().nonzero(
                    as_tuple=False
                ).flatten()
            clean = (
                diffusion_keypoints[batch_index].index_select(0, diffusion_indices) + 1.0
            ) / 2.0
            clean = clean.flatten(1)
            count = min(clean.shape[0], num_proposals)
            diffusion_inputs[batch_index, :count] = (
                clean[:count] * 2.0 - 1.0
            ) * diffusion.scale
        elif valid_count:
            clean = target_keypoints[..., :2].flatten(1)
            count = min(valid_count, num_proposals)
            diffusion_inputs[batch_index, :count] = (clean[:count] * 2.0 - 1.0) * diffusion.scale
        else:
            count = 0
        if count < num_proposals:
            diffusion_inputs[batch_index, count:] = random_tensor((
                num_proposals - count, num_body_points * 2,
            )) * diffusion.scale

    timesteps = torch.zeros(batch_size, dtype=torch.long, device=keypoints.device)
    if noise is None:
        noise = random_tensor(tuple(diffusion_inputs.shape))
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
            use_pose_costs = bool(keypoints[..., : self.num_body_points * 2].sum().item() > 0)
            if use_pose_costs:
                box_cost = torch.zeros_like(torch.cdist(boxes, target["boxes"], p=1))
                giou_cost = torch.zeros_like(box_cost)
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
            else:
                box_cost = torch.cdist(boxes, target["boxes"], p=1)
                giou_cost = -_generalized_box_iou(
                    box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target["boxes"])
                )
                keypoint_cost = torch.zeros_like(box_cost)
                oks = torch.ones_like(box_cost)
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
        dn_label_coef: float = 0.3,
        dn_bbox_coef: float = 0.5,
        interm_loss_coef: float = 1.0,
        no_interm_box_loss: bool = False,
        weights: DiffusionRegPoseLossWeights | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_body_points = num_body_points
        self.num_box_decoder_layers = num_box_decoder_layers
        self.dn_label_coef = dn_label_coef
        self.dn_bbox_coef = dn_bbox_coef
        self.interm_loss_coef = interm_loss_coef
        self.no_interm_box_loss = no_interm_box_loss
        self.weights = weights or DiffusionRegPoseLossWeights()
        self.matcher = matcher or DiffusionRegPoseMatcher(
            num_body_points=num_body_points, weights=self.weights
        )

    def _loss_for_outputs(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        *,
        include_keypoints: bool = True,
        include_classification_diagnostics: bool = False,
    ) -> dict[str, Tensor]:
        matches = self.matcher(outputs, targets)
        logits = outputs["pred_logits"]
        target_classes = torch.zeros_like(logits)
        matched_queries = torch.zeros(
            logits.shape[:2], dtype=torch.bool, device=logits.device,
        )
        for batch_index, (source, target) in enumerate(matches):
            if source.numel():
                target_classes[batch_index, source, targets[batch_index]["labels"][target]] = 1.0
                matched_queries[batch_index, source] = True
        probabilities = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, target_classes, reduction="none")
        p_t = probabilities * target_classes + (1 - probabilities) * (1 - target_classes)
        alpha_t = self.weights.focal_alpha * target_classes + (1 - self.weights.focal_alpha) * (1 - target_classes)
        num_boxes = max(sum(target["labels"].numel() for target in targets), 1)
        focal_loss = alpha_t * ce * (1 - p_t).pow(2)
        # ``sigmoid_focal_loss`` in the reference already averages over the
        # class dimension, normalizes by the target count, and the reference
        # criterion multiplies that result by the query count once.
        loss_ce = focal_loss.mean(1).sum() / num_boxes * logits.shape[1]
        loss_bbox = logits.new_zeros(())
        loss_giou = logits.new_zeros(())
        loss_keypoints = logits.new_zeros(())
        loss_oks = logits.new_zeros(())
        normalizer = num_boxes
        for batch_index, (source, target_indices) in enumerate(matches):
            if source.numel() == 0:
                continue
            target = targets[batch_index]
            pred_boxes = outputs["pred_boxes"][batch_index, source]
            target_boxes = target["boxes"][target_indices]
            loss_bbox = loss_bbox + F.l1_loss(pred_boxes, target_boxes, reduction="sum")
            loss_giou = loss_giou + (1 - torch.diag(_generalized_box_iou(
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
            "loss_ce": loss_ce,
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
        if include_classification_diagnostics:
            person_scores = probabilities[..., CROWDPOSE_PERSON_LABEL]
            zero = logits.new_zeros(())
            values.update({
                "loss_ce_positive": (
                    (focal_loss * target_classes).mean(1).sum() / num_boxes
                    * logits.shape[1]
                ).detach(),
                "loss_ce_negative": (
                    (focal_loss * (1 - target_classes)).mean(1).sum() / num_boxes
                    * logits.shape[1]
                ).detach(),
                "person_score_matched": (
                    person_scores[matched_queries].mean()
                    if matched_queries.any() else zero
                ).detach(),
                "person_score_unmatched": (
                    person_scores[~matched_queries].mean()
                    if (~matched_queries).any() else zero
                ).detach(),
                "matched_queries": matched_queries.sum().to(logits.dtype).detach(),
            })
        return values

    def _weighted_clean_loss(self, values: dict[str, Tensor]) -> Tensor:
        """Apply the reference loss weights to one decoder output."""
        return (
            self.weights.cls * values["loss_ce"]
            + self.weights.bbox * values["loss_bbox"]
            + self.weights.giou * values["loss_giou"]
            + self.weights.keypoints * values["loss_keypoints"]
            + self.weights.oks * values["loss_oks"]
        )

    def _denoising_losses(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Calculate the external denoising classification and box losses."""
        logits = outputs["dn_class_pred"]
        labels = outputs["dn_class_input"]
        boxes = outputs["dn_bbox_pred"]
        target_boxes = outputs["dn_bbox_input"]
        target_classes = torch.zeros(
            *logits.shape[:2], logits.shape[-1], dtype=logits.dtype, device=logits.device
        )
        valid_labels = labels < logits.shape[-1]
        target_classes.scatter_(
            2,
            labels.clamp_max(logits.shape[-1] - 1).unsqueeze(-1),
            valid_labels.unsqueeze(-1).to(logits.dtype),
        )
        probabilities = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, target_classes, reduction="none")
        p_t = probabilities * target_classes + (1 - probabilities) * (1 - target_classes)
        alpha_t = self.weights.focal_alpha * target_classes + (1 - self.weights.focal_alpha) * (1 - target_classes)
        num_targets = max(logits.shape[1], 1)
        loss_ce = (alpha_t * ce * (1 - p_t).pow(2)).mean(1).sum() / num_targets * logits.shape[1]
        loss_bbox = F.l1_loss(boxes, target_boxes, reduction="sum") / num_targets
        loss_giou = (
            1 - torch.diag(_generalized_box_iou(
                box_cxcywh_to_xyxy(boxes.flatten(0, 1)),
                box_cxcywh_to_xyxy(target_boxes.flatten(0, 1)),
            ))
        ).sum() / num_targets
        return {
            "dn_loss_ce": loss_ce,
            "dn_loss_bbox": loss_bbox,
            "dn_loss_giou": loss_giou,
        }

    def forward(self, outputs: dict, targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        losses = self._loss_for_outputs(
            outputs, targets, include_classification_diagnostics=True,
        )
        if "dn_class_pred" in outputs:
            dn_losses = self._denoising_losses(outputs)
            losses.update(dn_losses)
            losses["total"] = losses["total"] + (
                self.dn_label_coef * dn_losses["dn_loss_ce"]
                + self.weights.bbox * self.dn_bbox_coef * dn_losses["dn_loss_bbox"]
                + self.weights.giou * self.dn_bbox_coef * dn_losses["dn_loss_giou"]
            )
        intermediate = outputs.get("interm_outputs")
        if intermediate is not None:
            intermediate_losses = self._loss_for_outputs(
                intermediate, targets, include_keypoints=False,
            )
            coefficients = {
                "loss_ce": self.weights.cls * self.interm_loss_coef,
                "loss_bbox": (
                    0.0 if self.no_interm_box_loss else self.weights.bbox * self.interm_loss_coef
                ),
                "loss_giou": (
                    0.0 if self.no_interm_box_loss else self.weights.giou * self.interm_loss_coef
                ),
            }
            for name, coefficient in coefficients.items():
                losses[f"{name}_interm"] = intermediate_losses[name]
                losses["total"] = losses["total"] + coefficient * intermediate_losses[name]
        for index, auxiliary in enumerate(outputs.get("aux_outputs", [])):
            auxiliary_losses = self._loss_for_outputs(
                auxiliary, targets, include_keypoints=index >= self.num_box_decoder_layers
            )
            for name, value in auxiliary_losses.items():
                if name != "total":
                    losses[f"{name}_{index}"] = value
                    coefficient = {
                        "loss_ce": self.weights.cls,
                        "loss_bbox": self.weights.bbox,
                        "loss_giou": self.weights.giou,
                        "loss_keypoints": self.weights.keypoints,
                        "loss_oks": self.weights.oks,
                    }[name]
                    losses["total"] = losses["total"] + coefficient * value
            if "dn_aux_outputs" in outputs:
                auxiliary_dn = self._denoising_losses({
                    "dn_class_pred": outputs["dn_aux_outputs"][index]["pred_logits"],
                    "dn_bbox_pred": outputs["dn_aux_outputs"][index]["pred_boxes"],
                    "dn_class_input": outputs["dn_class_input"],
                    "dn_bbox_input": outputs["dn_bbox_input"],
                })
                for name, value in auxiliary_dn.items():
                    losses[f"{name}_{index}"] = value
                    coefficient = self.dn_label_coef if name == "dn_loss_ce" else (
                        self.weights.bbox if name == "dn_loss_bbox" else self.weights.giou
                    ) * self.dn_bbox_coef
                    losses["total"] = losses["total"] + coefficient * value
        return losses


class DiffusionRegPosePipeline(pl.LightningModule):
    """Lightning training wrapper around the checkpoint-compatible model."""

    def __init__(
        self,
        model: DiffusionRegPoseModel,
        criterion: DiffusionRegPoseCriterion,
        *,
        lr: float = 2e-4,
        lr_backbone: float | None = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int | None = None,
        lr_warmup_epochs: int = 1,
        lr_min_frac: float = 1e-2,
        scheduler_type: str = "cosine",
        lr_decay_milestones: tuple[int, ...] = (),
        split_weight_decay: bool = True,
        val_ap_every_n_epochs: int = 10,
        val_ann_file: str | None = None,
        diffusion_noise_seed: int | None = None,
        deterministic_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_frac = lr_min_frac
        self.scheduler_type = scheduler_type
        self.lr_decay_milestones = lr_decay_milestones
        self.split_weight_decay = split_weight_decay
        if val_ap_every_n_epochs < 1:
            raise ValueError("val_ap_every_n_epochs must be positive")
        self.val_ap_every_n_epochs = val_ap_every_n_epochs
        self.diffusion_noise_seed = diffusion_noise_seed
        self.deterministic_seed = deterministic_seed
        self._val_ap = DiffusionRegPoseAPMetric(val_ann_file) if val_ann_file else None
        self._val_bbox_ap = (
            DiffusionRegPoseBBoxAPMetric(val_ann_file) if val_ann_file else None
        )
        self.save_hyperparameters(ignore=["model", "criterion"])

    def forward(
        self,
        images: Tensor,
        diffusion_keypoints: Tensor,
        timesteps: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        padding_mask: Tensor | None = None,
    ) -> dict:
        return self.model(images, diffusion_keypoints, timesteps, targets, padding_mask)

    def _losses_for_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        diffusion_keypoints, timesteps, targets = build_diffusionregpose_targets(
            batch,
            self.model,
            num_proposals=self.model.num_proposals,
            num_body_points=self.model.transformer.decoder.num_body_points,
            noise_seed=self.diffusion_noise_seed,
        )
        if self.deterministic_seed is None:
            rng_context = nullcontext()
        else:
            rng_context = torch.random.fork_rng(
                devices=[batch["image"].device] if batch["image"].is_cuda else [],
            )
        with rng_context:
            if self.deterministic_seed is not None:
                torch.manual_seed(self.deterministic_seed)
            outputs = self(
                batch["image"], diffusion_keypoints, timesteps, targets,
                batch.get("padding_mask"),
            )
        return self.criterion(outputs, targets)

    def _log_losses(self, prefix: str, losses: dict[str, Tensor], batch_size: int) -> None:
        classification_diagnostics = {
            "loss_ce_positive",
            "loss_ce_negative",
            "person_score_matched",
            "person_score_unmatched",
            "matched_queries",
        }
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
                metric_prefix = f"debug/{prefix}" if name in classification_diagnostics else prefix
                self.log(
                    f"{metric_prefix}/{name}", value.detach(), batch_size=batch_size,
                    on_step=False, on_epoch=True,
                )

    def _ap_epoch_active(self) -> bool:
        if self._trainer is not None and getattr(self.trainer, "sanity_checking", False):
            return False
        max_epochs = self.trainer.max_epochs if self._trainer is not None else self.max_epochs
        return self._val_ap is not None and diffusionregpose_ap_epoch_active(
            self.current_epoch, max_epochs, self.val_ap_every_n_epochs
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
        if self._ap_epoch_active():
            sampling_generator = None
            if self.deterministic_seed is not None:
                sampling_generator = torch.Generator(device=batch["image"].device)
                sampling_generator.manual_seed(self.deterministic_seed)
            sampling_kwargs = {"padding_mask": batch.get("padding_mask")}
            if sampling_generator is not None:
                sampling_kwargs["generator"] = sampling_generator
            sampled_outputs = self.model.sample_keypoints(batch["image"], **sampling_kwargs)
            ap_results = _postprocess_diffusionregpose_ap(
                sampled_outputs,
                batch,
                num_body_points=self.model.transformer.decoder.num_body_points,
            )
            update_diffusionregpose_ap_metric(
                self._val_ap,
                sampled_outputs,
                batch,
                num_body_points=self.model.transformer.decoder.num_body_points,
                results=ap_results,
            )
            assert self._val_bbox_ap is not None
            update_diffusionregpose_bbox_ap_metric(
                self._val_bbox_ap,
                ap_results,
                batch,
            )

    def on_validation_epoch_end(self) -> None:
        if not self._ap_epoch_active():
            return
        assert self._val_ap is not None
        assert self._val_bbox_ap is not None
        for metric in (self._val_ap, self._val_bbox_ap):
            for name, value in metric.compute().items():
                self.log(
                    f"ap-val/{name}",
                    value,
                    sync_dist=True,
                    prog_bar=name == "crowdpose/AP",
                )
        self._val_ap.reset()
        self._val_bbox_ap.reset()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        del batch_idx
        losses = self._losses_for_batch(batch)
        self._log_losses("test", losses, batch["image"].shape[0])

    def configure_optimizers(self):
        return build_adamw_with_schedule(
            self,
            lr=self.lr,
            lr_backbone=self.lr_backbone,
            weight_decay=self.weight_decay,
            lr_T_max=self.max_epochs,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
            scheduler_type=self.scheduler_type,
            lr_decay_milestones=self.lr_decay_milestones,
            split_weight_decay=self.split_weight_decay,
        )


class DiffusionRegPoseLossWeightsConfig(BaseConfig):
    cls: float = 2.0
    bbox: float = 5.0
    giou: float = 2.0
    keypoints: float = 10.0
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
    dn_number: int = 100
    dn_box_noise_scale: float = 0.4
    dn_label_noise_ratio: float = 0.5
    dn_label_coef: float = 0.3
    dn_bbox_coef: float = 0.5
    interm_loss_coef: float = 1.0
    no_interm_box_loss: bool = False
    freeze_backbone_stem_and_layer1: bool = True
    pretrained_checkpoint: str | None = None
    lr: float = 2e-4
    lr_backbone: float | None = 1e-4
    weight_decay: float = 1e-4
    lr_warmup_epochs: int = 0
    lr_min_frac: float = 1e-2
    scheduler_type: str = "multistep"
    lr_decay_milestones: tuple[int, ...] = (31, 61)
    split_weight_decay: bool = False
    val_ap_every_n_epochs: int = 10
    diffusion_noise_seed: int | None = None
    deterministic_seed: int | None = None
    weights: DiffusionRegPoseLossWeightsConfig = DiffusionRegPoseLossWeightsConfig()

    @field_validator("num_classes")
    @classmethod
    def _validate_person_class_is_present(cls, value: int) -> int:
        if value <= CROWDPOSE_PERSON_LABEL:
            raise ValueError(
                "num_classes must include CrowdPose person label "
                f"{CROWDPOSE_PERSON_LABEL}, got {value}"
            )
        return value

    def build(
        self,
        *,
        max_epochs: int | None = None,
        val_ann_file: str | None = None,
    ) -> DiffusionRegPosePipeline:
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
            dn_number=self.dn_number,
            dn_box_noise_scale=self.dn_box_noise_scale,
            dn_label_noise_ratio=self.dn_label_noise_ratio,
            freeze_backbone_stem_and_layer1=self.freeze_backbone_stem_and_layer1,
        )
        if self.pretrained_checkpoint:
            checkpoint_path = Path(self.pretrained_checkpoint).expanduser()
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            if not isinstance(state, dict):
                raise ValueError(f"checkpoint does not contain a state dictionary: {checkpoint_path}")
            state = {
                key.removeprefix("module."): value
                for key, value in state.items()
                if isinstance(value, Tensor)
            }
            model_state = model.state_dict()
            compatible_state = {
                key: value
                for key, value in state.items()
                if key in model_state and model_state[key].shape == value.shape
            }
            skipped_keys = set(state) - set(compatible_state)
            incompatible = model.load_state_dict(compatible_state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys or skipped_keys:
                warnings.warn(
                    f"Loaded {checkpoint_path} with {len(incompatible.missing_keys)} missing and "
                    f"{len(incompatible.unexpected_keys)} unexpected and {len(skipped_keys)} "
                    "shape-incompatible keys.",
                    stacklevel=2,
                )
        weights = self.weights.to_dataclass()
        return DiffusionRegPosePipeline(
            model,
            DiffusionRegPoseCriterion(
                num_classes=self.num_classes,
                num_body_points=self.num_body_points,
                num_box_decoder_layers=self.num_box_decoder_layers,
                dn_label_coef=self.dn_label_coef,
                dn_bbox_coef=self.dn_bbox_coef,
                interm_loss_coef=self.interm_loss_coef,
                no_interm_box_loss=self.no_interm_box_loss,
                weights=weights,
            ),
            lr=self.lr,
            lr_backbone=self.lr_backbone,
            weight_decay=self.weight_decay,
            max_epochs=max_epochs,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
            scheduler_type=self.scheduler_type,
            lr_decay_milestones=self.lr_decay_milestones,
            split_weight_decay=self.split_weight_decay,
            val_ap_every_n_epochs=self.val_ap_every_n_epochs,
            val_ann_file=val_ann_file,
            diffusion_noise_seed=self.diffusion_noise_seed,
            deterministic_seed=self.deterministic_seed,
        )
