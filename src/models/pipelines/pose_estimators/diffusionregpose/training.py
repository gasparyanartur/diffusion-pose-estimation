"""Training targets, matching, and losses for DiffusionRegPose."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
from typing import Literal
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

_CRITICAL_CHECKPOINT_PREFIXES = {
    "input projections": ("input_proj.",),
    "level embeddings": ("transformer.level_embed",),
    "transformer encoder": ("transformer.encoder.",),
    "transformer decoder": ("transformer.decoder.",),
    "prediction heads": (
        "bbox_embed.",
        "class_embed.",
        "pose_embed.",
        "pose_hw_embed.",
    ),
}


def _normalize_kept_layer_indices(
    indices: Sequence[int],
    *,
    target_count: int,
    source_count: int,
    component: str,
) -> tuple[int, ...]:
    if len(indices) != target_count:
        raise ValueError(
            f"{component} keep-layers must contain {target_count} one-based "
            f"indices; got {len(indices)}"
        )
    if len(set(indices)) != len(indices):
        raise ValueError(f"{component} keep-layers must not contain duplicates")
    if any(index < 1 or index > source_count for index in indices):
        raise ValueError(
            f"{component} keep-layers must be between 1 and {source_count}; "
            f"got {tuple(indices)}"
        )
    return tuple(index - 1 for index in indices)


def remap_diffusionregpose_checkpoint_layers(
    checkpoint_state: Mapping[str, Tensor],
    *,
    encoder_layer_indices: Sequence[int],
    decoder_layer_indices: Sequence[int],
    target_num_encoder_layers: int,
    target_num_decoder_layers: int,
    target_num_box_decoder_layers: int,
    source_num_encoder_layers: int = 6,
    source_num_decoder_layers: int = 6,
    source_num_box_decoder_layers: int = 2,
) -> dict[str, Tensor]:
    """Map one-based retained EDPose layers into a reduced model state dict."""
    if target_num_box_decoder_layers != 1:
        raise ValueError(
            "selected decoder initialization currently requires one target "
            "box decoder layer"
        )
    encoder_indices = _normalize_kept_layer_indices(
        encoder_layer_indices,
        target_count=target_num_encoder_layers,
        source_count=source_num_encoder_layers,
        component="encoder",
    )
    decoder_indices = _normalize_kept_layer_indices(
        decoder_layer_indices,
        target_count=target_num_decoder_layers,
        source_count=source_num_decoder_layers,
        component="decoder",
    )
    source_box_indices = [
        index for index in decoder_indices if index < source_num_box_decoder_layers
    ]
    source_pose_indices = [
        index - source_num_box_decoder_layers
        for index in decoder_indices
        if index >= source_num_box_decoder_layers
    ]
    if len(source_box_indices) != target_num_box_decoder_layers:
        raise ValueError(
            "selected decoder layers must retain exactly one source box stage; "
            f"got zero-based source box indices {source_box_indices}"
        )
    target_pose_count = target_num_decoder_layers - target_num_box_decoder_layers
    if len(source_pose_indices) != target_pose_count:
        raise ValueError(
            "selected decoder layers do not provide the target number of pose "
            f"stages: expected {target_pose_count}, got {len(source_pose_indices)}"
        )

    remapped: dict[str, Tensor] = {}
    for key, value in checkpoint_state.items():
        normalized_key = key.removeprefix("module.")
        target_key = normalized_key
        for prefix, selected_indices in (
            ("transformer.encoder.layers", encoder_indices),
            ("transformer.decoder.layers", decoder_indices),
        ):
            marker = f"{prefix}."
            if normalized_key.startswith(marker):
                remainder = normalized_key[len(marker):]
                source_index_text, separator, suffix = remainder.partition(".")
                if not separator:
                    break
                source_index = int(source_index_text)
                if source_index not in selected_indices:
                    target_key = ""
                    break
                target_index = selected_indices.index(source_index)
                target_key = f"{prefix}.{target_index}.{suffix}"
                break
        if not target_key:
            continue

        for head_name in ("bbox_embed", "class_embed"):
            for prefix in ("", "transformer.decoder."):
                marker = f"{prefix}{head_name}."
                if not normalized_key.startswith(marker):
                    continue
                remainder = normalized_key[len(marker):]
                source_index_text, separator, suffix = remainder.partition(".")
                if not separator:
                    continue
                source_index = int(source_index_text)
                if source_index not in decoder_indices:
                    target_key = ""
                    break
                target_index = decoder_indices.index(source_index)
                target_key = f"{prefix}{head_name}.{target_index}.{suffix}"
                break
            if not target_key:
                break
        if not target_key:
            continue

        for head_name in ("pose_embed", "pose_hw_embed"):
            for prefix in ("", "transformer.decoder."):
                marker = f"{prefix}{head_name}."
                if not normalized_key.startswith(marker):
                    continue
                remainder = normalized_key[len(marker):]
                source_index_text, separator, suffix = remainder.partition(".")
                if not separator:
                    continue
                source_index = int(source_index_text)
                if source_index == 0 and head_name == "pose_embed":
                    target_key = f"{prefix}expansion_pose_embed.{suffix}"
                elif source_index in source_pose_indices:
                    target_index = source_pose_indices.index(source_index)
                    target_key = f"{prefix}{head_name}.{target_index}.{suffix}"
                else:
                    target_key = ""
                break
            if not target_key:
                break

        if target_key:
            remapped[target_key] = value
    return remapped


def diffusionregpose_checkpoint_component_coverage(
    model_state: Mapping[str, Tensor],
    checkpoint_state: Mapping[str, Tensor],
) -> dict[str, float]:
    """Measure shape-compatible checkpoint elements in critical model components."""
    coverage = {}
    for component, prefixes in _CRITICAL_CHECKPOINT_PREFIXES.items():
        component_state = {
            name: value
            for name, value in model_state.items()
            if name.startswith(prefixes)
        }
        total_elements = sum(value.numel() for value in component_state.values())
        if total_elements == 0:
            continue
        compatible_elements = sum(
            value.numel()
            for name, value in component_state.items()
            if name in checkpoint_state
            and checkpoint_state[name].shape == value.shape
        )
        coverage[component] = compatible_elements / total_elements
    return coverage


def validate_diffusionregpose_checkpoint_coverage(
    coverage: Mapping[str, float],
    *,
    minimum: float,
) -> None:
    """Reject pretrained checkpoints that leave critical components uninitialized."""
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum checkpoint coverage must be between zero and one")
    insufficient = {
        component: value
        for component, value in coverage.items()
        if value < minimum
    }
    if not insufficient:
        return
    details = ", ".join(
        f"{component}={value:.2%}"
        for component, value in insufficient.items()
    )
    raise ValueError(
        f"pretrained checkpoint critical coverage is below {minimum:.2%}: {details}"
    )


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
            person_mask = batch.get("diffusion_person_mask", batch.get("person_mask"))
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


def _pairwise_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    intersection_left_top = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    intersection_right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    intersection_size = (intersection_right_bottom - intersection_left_top).clamp_min(0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    )
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    )
    union = area1[:, None] + area2 - intersection
    return intersection / union.clamp_min(torch.finfo(intersection.dtype).eps)


def _diffusionregpose_pairwise_matching_cost(
    logits: Tensor,
    boxes: Tensor,
    keypoints: Tensor,
    target: dict[str, Tensor],
    *,
    num_body_points: int,
    sigmas: Tensor,
    weights: DiffusionRegPoseLossWeights,
) -> Tensor:
    probabilities = logits.sigmoid()
    alpha, gamma = weights.focal_alpha, 2.0
    negative_class_cost = (
        (1 - alpha)
        * probabilities.pow(gamma)
        * -(1 - probabilities + 1e-8).log()
    )
    positive_class_cost = (
        alpha
        * (1 - probabilities).pow(gamma)
        * -(probabilities + 1e-8).log()
    )
    class_cost = (
        positive_class_cost[:, target["labels"]]
        - negative_class_cost[:, target["labels"]]
    )

    coordinate_count = num_body_points * 2
    use_pose_costs = bool(keypoints[..., :coordinate_count].sum().item() > 0)
    if not use_pose_costs:
        box_cost = torch.cdist(boxes, target["boxes"], p=1)
        giou_cost = -_generalized_box_iou(
            box_cxcywh_to_xyxy(boxes),
            box_cxcywh_to_xyxy(target["boxes"]),
        )
        keypoint_cost = torch.zeros_like(box_cost)
        oks = torch.ones_like(box_cost)
    else:
        box_cost = torch.zeros_like(torch.cdist(boxes, target["boxes"], p=1))
        giou_cost = torch.zeros_like(box_cost)
        predicted_xy = keypoints[:, :coordinate_count].reshape(
            -1, num_body_points, 2,
        )
        target_xy = target["keypoints"][:, :coordinate_count].reshape(
            -1, num_body_points, 2,
        )
        target_visible = target["keypoints"][:, coordinate_count:]
        squared_distance = (
            predicted_xy[:, None] - target_xy[None]
        ).pow(2).sum(-1)
        variances = (sigmas.to(predicted_xy) * 2).pow(2)
        oks = torch.exp(
            -squared_distance
            / (target["area"][None, :, None] * variances[None, None] * 2)
        )
        oks = (
            (oks * target_visible[None]).sum(-1)
            / (target_visible.sum(-1)[None] + 1e-6)
        )
        keypoint_cost = (
            (
                keypoints[:, None, :coordinate_count]
                - target["keypoints"][None, :, :coordinate_count]
            ).abs()
            * target_visible[None].repeat_interleave(2, dim=-1)
        ).sum(-1)

    return (
        weights.set_cost_class * class_cost
        + weights.set_cost_bbox * box_cost
        + weights.set_cost_giou * giou_cost
        + weights.set_cost_keypoints * keypoint_cost
        + weights.set_cost_oks * (1 - oks.clamp_min(1e-6))
    )


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

    def normalization_count(
        self,
        matches: list[tuple[Tensor, Tensor]],
        targets: list[dict[str, Tensor]],
    ) -> int:
        del matches
        return sum(target["labels"].numel() for target in targets)

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
            cost = _diffusionregpose_pairwise_matching_cost(
                logits,
                boxes,
                keypoints,
                target,
                num_body_points=self.num_body_points,
                sigmas=self.sigmas,
                weights=self.weights,
            )
            row, column = linear_sum_assignment(cost.detach().cpu())
            matches.append((torch.as_tensor(row, device=logits.device), torch.as_tensor(column, device=logits.device)))
        return matches


_OTA_BOX_AND_CENTER_PENALTY = 100.0
_OTA_OUTSIDE_CANDIDATE_PENALTY = 10_000.0


class DiffusionRegPoseOTAMatcher(DiffusionRegPoseMatcher):
    """Assign complete person queries with box-IoU Dynamic-K matching."""

    def __init__(
        self,
        *,
        num_body_points: int = 14,
        weights: DiffusionRegPoseLossWeights | None = None,
        candidate_topk: int = 5,
        center_radius: float = 2.5,
    ) -> None:
        super().__init__(num_body_points=num_body_points, weights=weights)
        if candidate_topk < 1:
            raise ValueError("OTA candidate_topk must be positive")
        if center_radius <= 0:
            raise ValueError("OTA center_radius must be positive")
        self.candidate_topk = candidate_topk
        self.center_radius = center_radius

    def normalization_count(
        self,
        matches: list[tuple[Tensor, Tensor]],
        targets: list[dict[str, Tensor]],
    ) -> int:
        del targets
        return sum(source_indices.numel() for source_indices, _ in matches)

    def _candidate_penalties(self, boxes: Tensor, target_boxes: Tensor) -> Tensor:
        query_centers = boxes[:, :2]
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        in_boxes = (
            (query_centers[:, None, 0] > target_xyxy[None, :, 0])
            & (query_centers[:, None, 0] < target_xyxy[None, :, 2])
            & (query_centers[:, None, 1] > target_xyxy[None, :, 1])
            & (query_centers[:, None, 1] < target_xyxy[None, :, 3])
        )

        target_centers = target_boxes[:, :2]
        center_half_size = target_boxes[:, 2:] * self.center_radius
        center_left_top = target_centers - center_half_size
        center_right_bottom = target_centers + center_half_size
        in_centers = (
            (query_centers[:, None, 0] > center_left_top[None, :, 0])
            & (query_centers[:, None, 0] < center_right_bottom[None, :, 0])
            & (query_centers[:, None, 1] > center_left_top[None, :, 1])
            & (query_centers[:, None, 1] < center_right_bottom[None, :, 1])
        )

        in_any_candidate_region = (in_boxes | in_centers).any(dim=1)
        penalties = (~(in_boxes & in_centers)).to(boxes.dtype)
        penalties = penalties * _OTA_BOX_AND_CENTER_PENALTY
        penalties[~in_any_candidate_region] += _OTA_OUTSIDE_CANDIDATE_PENALTY
        return penalties

    def _dynamic_k_assignment(
        self,
        cost: Tensor,
        pairwise_iou: Tensor,
    ) -> tuple[Tensor, Tensor]:
        query_count = cost.shape[0]
        candidate_count = min(self.candidate_topk, query_count)
        top_ious = pairwise_iou.topk(candidate_count, dim=0).values
        dynamic_counts = top_ious.sum(dim=0).to(torch.long).clamp_min(1)

        matching = torch.zeros_like(cost, dtype=torch.bool)
        for target_index, dynamic_count in enumerate(dynamic_counts.tolist()):
            selected = cost[:, target_index].topk(
                dynamic_count, largest=False,
            ).indices
            matching[selected, target_index] = True

        conflicted = matching.sum(dim=1) > 1
        if conflicted.any():
            best_targets = cost[conflicted].argmin(dim=1)
            matching[conflicted] = False
            matching[conflicted, best_targets] = True

        while (matching.sum(dim=0) == 0).any():
            missing_target = (matching.sum(dim=0) == 0).nonzero(
                as_tuple=False,
            )[0, 0]
            matched_queries = matching.any(dim=1)
            movable_queries = ~matched_queries
            if matched_queries.any():
                assigned_targets = matching[matched_queries].to(torch.long).argmax(dim=1)
                assigned_counts = matching.sum(dim=0)
                movable_queries[matched_queries] = assigned_counts[assigned_targets] > 1
            candidates = movable_queries.nonzero(as_tuple=False).flatten()
            if candidates.numel() == 0:
                raise RuntimeError("OTA could not preserve one assignment per target")
            selected = candidates[cost[candidates, missing_target].argmin()]
            matching[selected] = False
            matching[selected, missing_target] = True

        source_indices = matching.any(dim=1).nonzero(as_tuple=False).flatten()
        target_indices = matching[source_indices].to(torch.long).argmax(dim=1)
        return source_indices, target_indices

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> list[tuple[Tensor, Tensor]]:
        matches: list[tuple[Tensor, Tensor]] = []
        for logits, boxes, keypoints, target in zip(
            outputs["pred_logits"],
            outputs["pred_boxes"],
            outputs["pred_keypoints"],
            targets,
        ):
            target_count = target["labels"].numel()
            if target_count == 0:
                empty = torch.empty(0, dtype=torch.long, device=logits.device)
                matches.append((empty, empty))
                continue
            if boxes.shape[0] < target_count:
                raise ValueError(
                    "OTA requires at least as many queries as targets; "
                    f"got {boxes.shape[0]} queries and {target_count} targets"
                )

            cost = _diffusionregpose_pairwise_matching_cost(
                logits,
                boxes,
                keypoints,
                target,
                num_body_points=self.num_body_points,
                sigmas=self.sigmas,
                weights=self.weights,
            )
            predicted_xyxy = box_cxcywh_to_xyxy(boxes)
            target_xyxy = box_cxcywh_to_xyxy(target["boxes"])
            pairwise_iou = _pairwise_box_iou(predicted_xyxy, target_xyxy)
            penalized_cost = cost + self._candidate_penalties(
                boxes, target["boxes"],
            )
            matches.append(self._dynamic_k_assignment(penalized_cost, pairwise_iou))
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

    def _loss_normalizer(
        self,
        matches: list[tuple[Tensor, Tensor]],
        targets: list[dict[str, Tensor]],
        reference: Tensor,
    ) -> Tensor:
        count = self.matcher.normalization_count(matches, targets)
        normalizer = reference.new_tensor(float(count))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(normalizer)
            normalizer = normalizer / torch.distributed.get_world_size()
        return normalizer.clamp_min(1)

    def _assignment_diagnostics(
        self,
        matches: list[tuple[Tensor, Tensor]],
        targets: list[dict[str, Tensor]],
        reference: Tensor,
    ) -> dict[str, Tensor]:
        per_target_counts = [
            torch.bincount(target_indices, minlength=target["labels"].numel())
            for (_, target_indices), target in zip(matches, targets)
            if target["labels"].numel()
        ]
        if not per_target_counts:
            zero = reference.new_zeros(())
            return {
                "matches_per_target_mean": zero,
                "matches_per_target_min": zero,
                "matches_per_target_max": zero,
            }
        counts = torch.cat(per_target_counts).to(reference)
        return {
            "matches_per_target_mean": counts.float().mean(),
            "matches_per_target_min": counts.min(),
            "matches_per_target_max": counts.max(),
        }

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
        normalizer = self._loss_normalizer(matches, targets, logits)
        focal_loss = alpha_t * ce * (1 - p_t).pow(2)
        # ``sigmoid_focal_loss`` in the reference already averages over the
        # class dimension, normalizes by the target count, and the reference
        # criterion multiplies that result by the query count once.
        loss_ce = focal_loss.mean(1).sum() / normalizer * logits.shape[1]
        loss_bbox = logits.new_zeros(())
        loss_giou = logits.new_zeros(())
        loss_keypoints = logits.new_zeros(())
        loss_oks = logits.new_zeros(())
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
                    (focal_loss * target_classes).mean(1).sum() / normalizer
                    * logits.shape[1]
                ).detach(),
                "loss_ce_negative": (
                    (focal_loss * (1 - target_classes)).mean(1).sum() / normalizer
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
            values.update({
                name: value.detach()
                for name, value in self._assignment_diagnostics(
                    matches, targets, logits,
                ).items()
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
                auxiliary,
                targets,
                include_keypoints=bool(
                    auxiliary.get("poseflow_full_pose_loss", index >= self.num_box_decoder_layers)
                ),
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
            dn_aux_outputs = outputs.get("dn_aux_outputs")
            if (
                isinstance(dn_aux_outputs, list)
                and index < len(dn_aux_outputs)
                and isinstance(dn_aux_outputs[index], dict)
            ):
                auxiliary_dn = self._denoising_losses({
                    "dn_class_pred": dn_aux_outputs[index]["pred_logits"],
                    "dn_bbox_pred": dn_aux_outputs[index]["pred_boxes"],
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
            "matches_per_target_mean",
            "matches_per_target_min",
            "matches_per_target_max",
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
        image = batch["image"]
        devices = [image.get_device()] if image.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            losses = self._losses_for_batch(batch)
            self._log_losses("val", losses, image.shape[0])
            if self._ap_epoch_active():
                sampling_generator = None
                if self.deterministic_seed is not None:
                    sampling_generator = torch.Generator(device=image.device)
                    sampling_generator.manual_seed(self.deterministic_seed)
                sampling_kwargs = {"padding_mask": batch.get("padding_mask")}
                if sampling_generator is not None:
                    sampling_kwargs["generator"] = sampling_generator
                sampled_outputs = self.model.sample_keypoints(image, **sampling_kwargs)
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
    feedforward_dim: int = 2048
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
    matching_strategy: Literal["hungarian", "ota"] = "hungarian"
    ota_candidate_topk: int = 5
    freeze_backbone_stem_and_layer1: bool = True
    freeze_backbone: bool = False
    freeze_backbone_and_transformer: bool = False
    pretrained_checkpoint: str | None = None
    pretrained_minimum_critical_coverage: float | None = None
    pretrained_keep_encoder_layers: tuple[int, ...] | None = None
    pretrained_keep_decoder_layers: tuple[int, ...] | None = None
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

    @field_validator("pretrained_minimum_critical_coverage")
    @classmethod
    def _validate_pretrained_coverage(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(
                "pretrained_minimum_critical_coverage must be between zero and one"
            )
        return value

    @field_validator("ota_candidate_topk")
    @classmethod
    def _validate_ota_candidate_topk(cls, value: int) -> int:
        if value < 1:
            raise ValueError("ota_candidate_topk must be positive")
        return value

    def build(
        self,
        *,
        max_epochs: int | None = None,
        val_ann_file: str | None = None,
        load_pretrained_checkpoint: bool = True,
    ) -> DiffusionRegPosePipeline:
        keep_encoder_layers = self.pretrained_keep_encoder_layers
        keep_decoder_layers = self.pretrained_keep_decoder_layers
        if (keep_encoder_layers is None) != (keep_decoder_layers is None):
            raise ValueError(
                "pretrained_keep_encoder_layers and "
                "pretrained_keep_decoder_layers must be provided together"
            )
        if (keep_encoder_layers or keep_decoder_layers) and not self.pretrained_checkpoint:
            raise ValueError(
                "selected pretrained layers require pretrained_checkpoint"
            )
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
            feedforward_dim=self.feedforward_dim,
            num_heads=self.num_heads,
            num_body_points=self.num_body_points,
            num_box_decoder_layers=self.num_box_decoder_layers,
            num_group=self.num_group,
            dn_number=self.dn_number,
            dn_box_noise_scale=self.dn_box_noise_scale,
            dn_label_noise_ratio=self.dn_label_noise_ratio,
            freeze_backbone_stem_and_layer1=self.freeze_backbone_stem_and_layer1,
            freeze_backbone=self.freeze_backbone,
            freeze_backbone_and_transformer=self.freeze_backbone_and_transformer,
            separate_expansion_pose_head=keep_decoder_layers is not None,
        )
        if self.pretrained_checkpoint and load_pretrained_checkpoint:
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
            if keep_encoder_layers is not None and keep_decoder_layers is not None:
                # Keep selected one-based EDPose layers so a reduced model is
                # initialized from its intended depth, rather than from the
                # first consecutive layers. This is transfer initialization,
                # not exact six-layer checkpoint inference.
                state = remap_diffusionregpose_checkpoint_layers(
                    state,
                    encoder_layer_indices=keep_encoder_layers,
                    decoder_layer_indices=keep_decoder_layers,
                    target_num_encoder_layers=self.num_encoder_layers,
                    target_num_decoder_layers=self.num_decoder_layers,
                    target_num_box_decoder_layers=self.num_box_decoder_layers,
                )
            model_state = model.state_dict()
            coverage = diffusionregpose_checkpoint_component_coverage(
                model_state, state
            )
            if self.pretrained_minimum_critical_coverage is not None:
                validate_diffusionregpose_checkpoint_coverage(
                    coverage,
                    minimum=self.pretrained_minimum_critical_coverage,
                )
            compatible_state = {
                key: value
                for key, value in state.items()
                if key in model_state and model_state[key].shape == value.shape
            }
            checkpoint_only_keys = set(state) - set(model_state)
            shape_incompatible_keys = {
                key
                for key in set(state) & set(model_state)
                if state[key].shape != model_state[key].shape
            }
            incompatible = model.load_state_dict(compatible_state, strict=False)
            if (
                incompatible.missing_keys
                or incompatible.unexpected_keys
                or checkpoint_only_keys
                or shape_incompatible_keys
            ):
                coverage_summary = ", ".join(
                    f"{component}={value:.2%}"
                    for component, value in coverage.items()
                )
                warnings.warn(
                    f"Loaded {checkpoint_path} with {len(incompatible.missing_keys)} missing and "
                    f"{len(incompatible.unexpected_keys)} unexpected model keys, "
                    f"{len(checkpoint_only_keys)} unused checkpoint keys, and "
                    f"{len(shape_incompatible_keys)} shape-incompatible keys. "
                    f"Critical coverage: {coverage_summary}.",
                    stacklevel=2,
                )
        weights = self.weights.to_dataclass()
        matcher: DiffusionRegPoseMatcher
        if self.matching_strategy == "ota":
            matcher = DiffusionRegPoseOTAMatcher(
                num_body_points=self.num_body_points,
                weights=weights,
                candidate_topk=self.ota_candidate_topk,
            )
        else:
            matcher = DiffusionRegPoseMatcher(
                num_body_points=self.num_body_points,
                weights=weights,
            )
        return DiffusionRegPosePipeline(
            model,
            DiffusionRegPoseCriterion(
                num_classes=self.num_classes,
                matcher=matcher,
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
