"""Continuous endpoint-flow training and sampling for PoseFlow."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Literal

import torch
from pydantic import field_validator, model_validator
from torch import Tensor, nn

from src.models.pipelines.pose_estimators.diffusionregpose.training import (
    CROWDPOSE_PERSON_LABEL,
    DiffusionRegPosePipeline,
    DiffusionRegPosePipelineConfig,
    _keypoints_to_xyxyzz,
    _postprocess_diffusionregpose_ap,
    diffusionregpose_ap_epoch_active,
    update_diffusionregpose_ap_metric,
    update_diffusionregpose_bbox_ap_metric,
)
from src.training.optim import build_adamw_with_schedule
from .model import (
    PoseFlowModel,
    PoseFlowV3Model,
    PoseFlowV4Model,
    PoseFlowV5Model,
    PoseFlowV6Model,
    PoseFlowV7Model,
    PoseFlowV8Model,
    PoseFlowV9Model,
    PoseFlowV10Model,
)
from .model_v11 import PoseFlowV11Model
from .state import (
    cxcywh_to_corners,
    euler_update_person_state,
    gather_person_slot_fields,
    interpolate_person_state,
    merge_person_slot_fields,
    pack_person_state,
)


def _adapt_v8_encoder_attention_levels(
    source_state: dict[str, Tensor],
    target_state: dict[str, Tensor],
    *,
    num_heads: int,
    source_levels: int = 4,
    target_levels: int = 3,
) -> tuple[dict[str, Tensor], tuple[str, ...]]:
    """Retain the first three pretrained level groups in V8 encoder attention."""
    adapted = dict(source_state)
    adapted_keys: list[str] = []
    for name, value in source_state.items():
        if ".encoder.layers." not in name or not any(
            component in name
            for component in (".sampling_offsets.", ".attention_weights.")
        ):
            continue
        target = target_state.get(name)
        if target is None or target.shape == value.shape or value.ndim < 1:
            continue
        if target.ndim != value.ndim or target.shape[1:] != value.shape[1:]:
            continue
        source_group = value.shape[0] // (num_heads * source_levels)
        target_group = target.shape[0] // (num_heads * target_levels)
        if (
            source_group <= 0
            or source_group != target_group
            or value.shape[0] != num_heads * source_levels * source_group
            or target.shape[0] != num_heads * target_levels * target_group
        ):
            continue
        grouped = value.reshape(num_heads, source_levels, source_group, *value.shape[1:])
        adapted[name] = grouped[:, :target_levels].reshape_as(target)
        adapted_keys.append(name)
    return adapted, tuple(adapted_keys)


def _flow_generator(
    tensor: Tensor, seed: int | None
) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed)
    return generator


def _transport_to_decoder_coordinates(state: Tensor, scale: float) -> Tensor:
    return ((state / scale) + 1.0).div(2.0).clamp(0.0, 1.0)


def build_poseflow_targets(
    batch: dict[str, Tensor],
    model: nn.Module,
    *,
    num_proposals: int,
    num_body_points: int,
    padding_policy: Literal["source_prior", "diffusionregpose"],
    noise_seed: int | None = None,
    flow_times: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, list[dict[str, Tensor]]]:
    """Build endpoint-flow decoder inputs and existing semantic task targets."""
    keypoints = batch["gt_keypoints"]
    proposal_keypoints = batch.get("gt_diffusion_keypoints", keypoints)
    labeled = batch["gt_labeled"].bool()
    boxes = batch["gt_boxes_canon"]
    batch_size, person_count, point_count, _ = keypoints.shape
    if point_count != num_body_points:
        raise ValueError("batch keypoint count does not match the model")
    if boxes.shape[:2] != (batch_size, person_count):
        raise ValueError("box and keypoint target counts must agree")
    if proposal_keypoints.shape[:1] != (batch_size,) or proposal_keypoints.shape[2:] != (
        point_count,
        2,
    ):
        raise ValueError("proposal keypoints must match batch and keypoint dimensions")
    if num_proposals < 1:
        raise ValueError("num_proposals must be positive")

    scale = float(model.scale)
    generator = _flow_generator(keypoints, noise_seed)
    source = torch.randn(
        batch_size,
        num_proposals,
        point_count * 2,
        device=keypoints.device,
        dtype=keypoints.dtype,
        generator=generator,
    )
    endpoint = torch.empty_like(source)
    targets: list[dict[str, Tensor]] = []
    for batch_index in range(batch_size):
        valid_indices = labeled[batch_index].any(dim=-1).nonzero(as_tuple=False).flatten()
        target_keypoints = (keypoints[batch_index].index_select(0, valid_indices) + 1.0) / 2.0
        target_visibility = labeled[batch_index].index_select(0, valid_indices).float()
        target_boxes = boxes[batch_index].index_select(0, valid_indices)
        targets.append({
            "labels": torch.full(
                (valid_indices.numel(),),
                CROWDPOSE_PERSON_LABEL,
                dtype=torch.long,
                device=keypoints.device,
            ),
            "boxes": target_boxes,
            "keypoints": _keypoints_to_xyxyzz(
                torch.cat((target_keypoints, target_visibility.unsqueeze(-1)), dim=-1)
            ),
            "area": (target_boxes[:, 2] * target_boxes[:, 3]).clamp_min(1e-6),
        })

        proposal_indices = batch.get("diffusion_person_mask", batch.get("person_mask"))
        if proposal_indices is None:
            proposal_indices = torch.arange(proposal_keypoints.shape[1], device=keypoints.device)
        else:
            proposal_indices = proposal_indices[batch_index].bool().nonzero(as_tuple=False).flatten()
        clean = proposal_keypoints[batch_index].index_select(0, proposal_indices)
        clean = ((clean + 1.0) / 2.0).flatten(1)
        count = min(clean.shape[0], num_proposals)
        if count:
            endpoint[batch_index, :count] = (clean[:count] * 2.0 - 1.0) * scale
        if count < num_proposals:
            if padding_policy == "source_prior":
                endpoint[batch_index, count:] = source[batch_index, count:]
            else:
                endpoint[batch_index, count:] = torch.randn(
                    num_proposals - count,
                    point_count * 2,
                    device=keypoints.device,
                    dtype=keypoints.dtype,
                    generator=generator,
                ) * scale

    if flow_times is None:
        times = torch.rand(
            batch_size, device=keypoints.device, dtype=keypoints.dtype, generator=generator
        )
    else:
        if (flow_times.shape != (batch_size,) or not torch.isfinite(flow_times).all()
                or ((flow_times < 0) | (flow_times > 1)).any()):
            raise ValueError('Explicit flow times must be finite [batch] values in [0,1]')
        times = flow_times.to(keypoints)
    state = (1.0 - times[:, None, None]) * source + times[:, None, None] * endpoint
    return _transport_to_decoder_coordinates(state, scale), source, times, targets


def build_poseflow_v3_targets(
    batch: dict[str, Tensor],
    model: nn.Module,
    *,
    num_queries: int,
    num_body_points: int,
    padding_policy: Literal["source_prior", "diffusionregpose"],
    noise_seed: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, list[dict[str, Tensor]]]:
    """Build joint corner-and-pose endpoint-flow states for PoseFlow V3."""
    keypoints = batch["gt_keypoints"]
    proposal_keypoints = batch.get("gt_diffusion_keypoints", keypoints)
    labeled = batch["gt_labeled"].bool()
    boxes = batch["gt_boxes_canon"]
    batch_size, person_count, point_count, _ = keypoints.shape
    if point_count != num_body_points:
        raise ValueError("batch keypoint count does not match the model")
    if boxes.shape[:2] != (batch_size, person_count):
        raise ValueError("box and keypoint target counts must agree")
    if num_queries < 1:
        raise ValueError("num_queries must be positive")
    state_width = 4 + 2 * num_body_points
    scale = float(model.scale)
    generator = _flow_generator(keypoints, noise_seed)
    source = torch.randn(
        batch_size, num_queries, state_width,
        device=keypoints.device, dtype=keypoints.dtype, generator=generator,
    )
    endpoint = torch.empty_like(source)
    targets: list[dict[str, Tensor]] = []
    proposal_masks = batch.get("diffusion_person_mask", batch.get("person_mask"))
    for batch_index in range(batch_size):
        valid_indices = labeled[batch_index].any(dim=-1).nonzero(as_tuple=False).flatten()
        target_keypoints = (keypoints[batch_index].index_select(0, valid_indices) + 1.0) / 2.0
        target_visibility = labeled[batch_index].index_select(0, valid_indices).float()
        target_boxes = boxes[batch_index].index_select(0, valid_indices)
        targets.append({
            "labels": torch.full(
                (valid_indices.numel(),), CROWDPOSE_PERSON_LABEL,
                dtype=torch.long, device=keypoints.device,
            ),
            "boxes": target_boxes,
            "keypoints": _keypoints_to_xyxyzz(
                torch.cat((target_keypoints, target_visibility.unsqueeze(-1)), dim=-1)
            ),
            "area": (target_boxes[:, 2] * target_boxes[:, 3]).clamp_min(1e-6),
        })
        proposal_indices = (
            torch.arange(proposal_keypoints.shape[1], device=keypoints.device)
            if proposal_masks is None
            else proposal_masks[batch_index].bool().nonzero(as_tuple=False).flatten()
        )
        clean_keypoints = ((
            proposal_keypoints[batch_index].index_select(0, proposal_indices) + 1.0
        ) / 2.0).flatten(1)
        clean_corners = cxcywh_to_corners(
            boxes[batch_index].index_select(0, proposal_indices)
        )
        clean_state = pack_person_state(clean_corners, clean_keypoints)
        count = min(clean_state.shape[0], num_queries)
        if count:
            endpoint[batch_index, :count] = (clean_state[:count] * 2.0 - 1.0) * scale
        if count < num_queries:
            if padding_policy == "source_prior":
                endpoint[batch_index, count:] = source[batch_index, count:]
            else:
                endpoint[batch_index, count:] = torch.randn(
                    num_queries - count, state_width,
                    device=keypoints.device, dtype=keypoints.dtype, generator=generator,
                ) * scale
    times = torch.rand(
        batch_size, device=keypoints.device, dtype=keypoints.dtype, generator=generator,
    )
    return interpolate_person_state(source, endpoint, times), source, times, targets


_V5_FROM_SCRATCH_SIZE = (0.2526, 0.4776)
_V5_CENTER_OFFSET_STD = 0.25
_V5_LOG_SCALE_STD = 0.25


def _normalized_state_to_flow_coordinates(state: Tensor, scale: float) -> Tensor:
    return (state * 2.0 - 1.0) * scale


def _sample_v5_from_scratch_sources(
    *,
    batch_size: int,
    source_count: int,
    num_body_points: int,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
    box_format: Literal["corners", "center_size"] = "corners",
) -> Tensor:
    state_width = 4 + 2 * num_body_points
    source = torch.randn(
        batch_size, source_count, state_width,
        device=device, dtype=dtype, generator=generator,
    )
    centers = torch.rand(
        batch_size, source_count, 2,
        device=device, dtype=dtype, generator=generator,
    )
    base_size = source.new_tensor(_V5_FROM_SCRATCH_SIZE)
    size_scales = torch.exp(torch.randn(
        batch_size, source_count, 2,
        device=device, dtype=dtype, generator=generator,
    ) * _V5_LOG_SCALE_STD)
    sizes = base_size * size_scales
    corners = torch.cat((centers - sizes / 2.0, centers + sizes / 2.0), dim=-1)
    box_state = corners if box_format == "corners" else torch.cat((centers, sizes), dim=-1)
    source[..., :4] = _normalized_state_to_flow_coordinates(box_state, scale)
    return source


def _augment_v5_person_states(
    clean_states: Tensor,
    *,
    source_count: int,
    generator: torch.Generator | None,
    box_format: Literal["corners", "center_size"] = "corners",
) -> Tensor:
    source_indices = torch.randint(
        clean_states.shape[0], (source_count,),
        device=clean_states.device, generator=generator,
    )
    selected = clean_states.index_select(0, source_indices)
    boxes = selected[..., :4]
    corners = (boxes if box_format == "corners" else cxcywh_to_corners(boxes)).view(source_count, 2, 2)
    keypoints = selected[..., 4:].view(source_count, -1, 2)
    original_center = corners.mean(dim=1)
    displacement = torch.randn(
        source_count, 2, device=selected.device, dtype=selected.dtype, generator=generator,
    ) * _V5_CENTER_OFFSET_STD
    scales = torch.exp(torch.randn(
        source_count, 2, device=selected.device, dtype=selected.dtype, generator=generator,
    ) * _V5_LOG_SCALE_STD)
    new_center = original_center + displacement
    transformed_corners = new_center[:, None] + (corners - original_center[:, None]) * scales[:, None]
    transformed_keypoints = new_center[:, None] + (keypoints - original_center[:, None]) * scales[:, None]
    box_state = (transformed_corners.flatten(1) if box_format == "corners"
                 else torch.cat((new_center, boxes[..., 2:] * scales), dim=-1))
    return torch.cat((box_state, transformed_keypoints.flatten(1)), dim=-1)


def _interpolate_v5_person_states(
    clean_states: Tensor,
    *,
    source_count: int,
    generator: torch.Generator | None,
) -> Tensor:
    first = torch.randint(
        clean_states.shape[0], (source_count,),
        device=clean_states.device, generator=generator,
    )
    offsets = torch.randint(
        1, clean_states.shape[0], (source_count,),
        device=clean_states.device, generator=generator,
    )
    second = (first + offsets).remainder(clean_states.shape[0])
    interpolation = torch.rand(
        source_count, 1, device=clean_states.device, dtype=clean_states.dtype, generator=generator,
    )
    return interpolation * clean_states.index_select(0, first) + (
        1.0 - interpolation
    ) * clean_states.index_select(0, second)


def build_poseflow_v5_targets(
    batch: dict[str, Tensor],
    model: nn.Module,
    *,
    num_queries: int,
    num_body_points: int,
    padding_policy: Literal["source_prior", "diffusionregpose"],
    noise_seed: int | None = None,
    box_format: Literal["corners", "center_size"] = "corners",
) -> tuple[Tensor, Tensor, Tensor, list[dict[str, Tensor]]]:
    """Build V5 endpoint-flow states from augmented and interpolated people."""
    keypoints = batch["gt_keypoints"]
    proposal_keypoints = batch.get("gt_diffusion_keypoints", keypoints)
    labeled = batch["gt_labeled"].bool()
    boxes = batch["gt_boxes_canon"]
    batch_size, person_count, point_count, _ = keypoints.shape
    if point_count != num_body_points:
        raise ValueError("batch keypoint count does not match the model")
    if boxes.shape[:2] != (batch_size, person_count):
        raise ValueError("box and keypoint target counts must agree")
    if num_queries < 1:
        raise ValueError("num_queries must be positive")

    scale = float(model.scale)
    generator = _flow_generator(keypoints, noise_seed)
    source = _sample_v5_from_scratch_sources(
        batch_size=batch_size,
        source_count=num_queries,
        num_body_points=num_body_points,
        scale=scale,
        device=keypoints.device,
        dtype=keypoints.dtype,
        generator=generator, box_format=box_format,
    )
    endpoint = torch.empty_like(source)
    targets: list[dict[str, Tensor]] = []
    proposal_masks = batch.get("diffusion_person_mask", batch.get("person_mask"))
    for batch_index in range(batch_size):
        valid_indices = labeled[batch_index].any(dim=-1).nonzero(as_tuple=False).flatten()
        target_keypoints = (keypoints[batch_index].index_select(0, valid_indices) + 1.0) / 2.0
        target_visibility = labeled[batch_index].index_select(0, valid_indices).float()
        target_boxes = boxes[batch_index].index_select(0, valid_indices)
        targets.append({
            "labels": torch.full(
                (valid_indices.numel(),), CROWDPOSE_PERSON_LABEL,
                dtype=torch.long, device=keypoints.device,
            ),
            "boxes": target_boxes,
            "keypoints": _keypoints_to_xyxyzz(
                torch.cat((target_keypoints, target_visibility.unsqueeze(-1)), dim=-1)
            ),
            "area": (target_boxes[:, 2] * target_boxes[:, 3]).clamp_min(1e-6),
        })
        if valid_indices.numel():
            clean_states = pack_person_state(
                (cxcywh_to_corners(target_boxes) if box_format == "corners" else target_boxes), target_keypoints.flatten(1),
            )
            augmented_count = num_queries // 2 if valid_indices.numel() > 1 else num_queries
            augmented = _augment_v5_person_states(
                clean_states, source_count=augmented_count, generator=generator, box_format=box_format,
            )
            source[batch_index, :augmented_count] = _normalized_state_to_flow_coordinates(augmented, scale)
            if augmented_count < num_queries:
                interpolated = _interpolate_v5_person_states(
                    clean_states,
                    source_count=num_queries - augmented_count,
                    generator=generator,
                )
                source[batch_index, augmented_count:] = _normalized_state_to_flow_coordinates(interpolated, scale)

        proposal_indices = (
            torch.arange(proposal_keypoints.shape[1], device=keypoints.device)
            if proposal_masks is None
            else proposal_masks[batch_index].bool().nonzero(as_tuple=False).flatten()
        )
        clean_keypoints = ((
            proposal_keypoints[batch_index].index_select(0, proposal_indices) + 1.0
        ) / 2.0).flatten(1)
        clean_boxes = boxes[batch_index].index_select(0, proposal_indices)
        clean_box_state = cxcywh_to_corners(clean_boxes) if box_format == "corners" else clean_boxes
        clean_state = pack_person_state(clean_box_state, clean_keypoints)
        count = min(clean_state.shape[0], num_queries)
        if count:
            endpoint[batch_index, :count] = _normalized_state_to_flow_coordinates(clean_state[:count], scale)
        if count < num_queries:
            if padding_policy == "source_prior":
                endpoint[batch_index, count:] = source[batch_index, count:]
            else:
                endpoint[batch_index, count:] = torch.randn(
                    num_queries - count, source.shape[-1], device=keypoints.device,
                    dtype=keypoints.dtype, generator=generator,
                ) * scale
    times = torch.rand(
        batch_size, device=keypoints.device, dtype=keypoints.dtype, generator=generator,
    )
    return interpolate_person_state(source, endpoint, times), source, times, targets


@torch.no_grad()
def flow_sample_keypoints(
    model: nn.Module,
    images: Tensor,
    *,
    padding_mask: Tensor | None = None,
    num_proposals: int | None = None,
    sampling_steps: int,
    renewal: bool = False,
    renewal_score_threshold: float = 0.5,
    ensemble: bool = False,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor | list[dict[str, Tensor]]]:
    """Run fixed-source forward-Euler endpoint flow sampling."""
    proposal_count = model.num_proposals if num_proposals is None else num_proposals
    if proposal_count < 1:
        raise ValueError(f"num_proposals must be positive; got {proposal_count}")
    if proposal_count != model.transformer.decoder.num_group:
        raise ValueError("num_proposals must equal the configured decoder group count")
    if sampling_steps < 1:
        raise ValueError(f"sampling_steps must be positive; got {sampling_steps}")
    if not 0.0 <= renewal_score_threshold <= 1.0:
        raise ValueError("renewal_score_threshold must be in [0, 1]")

    source = torch.randn(
        images.shape[0],
        proposal_count,
        model.transformer.decoder.num_body_points * 2,
        device=images.device,
        dtype=images.dtype,
        generator=generator,
    )
    state = source.clone()
    final_output: dict[str, Tensor | list[dict[str, Tensor]]] | None = None
    step_predictions: list[dict[str, Tensor]] = []
    renewed_slots: list[Tensor] = []
    for step in range(sampling_steps):
        times = torch.full(
            (images.shape[0],), step / sampling_steps, device=images.device, dtype=images.dtype
        )
        final_output = model(
            images,
            state,
            times,
            padding_mask=padding_mask,
        )
        predicted_keypoints = final_output["pred_keypoints"]
        if not isinstance(predicted_keypoints, Tensor):
            raise RuntimeError("model forward returned an invalid keypoint prediction")
        for name in ("pred_logits", "pred_boxes"):
            if not isinstance(final_output[name], Tensor):
                raise RuntimeError(f"model forward returned invalid {name}")
        step_predictions.append({
            name: final_output[name]
            for name in ("pred_logits", "pred_boxes", "pred_keypoints")
        })
        coordinate_count = model.transformer.decoder.num_body_points * 2
        endpoint = (predicted_keypoints[..., :coordinate_count] * 2.0 - 1.0) * model.scale
        state = state + (endpoint - source) / sampling_steps
        if renewal and step < sampling_steps - 1:
            scores = final_output["pred_logits"].sigmoid().amax(dim=-1)
            weak_slots = scores < renewal_score_threshold
            replacement = torch.randn(
                source.shape, device=source.device, dtype=source.dtype, generator=generator
            )
            source = torch.where(weak_slots[..., None], replacement, source)
            state = torch.where(weak_slots[..., None], replacement, state)
            renewed_slots.append(weak_slots.sum(dim=1))
    if final_output is None:
        raise RuntimeError("flow sampler did not execute")
    if ensemble:
        # Candidate selection must happen independently at every flow step.
        # Keeping these sets separate lets the evaluator aggregate selected
        # candidates before one cross-step pose NMS, matching the reference
        # detector's inference ordering.
        final_output["flow_step_predictions"] = step_predictions
    final_output["flow_keypoints"] = state
    final_output["flow_source"] = source
    final_output["flow_renewed_slots"] = (
        torch.stack(renewed_slots) if renewed_slots else torch.zeros(
            sampling_steps - 1, images.shape[0], dtype=torch.long, device=images.device
        )
    )
    return final_output


@torch.no_grad()
def flow_sample_person_states(
    model: PoseFlowV3Model,
    images: Tensor,
    *,
    padding_mask: Tensor | None = None,
    num_proposals: int | None = None,
    sampling_steps: int,
    ensemble: bool = False,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor | list[dict[str, Tensor]]]:
    """Run fixed-source Euler flow with 50 carried and 150 fresh V3 slots."""
    active_count = model.num_proposals if num_proposals is None else num_proposals
    if active_count < 1:
        raise ValueError(f"num_proposals must be positive; got {active_count}")
    if active_count != model.transformer.decoder.num_group:
        raise ValueError("num_proposals must equal the configured decoder group count")
    if active_count >= model.num_queries:
        raise ValueError("PoseFlow V3 requires room for at least one fresh query")
    if sampling_steps < 1:
        raise ValueError(f"sampling_steps must be positive; got {sampling_steps}")
    state_width = 4 + 2 * model.transformer.decoder.num_body_points
    fresh_count = model.num_queries - active_count
    source = torch.randn(
        images.shape[0], model.num_queries, state_width,
        device=images.device, dtype=images.dtype, generator=generator,
    )
    state = source.clone()
    active_query_features: Tensor | None = None
    active_image_references: Tensor | None = None
    final_output: dict[str, Tensor | list[dict[str, Tensor]]] | None = None
    step_predictions: list[dict[str, Tensor]] = []
    for step in range(sampling_steps):
        times = torch.full(
            (images.shape[0],), step / sampling_steps,
            device=images.device, dtype=images.dtype,
        )
        final_output = model(
            images,
            state[..., 4:],
            times,
            padding_mask=padding_mask,
            person_state=state,
            active_query_features=active_query_features,
            active_image_references=active_image_references,
            return_selection_metadata=True,
        )
        predicted_boxes = final_output.get("pred_boxes")
        predicted_keypoints = final_output.get("pred_keypoints")
        selected_indices = final_output.get("selected_person_indices")
        selected_features = final_output.get("selected_person_features")
        selected_references = final_output.get("selected_person_references")
        if not all(isinstance(value, Tensor) for value in (
            predicted_boxes, predicted_keypoints, selected_indices,
            selected_features, selected_references,
        )):
            raise RuntimeError("PoseFlow V3 forward returned incomplete person selection metadata")
        step_predictions.append({
            name: final_output[name]
            for name in ("pred_logits", "pred_boxes", "pred_keypoints")
            if isinstance(final_output[name], Tensor)
        })
        selected_fields = gather_person_slot_fields(
            {"state": state, "source": source}, selected_indices
        )
        endpoint = pack_person_state(
            cxcywh_to_corners(predicted_boxes),
            predicted_keypoints[..., : 2 * model.transformer.decoder.num_body_points],
        )
        active_state = euler_update_person_state(
            selected_fields["state"],
            (endpoint * 2.0 - 1.0) * model.scale,
            selected_fields["source"],
            sampling_steps=sampling_steps,
        )
        if step == sampling_steps - 1:
            state, source = active_state, selected_fields["source"]
            break
        fresh_source = torch.randn(
            images.shape[0], fresh_count, state_width,
            device=images.device, dtype=images.dtype, generator=generator,
        )
        pooled = merge_person_slot_fields(
            {"state": active_state, "source": selected_fields["source"]},
            {"state": fresh_source, "source": fresh_source},
        )
        state, source = pooled["state"], pooled["source"]
        active_query_features = selected_features
        active_image_references = selected_references
    if final_output is None:
        raise RuntimeError("PoseFlow V3 sampler did not execute")
    if ensemble:
        final_output["flow_step_predictions"] = step_predictions
    final_output["flow_person_state"] = state
    final_output["flow_person_source"] = source
    final_output["flow_renewed_slots"] = torch.zeros(
        sampling_steps - 1, images.shape[0], dtype=torch.long, device=images.device
    )
    return final_output


@torch.no_grad()
def flow_sample_poseflow_v4_states(
    model: PoseFlowV4Model,
    images: Tensor,
    *,
    padding_mask: Tensor | None = None,
    num_proposals: int | None = None,
    sampling_steps: int,
    ensemble: bool = False,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor | list[dict[str, Tensor]]]:
    """Run V4 coordinate flow; hypotheses are recomputed at every call."""
    active_count = model.num_proposals if num_proposals is None else num_proposals
    if active_count != model.num_proposals or active_count != model.transformer.decoder.num_group:
        raise ValueError("V4 sampling requires the configured active proposal count")
    if active_count < 1 or active_count >= model.num_queries:
        raise ValueError("V4 requires between one and num_queries-minus-one active slots")
    if sampling_steps < 1:
        raise ValueError("sampling_steps must be positive")
    state_width = 4 + 2 * model.transformer.decoder.num_body_points
    fresh_count = model.num_queries - active_count
    source = torch.randn(
        images.shape[0], model.num_queries, state_width,
        device=images.device, dtype=images.dtype, generator=generator,
    )
    state = source.clone()
    final_output: dict[str, Tensor | list[dict[str, Tensor]]] | None = None
    step_predictions: list[dict[str, Tensor]] = []
    for step in range(sampling_steps):
        times = torch.full(
            (images.shape[0],), step / sampling_steps,
            device=images.device, dtype=images.dtype,
        )
        final_output = model(
            images,
            state[..., 4:],
            times,
            padding_mask=padding_mask,
            person_state=state,
            return_selection_metadata=True,
        )
        predicted_boxes = final_output.get("pred_boxes")
        predicted_keypoints = final_output.get("pred_keypoints")
        selected_indices = final_output.get("selected_person_indices")
        if not all(isinstance(value, Tensor) for value in (
            predicted_boxes, predicted_keypoints, selected_indices,
        )):
            raise RuntimeError("PoseFlow V4 forward returned incomplete selection metadata")
        step_predictions.append({
            name: final_output[name]
            for name in ("pred_logits", "pred_boxes", "pred_keypoints")
            if isinstance(final_output[name], Tensor)
        })
        selected_fields = gather_person_slot_fields(
            {"state": state, "source": source}, selected_indices
        )
        endpoint = pack_person_state(
            cxcywh_to_corners(predicted_boxes),
            predicted_keypoints[..., : 2 * model.transformer.decoder.num_body_points],
        )
        active_state = euler_update_person_state(
            selected_fields["state"],
            (endpoint * 2.0 - 1.0) * model.scale,
            selected_fields["source"],
            sampling_steps=sampling_steps,
        )
        if step == sampling_steps - 1:
            state, source = active_state, selected_fields["source"]
            break
        fresh_source = torch.randn(
            images.shape[0], fresh_count, state_width,
            device=images.device, dtype=images.dtype, generator=generator,
        )
        pooled = merge_person_slot_fields(
            {"state": active_state, "source": selected_fields["source"]},
            {"state": fresh_source, "source": fresh_source},
        )
        state, source = pooled["state"], pooled["source"]
    if final_output is None:
        raise RuntimeError("V4 flow sampler did not execute")
    if ensemble:
        final_output["flow_step_predictions"] = step_predictions
    final_output["flow_person_state"] = state
    final_output["flow_person_source"] = source
    final_output["flow_renewed_slots"] = torch.zeros(
        sampling_steps - 1, images.shape[0], dtype=torch.long, device=images.device
    )
    return final_output


@torch.no_grad()
def flow_sample_poseflow_v5_states(
    model: PoseFlowV5Model,
    images: Tensor,
    *,
    padding_mask: Tensor | None = None,
    num_proposals: int | None = None,
    sampling_steps: int,
    ensemble: bool = False,
    generator: torch.Generator | None = None,
    collect_step_diagnostics: bool = False,
    box_format: Literal["corners", "center_size"] = "corners",
    forward_kwargs: dict[str, object] | None = None,
) -> dict[str, Tensor | list[dict[str, Tensor]]]:
    """Run V5 Euler flow with from-scratch sources and V4 carry/refill."""
    active_count = model.num_proposals if num_proposals is None else num_proposals
    if active_count != model.num_proposals or active_count != model.transformer.decoder.num_group:
        raise ValueError("V5 sampling requires the configured active proposal count")
    if active_count < 1 or active_count >= model.num_queries:
        raise ValueError("V5 requires between one and num_queries-minus-one active slots")
    if sampling_steps < 1:
        raise ValueError("sampling_steps must be positive")
    fresh_count = model.num_queries - active_count
    source = _sample_v5_from_scratch_sources(
        batch_size=images.shape[0],
        source_count=model.num_queries,
        num_body_points=model.transformer.decoder.num_body_points,
        scale=float(model.scale),
        device=images.device,
        dtype=images.dtype,
        generator=generator, box_format=box_format,
    )
    state = source.clone()
    final_output: dict[str, Tensor | list[dict[str, Tensor]]] | None = None
    step_predictions: list[dict[str, Tensor]] = []
    step_diagnostics: list[dict[str, Tensor]] = []
    slot_ids = (
        torch.arange(model.num_queries, device=images.device).expand(images.shape[0], -1)
        if collect_step_diagnostics else None
    )
    for step in range(sampling_steps):
        times = torch.full(
            (images.shape[0],), step / sampling_steps,
            device=images.device, dtype=images.dtype,
        )
        final_output = model(
            images,
            state[..., 4:],
            times,
            padding_mask=padding_mask,
            person_state=state,
            return_selection_metadata=True,
            **({"collect_diagnostics": collect_step_diagnostics} if box_format == "center_size" else {}),
            **(forward_kwargs or {}),
        )
        predicted_boxes = final_output.get("pred_boxes")
        predicted_keypoints = final_output.get("pred_keypoints")
        selected_indices = final_output.get("selected_person_indices")
        if not all(isinstance(value, Tensor) for value in (
            predicted_boxes, predicted_keypoints, selected_indices,
        )):
            raise RuntimeError("PoseFlow V5 forward returned incomplete selection metadata")
        step_predictions.append({
            name: final_output[name]
            for name in ("pred_logits", "pred_boxes", "pred_keypoints")
            if isinstance(final_output[name], Tensor)
        })
        selected_fields = gather_person_slot_fields(
            {"state": state, "source": source}, selected_indices
        )
        if collect_step_diagnostics:
            assert slot_ids is not None
            step_diagnostics.append({
                **step_predictions[-1],
                "time": times,
                "state": state,
                "source": source,
                "slot_ids": slot_ids,
                "selected_indices": selected_indices,
                "selected_slot_ids": torch.gather(slot_ids, 1, selected_indices),
                "selected_from_fresh": (selected_indices >= active_count)
                if step else torch.ones_like(selected_indices, dtype=torch.bool),
                "selected_state": selected_fields["state"],
                "selected_source": selected_fields["source"],
            })
        endpoint = pack_person_state(
            cxcywh_to_corners(predicted_boxes) if box_format == "corners" else predicted_boxes,
            predicted_keypoints[..., : 2 * model.transformer.decoder.num_body_points],
        )
        active_state = euler_update_person_state(
            selected_fields["state"],
            (endpoint * 2.0 - 1.0) * model.scale,
            selected_fields["source"],
            sampling_steps=sampling_steps,
        )
        if box_format == "center_size":
            minimum_size = (1e-6 * 2.0 - 1.0) * model.scale
            active_state = torch.cat((
                active_state[..., :2], active_state[..., 2:4].clamp_min(minimum_size),
                active_state[..., 4:],
            ), dim=-1)
            if collect_step_diagnostics:
                step_diagnostics[-1]["token_diagnostics"] = model.last_token_diagnostics()
        if collect_step_diagnostics:
            step_diagnostics[-1]["state_update_l1"] = (
                active_state - selected_fields["state"]
            ).abs().mean(dim=-1)
        if step == sampling_steps - 1:
            state, source = active_state, selected_fields["source"]
            break
        fresh_source = _sample_v5_from_scratch_sources(
            batch_size=images.shape[0],
            source_count=fresh_count,
            num_body_points=model.transformer.decoder.num_body_points,
            scale=float(model.scale),
            device=images.device,
            dtype=images.dtype,
            generator=generator, box_format=box_format,
        )
        pooled = merge_person_slot_fields(
            {"state": active_state, "source": selected_fields["source"]},
            {"state": fresh_source, "source": fresh_source},
        )
        state, source = pooled["state"], pooled["source"]
        if collect_step_diagnostics:
            fresh_ids = torch.arange(
                (step + 1) * model.num_queries,
                (step + 1) * model.num_queries + fresh_count,
                device=images.device,
            ).expand(images.shape[0], -1)
            slot_ids = torch.cat((step_diagnostics[-1]["selected_slot_ids"], fresh_ids), dim=1)
    if final_output is None:
        raise RuntimeError("V5 flow sampler did not execute")
    if ensemble:
        final_output["flow_step_predictions"] = step_predictions
    if collect_step_diagnostics:
        final_output["flow_step_diagnostics"] = step_diagnostics
    final_output["flow_person_state"] = state
    final_output["flow_person_source"] = source
    final_output["flow_renewed_slots"] = torch.zeros(
        sampling_steps - 1, images.shape[0], dtype=torch.long, device=images.device
    )
    return final_output


class PoseFlowPipeline(DiffusionRegPosePipeline):
    """DiffusionRegPose architecture trained and sampled as endpoint flow."""

    def __init__(self, *args, flow_padding_policy: Literal["source_prior", "diffusionregpose"], flow_sampling_steps: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flow_padding_policy = flow_padding_policy
        self.flow_sampling_steps = flow_sampling_steps
        self._latest_codebook_diagnostics: dict[str, Tensor] | None = None

    def _losses_for_batch(
        self,
        batch: dict[str, Tensor],
        *,
        collect_codebook_attention: bool = False,
    ) -> dict[str, Tensor]:
        decoder_state, _, times, targets = build_poseflow_targets(
            batch,
            self.model,
            num_proposals=self.model.num_proposals,
            num_body_points=self.model.transformer.decoder.num_body_points,
            padding_policy=self.flow_padding_policy,
            noise_seed=self.diffusion_noise_seed,
        )
        rng_context = nullcontext() if self.deterministic_seed is None else torch.random.fork_rng(
            devices=[batch["image"].device] if batch["image"].is_cuda else []
        )
        with rng_context:
            if self.deterministic_seed is not None:
                torch.manual_seed(self.deterministic_seed)
            outputs = self.model(
                batch["image"], (decoder_state * 2.0 - 1.0) * self.model.scale, times,
                targets,
                batch.get("padding_mask"),
                collect_codebook_diagnostics=self.model.pose_codebook is not None,
                collect_codebook_attention=collect_codebook_attention,
            )
        self._latest_codebook_diagnostics = self.model.last_codebook_diagnostics()
        return self.criterion(outputs, targets)

    def _log_codebook_diagnostics(self, prefix: str, batch_size: int) -> None:
        if self._latest_codebook_diagnostics is None:
            return
        for name, value in self._latest_codebook_diagnostics.items():
            self.log(
                f"debug/codebook/{prefix}/{name}",
                value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        losses = self._losses_for_batch(batch)
        self._log_losses("train", losses, batch["image"].shape[0])
        self._log_codebook_diagnostics("train", batch["image"].shape[0])
        return losses["total"]

    def sample_keypoints(
        self,
        images: Tensor,
        *,
        padding_mask: Tensor | None = None,
        num_proposals: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        """Sample endpoint-flow pose predictions with the configured solver."""
        return self.model.sample_keypoints(
            images,
            padding_mask=padding_mask,
            num_proposals=num_proposals,
            generator=generator,
        )

    def _validation_sample(
        self, image: Tensor, padding_mask: Tensor | None,
        generator: torch.Generator | None, *, collect_diagnostics: bool,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        del collect_diagnostics
        return self.sample_keypoints(image, padding_mask=padding_mask, generator=generator)

    def _log_validation_solution(
        self, batch: dict[str, Tensor], output: dict,
        results: list[dict[str, Tensor]], *, batch_index: int,
    ) -> None:
        del batch, output, results, batch_index

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        image = batch["image"]
        devices = [image.get_device()] if image.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            losses = self._losses_for_batch(
                batch,
                collect_codebook_attention=batch_idx == 0,
            )
            self._log_losses("val", losses, image.shape[0])
            self._log_codebook_diagnostics("val", image.shape[0])
            if self._ap_epoch_active():
                generator = None
                if self.deterministic_seed is not None:
                    generator = torch.Generator(device=image.device)
                    generator.manual_seed(self.deterministic_seed)
                sampled_outputs = self._validation_sample(
                    image, batch.get("padding_mask"), generator,
                    collect_diagnostics=batch_idx == 0,
                )
                results = _postprocess_diffusionregpose_ap(
                    sampled_outputs, batch,
                    num_body_points=self.model.transformer.decoder.num_body_points,
                )
                update_diffusionregpose_ap_metric(
                    self._val_ap, sampled_outputs, batch,
                    num_body_points=self.model.transformer.decoder.num_body_points,
                    results=results,
                )
                assert self._val_bbox_ap is not None
                update_diffusionregpose_bbox_ap_metric(self._val_bbox_ap, results, batch)
                self._log_validation_solution(
                    batch, sampled_outputs, results, batch_index=batch_idx,
                )


class PoseFlowV3Pipeline(PoseFlowPipeline):
    """Training wrapper for the recurrent joint box-and-pose PoseFlow state."""

    def _losses_for_batch(
        self,
        batch: dict[str, Tensor],
        *,
        collect_codebook_attention: bool = False,
    ) -> dict[str, Tensor]:
        del collect_codebook_attention
        model = self.model
        if not isinstance(model, PoseFlowV3Model):
            raise TypeError("PoseFlowV3Pipeline requires PoseFlowV3Model")
        person_state, _, times, targets = build_poseflow_v3_targets(
            batch,
            model,
            num_queries=model.num_queries,
            num_body_points=model.transformer.decoder.num_body_points,
            padding_policy=self.flow_padding_policy,
            noise_seed=self.diffusion_noise_seed,
        )
        rng_context = nullcontext() if self.deterministic_seed is None else torch.random.fork_rng(
            devices=[batch["image"].device] if batch["image"].is_cuda else []
        )
        with rng_context:
            if self.deterministic_seed is not None:
                torch.manual_seed(self.deterministic_seed)
            outputs = model(
                batch["image"],
                person_state[..., 4:],
                times,
                targets,
                batch.get("padding_mask"),
                person_state=person_state,
            )
        self._latest_codebook_diagnostics = None
        return self.criterion(outputs, targets)


class PoseFlowV4Pipeline(PoseFlowPipeline):
    """Training wrapper for V4's independent image-token hypotheses."""

    def __init__(self, *args, corner_order_loss_weight: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.corner_order_loss_weight = corner_order_loss_weight
        self._latest_v4_diagnostics: dict[str, Tensor] | None = None

    def _log_v4_diagnostics(self, prefix: str, batch_size: int) -> None:
        if self._latest_v4_diagnostics is None:
            return
        for name, value in self._latest_v4_diagnostics.items():
            self.log(
                f"debug/{prefix}/{name}",
                value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

    def _log_codebook_diagnostics(self, prefix: str, batch_size: int) -> None:
        super()._log_codebook_diagnostics(prefix, batch_size)
        self._log_v4_diagnostics(prefix, batch_size)

    def _losses_for_batch(
        self,
        batch: dict[str, Tensor],
        *,
        collect_codebook_attention: bool = False,
    ) -> dict[str, Tensor]:
        del collect_codebook_attention
        model = self.model
        if not isinstance(model, PoseFlowV4Model):
            raise TypeError("PoseFlowV4Pipeline requires PoseFlowV4Model")
        person_state, _, times, targets = build_poseflow_v3_targets(
            batch,
            model,
            num_queries=model.num_queries,
            num_body_points=model.transformer.decoder.num_body_points,
            padding_policy=self.flow_padding_policy,
            noise_seed=self.diffusion_noise_seed,
        )
        rng_context = nullcontext() if self.deterministic_seed is None else torch.random.fork_rng(
            devices=[batch["image"].device] if batch["image"].is_cuda else []
        )
        with rng_context:
            if self.deterministic_seed is not None:
                torch.manual_seed(self.deterministic_seed)
            outputs = model(
                batch["image"],
                person_state[..., 4:],
                times,
                targets,
                batch.get("padding_mask"),
                person_state=person_state,
                collect_diagnostics=True,
            )
        self._latest_v4_diagnostics = model.last_v4_diagnostics()
        losses = self.criterion(outputs, targets)
        penalty = outputs.get("corner_order_penalty")
        if not isinstance(penalty, Tensor):
            raise RuntimeError("PoseFlow V4 output omitted corner-order penalty")
        losses["loss_corner_order"] = penalty
        losses["total"] = losses["total"] + self.corner_order_loss_weight * penalty
        return losses


class PoseFlowPipelineConfig(DiffusionRegPosePipelineConfig):
    """PoseFlow configuration; OTA is the default task-loss assignment."""

    matching_strategy: Literal["hungarian", "ota"] = "ota"
    flow_padding_policy: Literal["source_prior", "diffusionregpose"] = "source_prior"
    flow_sampling_steps: int = 1
    flow_renewal: bool = False
    flow_renewal_score_threshold: float = 0.5
    flow_ensemble: bool = False
    pose_codebook_size: int = 0
    pose_codebook_layers: tuple[int, ...] | None = None
    pose_codebook_initial_gate: float = 0.0

    @field_validator("flow_sampling_steps")
    @classmethod
    def _validate_flow_sampling_steps(cls, value: int) -> int:
        if value < 1:
            raise ValueError("flow_sampling_steps must be positive")
        return value

    @field_validator("flow_renewal_score_threshold")
    @classmethod
    def _validate_flow_renewal_score_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("flow_renewal_score_threshold must be in [0, 1]")
        return value

    @field_validator("pose_codebook_size")
    @classmethod
    def _validate_pose_codebook_size(cls, value: int) -> int:
        if value < 0:
            raise ValueError("pose_codebook_size must be non-negative")
        return value

    @field_validator("pose_codebook_initial_gate")
    @classmethod
    def _validate_pose_codebook_initial_gate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("pose_codebook_initial_gate must be finite")
        return value

    @model_validator(mode="after")
    def _validate_pose_codebook_layers(self) -> "PoseFlowPipelineConfig":
        layers = self.pose_codebook_layers
        if self.pose_codebook_size == 0:
            if layers:
                raise ValueError("pose_codebook_layers requires a positive pose_codebook_size")
            return self
        if layers is None:
            return self
        if len(set(layers)) != len(layers):
            raise ValueError("pose_codebook_layers must not contain duplicates")
        if any(layer < 0 or layer >= self.num_decoder_layers for layer in layers):
            raise ValueError("pose_codebook_layers must select decoder layers")
        return self

    def _build_poseflow_model(self) -> PoseFlowModel:
        return PoseFlowModel(
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
            separate_expansion_pose_head=self.pretrained_keep_decoder_layers is not None,
            flow_sampling_steps=self.flow_sampling_steps,
            flow_renewal=self.flow_renewal,
            flow_renewal_score_threshold=self.flow_renewal_score_threshold,
            flow_ensemble=self.flow_ensemble,
            pose_codebook_size=self.pose_codebook_size,
            pose_codebook_layers=self.pose_codebook_layers,
            pose_codebook_initial_gate=self.pose_codebook_initial_gate,
        )

    def build(self, **kwargs) -> PoseFlowPipeline:
        diffusion_pipeline = super().build(**kwargs)
        model = self._build_poseflow_model()
        if self.pose_codebook_size == 0:
            model.load_state_dict(diffusion_pipeline.model.state_dict(), strict=True)
        else:
            load_result = model.load_state_dict(diffusion_pipeline.model.state_dict(), strict=False)
            assert model.pose_codebook is not None
            expected_missing = {
                f"pose_codebook.{name}" for name in model.pose_codebook.state_dict()
            }
            if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
                raise RuntimeError("PoseFlow codebook initialization changed existing model tensors")
        return PoseFlowPipeline(
            model,
            diffusion_pipeline.criterion,
            lr=diffusion_pipeline.lr,
            lr_backbone=diffusion_pipeline.lr_backbone,
            weight_decay=diffusion_pipeline.weight_decay,
            max_epochs=diffusion_pipeline.max_epochs,
            lr_warmup_epochs=diffusion_pipeline.lr_warmup_epochs,
            lr_min_frac=diffusion_pipeline.lr_min_frac,
            scheduler_type=diffusion_pipeline.scheduler_type,
            lr_decay_milestones=diffusion_pipeline.lr_decay_milestones,
            split_weight_decay=diffusion_pipeline.split_weight_decay,
            val_ap_every_n_epochs=diffusion_pipeline.val_ap_every_n_epochs,
            val_ann_file=kwargs.get("val_ann_file"),
            diffusion_noise_seed=diffusion_pipeline.diffusion_noise_seed,
            deterministic_seed=diffusion_pipeline.deterministic_seed,
            flow_padding_policy=self.flow_padding_policy,
            flow_sampling_steps=self.flow_sampling_steps,
        )


class PoseFlowV3PipelineConfig(PoseFlowPipelineConfig):
    """Separate configuration and checkpoint family for joint-state PoseFlow."""

    flow_state_conditioning: bool = True

    @model_validator(mode="after")
    def _validate_v3_configuration(self) -> "PoseFlowV3PipelineConfig":
        if self.pose_codebook_size:
            raise ValueError("PoseFlow V3 does not combine the codebook experiment")
        if self.num_proposals != self.num_group:
            raise ValueError("PoseFlow V3 requires num_proposals to equal num_group")
        if self.num_proposals >= self.num_queries:
            raise ValueError("PoseFlow V3 requires num_queries to exceed num_proposals")
        if self.flow_renewal:
            raise ValueError("PoseFlow V3 uses its fixed 50-plus-fresh refill, not score renewal")
        return self

    def _build_poseflow_model(self) -> PoseFlowV3Model:
        return PoseFlowV3Model(
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
            separate_expansion_pose_head=self.pretrained_keep_decoder_layers is not None,
            flow_sampling_steps=self.flow_sampling_steps,
            flow_renewal=False,
            flow_renewal_score_threshold=self.flow_renewal_score_threshold,
            flow_ensemble=self.flow_ensemble,
            flow_state_conditioning=self.flow_state_conditioning,
        )

    def build(self, **kwargs) -> PoseFlowV3Pipeline:
        diffusion_pipeline = DiffusionRegPosePipelineConfig.build(self, **kwargs)
        model = self._build_poseflow_model()
        load_result = model.load_state_dict(diffusion_pipeline.model.state_dict(), strict=False)
        expected_missing = {
            f"person_state_embedding.{name}"
            for name in model.person_state_embedding.state_dict()
        }
        if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
            raise RuntimeError("PoseFlow V3 initialization changed shared model tensors")
        return PoseFlowV3Pipeline(
            model,
            diffusion_pipeline.criterion,
            lr=diffusion_pipeline.lr,
            lr_backbone=diffusion_pipeline.lr_backbone,
            weight_decay=diffusion_pipeline.weight_decay,
            max_epochs=diffusion_pipeline.max_epochs,
            lr_warmup_epochs=diffusion_pipeline.lr_warmup_epochs,
            lr_min_frac=diffusion_pipeline.lr_min_frac,
            scheduler_type=diffusion_pipeline.scheduler_type,
            lr_decay_milestones=diffusion_pipeline.lr_decay_milestones,
            split_weight_decay=diffusion_pipeline.split_weight_decay,
            val_ap_every_n_epochs=diffusion_pipeline.val_ap_every_n_epochs,
            val_ann_file=kwargs.get("val_ann_file"),
            diffusion_noise_seed=diffusion_pipeline.diffusion_noise_seed,
            deterministic_seed=diffusion_pipeline.deterministic_seed,
            flow_padding_policy=self.flow_padding_policy,
            flow_sampling_steps=self.flow_sampling_steps,
        )


class PoseFlowV4PipelineConfig(PoseFlowV3PipelineConfig):
    """Separate configuration and checkpoint family for PoseFlow V4."""

    fourier_bands: int = 8
    corner_order_loss_weight: float = 1.0

    @model_validator(mode="after")
    def _validate_v4_configuration(self) -> "PoseFlowV4PipelineConfig":
        super()._validate_v3_configuration()
        if self.fourier_bands < 1:
            raise ValueError("fourier_bands must be positive")
        if self.corner_order_loss_weight < 0.0:
            raise ValueError("corner_order_loss_weight must be non-negative")
        return self

    def _build_poseflow_model(self) -> PoseFlowV4Model:
        return PoseFlowV4Model(
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
            separate_expansion_pose_head=self.pretrained_keep_decoder_layers is not None,
            flow_sampling_steps=self.flow_sampling_steps,
            flow_renewal=False,
            flow_renewal_score_threshold=self.flow_renewal_score_threshold,
            flow_ensemble=self.flow_ensemble,
            flow_state_conditioning=self.flow_state_conditioning,
            fourier_bands=self.fourier_bands,
            corner_order_loss_weight=self.corner_order_loss_weight,
        )

    def build(self, **kwargs) -> PoseFlowV4Pipeline:
        diffusion_pipeline = DiffusionRegPosePipelineConfig.build(self, **kwargs)
        model = self._build_poseflow_model()
        load_result = model.load_state_dict(diffusion_pipeline.model.state_dict(), strict=False)
        expected_missing = {
            f"hypothesis_stage.{name}" for name in model.hypothesis_stage.state_dict()
        } | {
            f"initial_coordinate_head.{name}" for name in model.initial_coordinate_head.state_dict()
        } | {
            f"initial_class_head.{name}" for name in model.initial_class_head.state_dict()
        } | {
            f"v4_box_heads.{name}" for name in model.v4_box_heads.state_dict()
        } | {
            f"v4_class_heads.{name}" for name in model.v4_class_heads.state_dict()
        } | {
            f"v4_pose_heads.{name}" for name in model.v4_pose_heads.state_dict()
        } | {
            f"v4_pose_hw_heads.{name}" for name in model.v4_pose_hw_heads.state_dict()
        }
        if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
            raise RuntimeError("PoseFlow V4 initialization changed shared model tensors")
        return PoseFlowV4Pipeline(
            model,
            diffusion_pipeline.criterion,
            lr=diffusion_pipeline.lr,
            lr_backbone=diffusion_pipeline.lr_backbone,
            weight_decay=diffusion_pipeline.weight_decay,
            max_epochs=diffusion_pipeline.max_epochs,
            lr_warmup_epochs=diffusion_pipeline.lr_warmup_epochs,
            lr_min_frac=diffusion_pipeline.lr_min_frac,
            scheduler_type=diffusion_pipeline.scheduler_type,
            lr_decay_milestones=diffusion_pipeline.lr_decay_milestones,
            split_weight_decay=diffusion_pipeline.split_weight_decay,
            val_ap_every_n_epochs=diffusion_pipeline.val_ap_every_n_epochs,
            val_ann_file=kwargs.get("val_ann_file"),
            diffusion_noise_seed=diffusion_pipeline.diffusion_noise_seed,
            deterministic_seed=diffusion_pipeline.deterministic_seed,
            flow_padding_policy=self.flow_padding_policy,
            flow_sampling_steps=self.flow_sampling_steps,
            corner_order_loss_weight=self.corner_order_loss_weight,
        )


class PoseFlowV5Pipeline(PoseFlowV4Pipeline):
    """V5 wrapper with augmented training sources and center supervision."""

    def __init__(
        self,
        *args,
        center_loss_factor: float,
        decoder_loss_mode: Literal["all", "first_last"],
        lr_cycle_restart_epoch: int,
        lr_cycle_first_min: float,
        lr_cycle_restart_peak: float,
        lr_cycle_final_min: float,
        solution_feature_diagnostics: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.center_loss_factor = center_loss_factor
        self.decoder_loss_mode = decoder_loss_mode
        self.lr_cycle_restart_epoch = lr_cycle_restart_epoch
        self.lr_cycle_first_min = lr_cycle_first_min
        self.lr_cycle_restart_peak = lr_cycle_restart_peak
        self.lr_cycle_final_min = lr_cycle_final_min
        self.solution_feature_diagnostics = solution_feature_diagnostics
        self.lr_warmup_epochs_by_group: dict[str, int] | None = None

    def configure_optimizers(self):
        return build_adamw_with_schedule(
            self,
            lr=self.lr,
            lr_backbone=self.lr_backbone,
            weight_decay=self.weight_decay,
            lr_T_max=self.max_epochs,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_warmup_epochs_by_group=self.lr_warmup_epochs_by_group,
            lr_min_frac=self.lr_min_frac,
            scheduler_type=self.scheduler_type,
            lr_cycle_restart_epoch=self.lr_cycle_restart_epoch,
            lr_cycle_first_min=self.lr_cycle_first_min,
            lr_cycle_restart_peak=self.lr_cycle_restart_peak,
            lr_cycle_final_min=self.lr_cycle_final_min,
            split_weight_decay=self.split_weight_decay,
        )

    def _validation_sample(
        self, image: Tensor, padding_mask: Tensor | None,
        generator: torch.Generator | None, *, collect_diagnostics: bool,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        if (type(self.model) not in (PoseFlowV5Model, PoseFlowV9Model, PoseFlowV10Model, PoseFlowV11Model)
                or self.model.flow_ensemble):
            return self.sample_keypoints(image, padding_mask=padding_mask, generator=generator)
        self._validation_latent_rows = []
        self._validation_attention_rows = []
        self._validation_feature_outputs = {}
        if collect_diagnostics and self.solution_feature_diagnostics:
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state(image.device) if image.is_cuda else None
            generator_state = generator.get_state() if generator is not None else None
        if collect_diagnostics and isinstance(self.model, PoseFlowV9Model):
            from src.eval.poseflow_diagnostics import capture_v9_attention

            with capture_v9_attention(self.model) as rows:
                output = self.model.sample_keypoints(
                    image, padding_mask=padding_mask, generator=generator,
                    collect_step_diagnostics=True,
                )
            self._validation_attention_rows = rows
        elif collect_diagnostics and isinstance(self.model, PoseFlowV10Model):
            from src.eval.poseflow_diagnostics import capture_v10_latents

            with capture_v10_latents(self.model) as rows:
                output = self.model.sample_keypoints(
                    image, padding_mask=padding_mask, generator=generator,
                    collect_step_diagnostics=True,
                )
            self._validation_latent_rows = rows
        else:
            output = self.model.sample_keypoints(
                image, padding_mask=padding_mask, generator=generator,
                collect_step_diagnostics=collect_diagnostics,
            )
        if collect_diagnostics and self.solution_feature_diagnostics:
            from src.eval.poseflow_diagnostics import FEATURE_PATHS, ablate_poseflow_path

            devices = [image.device] if image.is_cuda else []
            for path in FEATURE_PATHS:
                with torch.random.fork_rng(devices=devices):
                    torch.random.set_rng_state(cpu_rng_state)
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state(cuda_rng_state, image.device)
                    replay_generator = None
                    if generator_state is not None:
                        replay_generator = torch.Generator(device=image.device)
                        replay_generator.set_state(generator_state)
                    with ablate_poseflow_path(self.model, path):
                        self._validation_feature_outputs[path] = self.model.sample_keypoints(
                            image, padding_mask=padding_mask, generator=replay_generator,
                        )
        return output

    def _log_validation_solution(
        self, batch: dict[str, Tensor], output: dict,
        results: list[dict[str, Tensor]], *, batch_index: int,
    ) -> None:
        if batch_index != 0 or type(self.model) not in (
            PoseFlowV5Model, PoseFlowV9Model, PoseFlowV10Model, PoseFlowV11Model,
        ):
            return
        if self.model.flow_ensemble:
            self.log(
                "debug/solution/val_first_batch/ensemble_skipped", 1.0,
                on_step=False, on_epoch=True, sync_dist=True, batch_size=1,
            )
            return
        from src.eval.metrics import CROWDPOSE_SIGMAS
        from src.eval.poseflow_training_diagnostics import (
            summarize_feature_intervention, summarize_validation_solution,
        )

        sigmas = torch.as_tensor(CROWDPOSE_SIGMAS, device=batch["image"].device)
        values = summarize_validation_solution(batch, output, results, sigmas)
        values["flow/sampling_steps"] = float(self.flow_sampling_steps)
        values["flow/deterministic_seed"] = float(
            -1 if self.deterministic_seed is None else self.deterministic_seed
        )
        for step, row in enumerate(output.get("flow_step_diagnostics", [])):
            for layer, diagnostics in enumerate(row.get("token_diagnostics", [])):
                for name, value in diagnostics.items():
                    values[f"tokens/step_{step}/layer_{layer}/{name}"] = float(value)
        for row in getattr(self, "_validation_latent_rows", []):
            prefix = f"latent/step_{row['step']}/layer_{row['layer']}"
            for name, value in row.items():
                if name not in ("step", "layer"):
                    values[f"{prefix}/{name}"] = float(value)
        for row in getattr(self, "_validation_attention_rows", []):
            prefix = f"attention/step_{row['step']}/layer_{row['layer']}"
            for name in ("off_edge_bias", "raw_logit_std", "nonedge_attention_mass"):
                values[f"{prefix}/{name}"] = float(row[name])
        for path, changed in getattr(self, "_validation_feature_outputs", {}).items():
            feature_values = summarize_feature_intervention(batch, output, changed, sigmas)
            for name, value in feature_values.items():
                values[f"feature/{path}/{name}"] = value
        for name, value in values.items():
            self.log(
                f"debug/solution/val_first_batch/{name}", value,
                on_step=False, on_epoch=True, sync_dist=True,
                batch_size=1,
            )

    def _log_v4_diagnostics(self, prefix: str, batch_size: int) -> None:
        if self._latest_v4_diagnostics is None:
            return
        for name, value in self._latest_v4_diagnostics.items():
            self.log(
                f"debug/{prefix}/{name}", value,
                batch_size=batch_size, on_step=False, on_epoch=True,
            )

    def _center_losses(
        self, outputs: dict[str, Tensor | list[dict[str, Tensor]]], targets: list[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        predictions = [*outputs.get("aux_outputs", []), outputs]
        center_losses: dict[str, Tensor] = {}
        for stage_index, prediction in enumerate(predictions):
            if not isinstance(prediction, dict):
                raise RuntimeError("V5 auxiliary prediction must be a dictionary")
            points = prediction.get("pred_box_points")
            if not isinstance(points, Tensor):
                raise RuntimeError("V5 prediction omitted box points")
            matches = self.criterion.matcher(prediction, targets)
            normalizer = max(sum(target["labels"].numel() for target in targets), 1)
            direct = points.new_zeros(())
            midpoint = points.new_zeros(())
            for batch_index, (source, target_indices) in enumerate(matches):
                if not source.numel():
                    continue
                target_center = targets[batch_index]["boxes"].index_select(0, target_indices)[:, :2]
                predicted_corners = points[batch_index].index_select(0, source)[:, :4]
                predicted_center = points[batch_index].index_select(0, source)[:, 4:]
                direct = direct + (predicted_center - target_center).abs().mean(dim=-1).sum()
                midpoint = midpoint + (
                    (predicted_corners[:, :2] + predicted_corners[:, 2:]) / 2.0 - target_center
                ).abs().mean(dim=-1).sum()
            suffix = f"_{stage_index}" if stage_index < len(predictions) - 1 else ""
            center_losses[f"loss_center_direct{suffix}"] = direct / normalizer
            center_losses[f"loss_center_midpoint{suffix}"] = midpoint / normalizer
        return center_losses

    def _first_last_total(self, losses: dict[str, Tensor]) -> Tensor:
        first_losses = {
            name: losses[f"{name}_0"]
            for name in ("loss_ce", "loss_bbox", "loss_giou", "loss_keypoints", "loss_oks")
        }
        total = self.criterion._weighted_clean_loss(losses)
        total = total + self.criterion._weighted_clean_loss(first_losses)
        if "dn_loss_ce" in losses:
            total = total + (
                self.criterion.dn_label_coef * losses["dn_loss_ce"]
                + self.criterion.weights.bbox * self.criterion.dn_bbox_coef * losses["dn_loss_bbox"]
                + self.criterion.weights.giou * self.criterion.dn_bbox_coef * losses["dn_loss_giou"]
            )
        center_weight = self.center_loss_factor * self.criterion.weights.giou
        for suffix in ("_0", ""):
            total = total + center_weight * (
                losses[f"loss_center_direct{suffix}"]
                + losses[f"loss_center_midpoint{suffix}"]
            )
        return total + self.corner_order_loss_weight * losses["loss_corner_order"]

    def _losses_for_batch(
        self,
        batch: dict[str, Tensor],
        *,
        collect_codebook_attention: bool = False,
    ) -> dict[str, Tensor]:
        del collect_codebook_attention
        model = self.model
        if not isinstance(model, PoseFlowV5Model):
            raise TypeError("PoseFlowV5Pipeline requires PoseFlowV5Model")
        person_state, _, times, targets = build_poseflow_v5_targets(
            batch,
            model,
            num_queries=model.num_queries,
            num_body_points=model.transformer.decoder.num_body_points,
            padding_policy=self.flow_padding_policy,
            noise_seed=self.diffusion_noise_seed,
        )
        rng_context = nullcontext() if self.deterministic_seed is None else torch.random.fork_rng(
            devices=[batch["image"].device] if batch["image"].is_cuda else []
        )
        with rng_context:
            if self.deterministic_seed is not None:
                torch.manual_seed(self.deterministic_seed)
            outputs = model(
                batch["image"], person_state[..., 4:], times, targets,
                batch.get("padding_mask"), person_state=person_state, collect_diagnostics=True,
            )
        self._latest_v4_diagnostics = model.last_v4_diagnostics()
        losses = self.criterion(outputs, targets)
        penalty = outputs.get("corner_order_penalty")
        if not isinstance(penalty, Tensor):
            raise RuntimeError("PoseFlow V5 output omitted corner-order penalty")
        losses["loss_corner_order"] = penalty
        losses["total"] = losses["total"] + self.corner_order_loss_weight * penalty
        center_losses = self._center_losses(outputs, targets)
        losses.update(center_losses)
        center_weight = self.center_loss_factor * self.criterion.weights.giou
        losses["total"] = losses["total"] + center_weight * sum(center_losses.values())
        if self.decoder_loss_mode == "first_last":
            losses["total"] = self._first_last_total(losses)
        return losses


class PoseFlowV5PipelineConfig(PoseFlowV4PipelineConfig):
    """Distinct V5 checkpoint family for source augmentation and center losses."""

    lr: float = 1e-4
    lr_backbone: float | None = 5e-5
    scheduler_type: Literal["two_cycle_cosine"] = "two_cycle_cosine"
    lr_cycle_restart_epoch: int = 40
    lr_cycle_first_min: float = 1e-5
    lr_cycle_restart_peak: float = 3e-5
    lr_cycle_final_min: float = 2e-6
    center_loss_factor: float = 0.1
    decoder_loss_mode: Literal["all", "first_last"] = "all"
    solution_feature_diagnostics: bool = False

    @model_validator(mode="after")
    def _validate_v5_configuration(self) -> "PoseFlowV5PipelineConfig":
        super()._validate_v4_configuration()
        if self.lr_cycle_restart_epoch < 1:
            raise ValueError("lr_cycle_restart_epoch must be positive")
        if min(
            self.lr_cycle_first_min,
            self.lr_cycle_restart_peak,
            self.lr_cycle_final_min,
            self.center_loss_factor,
        ) < 0.0:
            raise ValueError("V5 learning rates and center_loss_factor must be non-negative")
        return self

    def _poseflow_model_kwargs(self) -> dict[str, object]:
        return {
            "backbone_pretrained": self.backbone_pretrained,
            "timesteps": self.timesteps,
            "sampling_timesteps": self.sampling_timesteps,
            "num_queries": self.num_queries,
            "num_proposals": self.num_proposals,
            "num_classes": self.num_classes,
            "num_feature_levels": self.num_feature_levels,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "hidden_dim": self.hidden_dim,
            "feedforward_dim": self.feedforward_dim,
            "num_heads": self.num_heads,
            "num_body_points": self.num_body_points,
            "num_box_decoder_layers": self.num_box_decoder_layers,
            "num_group": self.num_group,
            "dn_number": self.dn_number,
            "dn_box_noise_scale": self.dn_box_noise_scale,
            "dn_label_noise_ratio": self.dn_label_noise_ratio,
            "freeze_backbone_stem_and_layer1": self.freeze_backbone_stem_and_layer1,
            "freeze_backbone": self.freeze_backbone,
            "freeze_backbone_and_transformer": self.freeze_backbone_and_transformer,
            "separate_expansion_pose_head": self.pretrained_keep_decoder_layers is not None,
            "flow_sampling_steps": self.flow_sampling_steps,
            "flow_renewal": False,
            "flow_renewal_score_threshold": self.flow_renewal_score_threshold,
            "flow_ensemble": self.flow_ensemble,
            "flow_state_conditioning": self.flow_state_conditioning,
            "fourier_bands": self.fourier_bands,
            "corner_order_loss_weight": self.corner_order_loss_weight,
        }

    def _build_poseflow_model(self) -> PoseFlowV5Model:
        return PoseFlowV5Model(**self._poseflow_model_kwargs())

    def _additional_initialization_missing_keys(self, model: PoseFlowV5Model) -> set[str]:
        return set()

    def _pipeline_class(self) -> type[PoseFlowV5Pipeline]:
        return PoseFlowV5Pipeline

    def build(self, **kwargs) -> PoseFlowV5Pipeline:
        diffusion_pipeline = DiffusionRegPosePipelineConfig.build(self, **kwargs)
        model = self._build_poseflow_model()
        load_result = model.load_state_dict(diffusion_pipeline.model.state_dict(), strict=False)
        expected_missing = {
            f"hypothesis_stage.{name}" for name in model.hypothesis_stage.state_dict()
        } | {
            f"initial_coordinate_head.{name}" for name in model.initial_coordinate_head.state_dict()
        } | {
            f"initial_class_head.{name}" for name in model.initial_class_head.state_dict()
        } | {
            f"v5_box_heads.{name}" for name in model.v5_box_heads.state_dict()
        } | {
            f"v4_class_heads.{name}" for name in model.v4_class_heads.state_dict()
        } | {
            f"v4_pose_heads.{name}" for name in model.v4_pose_heads.state_dict()
        } | {
            f"v4_pose_hw_heads.{name}" for name in model.v4_pose_hw_heads.state_dict()
        }
        expected_missing |= self._additional_initialization_missing_keys(model)
        if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
            raise RuntimeError("PoseFlow V5 initialization changed shared model tensors")
        return self._pipeline_class()(
            model,
            diffusion_pipeline.criterion,
            lr=diffusion_pipeline.lr,
            lr_backbone=diffusion_pipeline.lr_backbone,
            weight_decay=diffusion_pipeline.weight_decay,
            max_epochs=diffusion_pipeline.max_epochs,
            lr_warmup_epochs=diffusion_pipeline.lr_warmup_epochs,
            lr_min_frac=diffusion_pipeline.lr_min_frac,
            scheduler_type=self.scheduler_type,
            lr_decay_milestones=diffusion_pipeline.lr_decay_milestones,
            split_weight_decay=diffusion_pipeline.split_weight_decay,
            val_ap_every_n_epochs=diffusion_pipeline.val_ap_every_n_epochs,
            val_ann_file=kwargs.get("val_ann_file"),
            diffusion_noise_seed=diffusion_pipeline.diffusion_noise_seed,
            deterministic_seed=diffusion_pipeline.deterministic_seed,
            flow_padding_policy=self.flow_padding_policy,
            flow_sampling_steps=self.flow_sampling_steps,
            corner_order_loss_weight=self.corner_order_loss_weight,
            center_loss_factor=self.center_loss_factor,
            decoder_loss_mode=self.decoder_loss_mode,
            lr_cycle_restart_epoch=self.lr_cycle_restart_epoch,
            lr_cycle_first_min=self.lr_cycle_first_min,
            lr_cycle_restart_peak=self.lr_cycle_restart_peak,
            lr_cycle_final_min=self.lr_cycle_final_min,
            solution_feature_diagnostics=self.solution_feature_diagnostics,
        )


class PoseFlowV9Pipeline(PoseFlowV5Pipeline):
    """V5 training and evaluation with skeleton-biased joint attention."""


class PoseFlowV9PipelineConfig(PoseFlowV5PipelineConfig):
    """Distinct V9 checkpoint family using the V5 initialization path."""

    def _build_poseflow_model(self) -> PoseFlowV9Model:
        return PoseFlowV9Model(**self._poseflow_model_kwargs())

    def _additional_initialization_missing_keys(self, model: PoseFlowV5Model) -> set[str]:
        return {"off_edge_attention_bias"}

    def _pipeline_class(self) -> type[PoseFlowV5Pipeline]:
        return PoseFlowV9Pipeline


class PoseFlowV10Pipeline(PoseFlowV5Pipeline):
    """Whole-person latent refinement with configured LR scheduling."""

    lr_main_warmup_epochs = 1
    lr_backbone_warmup_epochs = 10

    def on_train_epoch_start(self) -> None:
        for group in self.trainer.optimizers[0].param_groups:
            self.log(f"debug/lr/{group['schedule_group']}", group["lr"],
                     on_step=False, on_epoch=True, sync_dist=True)


class PoseFlowV10PipelineConfig(PoseFlowV5PipelineConfig):
    scheduler_type: Literal["cosine", "two_cycle_cosine"] = "cosine"
    lr_backbone: float | None = 1e-5
    lr_main_warmup_epochs: int = 1
    lr_backbone_warmup_epochs: int = 10

    @model_validator(mode="after")
    def _validate_v10_warmups(self):
        if self.lr_main_warmup_epochs < 0 or self.lr_backbone_warmup_epochs < 0:
            raise ValueError("V10 warmup durations must be non-negative")
        if self.scheduler_type == "two_cycle_cosine" and (
            self.lr_main_warmup_epochs or self.lr_backbone_warmup_epochs
            or self.lr_warmup_epochs
        ):
            raise ValueError("V10 two-cycle scheduling requires zero warmup")
        return self

    def build(self, **kwargs) -> PoseFlowV10Pipeline:
        pipeline = super().build(**kwargs)
        pipeline.lr_main_warmup_epochs = self.lr_main_warmup_epochs
        pipeline.lr_backbone_warmup_epochs = self.lr_backbone_warmup_epochs
        if self.scheduler_type == "cosine":
            pipeline.lr_warmup_epochs_by_group = {
                "main": self.lr_main_warmup_epochs,
                "backbone": self.lr_backbone_warmup_epochs,
            }
        return pipeline

    def _build_poseflow_model(self) -> PoseFlowV10Model:
        return PoseFlowV10Model(**self._poseflow_model_kwargs())

    def _additional_initialization_missing_keys(self, model: PoseFlowV5Model) -> set[str]:
        return {f"dn_box_conditioning.{name}" for name in model.dn_box_conditioning.state_dict()}

    def _pipeline_class(self) -> type[PoseFlowV5Pipeline]:
        return PoseFlowV10Pipeline


class PoseFlowV6Pipeline(PoseFlowV5Pipeline):
    """V5 flow training with DINO multi-scale decoder fusion."""

    def __init__(
        self,
        *args,
        lr_main_warmup_epochs: int,
        lr_backbone_warmup_epochs: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lr_main_warmup_epochs = lr_main_warmup_epochs
        self.lr_backbone_warmup_epochs = lr_backbone_warmup_epochs

    def configure_optimizers(self):
        return build_adamw_with_schedule(
            self,
            lr=self.lr,
            lr_backbone=self.lr_backbone,
            weight_decay=self.weight_decay,
            lr_T_max=self.max_epochs,
            lr_min_frac=self.lr_min_frac,
            scheduler_type="cosine",
            lr_warmup_epochs_by_group={
                "main": self.lr_main_warmup_epochs,
                "backbone": self.lr_backbone_warmup_epochs,
            },
            split_weight_decay=self.split_weight_decay,
        )


class PoseFlowV6PipelineConfig(PoseFlowV5PipelineConfig):
    """Distinct V6 checkpoint family with frozen DINOv3 stage features."""

    scheduler_type: Literal["cosine"] = "cosine"
    lr: float = 1e-4
    lr_backbone: float | None = 1e-5
    lr_main_warmup_epochs: int = 1
    lr_backbone_warmup_epochs: int = 10
    dino_model_name: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
    dino_frozen: bool = True
    dino_hidden_state_indices: tuple[int, ...] = (2, 3, 4)
    dino_feature_channels: tuple[int, ...] = (192, 384, 768)

    @model_validator(mode="after")
    def _validate_v6_schedule(self) -> "PoseFlowV6PipelineConfig":
        if self.lr_main_warmup_epochs < 0 or self.lr_backbone_warmup_epochs < 0:
            raise ValueError("V6 warmup durations must be non-negative")
        if not self.dino_hidden_state_indices or len(self.dino_hidden_state_indices) != len(
            self.dino_feature_channels
        ):
            raise ValueError("V6 DINO stage indices and channels must be non-empty and aligned")
        return self

    def _build_poseflow_model(self) -> PoseFlowV6Model:
        return PoseFlowV6Model(
            **self._poseflow_model_kwargs(),
            dino_model_name=self.dino_model_name,
            dino_frozen=self.dino_frozen,
            dino_hidden_state_indices=self.dino_hidden_state_indices,
            dino_feature_channels=self.dino_feature_channels,
        )

    def build(self, **kwargs) -> PoseFlowV6Pipeline:
        diffusion_pipeline = DiffusionRegPosePipelineConfig.build(self, **kwargs)
        model = self._build_poseflow_model()
        load_result = model.load_state_dict(
            diffusion_pipeline.model.state_dict(), strict=False
        )
        expected_missing = {
            f"{prefix}.{name}"
            for prefix in (
                "hypothesis_stage",
                "initial_coordinate_head",
                "initial_class_head",
                "v5_box_heads",
                "v4_class_heads",
                "v4_pose_heads",
                "v4_pose_hw_heads",
                "dino_backbone",
                "dino_input_projections",
                "dino_decoder_fusion",
            )
            for name in getattr(model, prefix).state_dict()
        }
        if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
            raise RuntimeError("PoseFlow V6 initialization changed shared model tensors")
        return PoseFlowV6Pipeline(
            model,
            diffusion_pipeline.criterion,
            lr=diffusion_pipeline.lr,
            lr_backbone=diffusion_pipeline.lr_backbone,
            weight_decay=diffusion_pipeline.weight_decay,
            max_epochs=diffusion_pipeline.max_epochs,
            lr_warmup_epochs=diffusion_pipeline.lr_warmup_epochs,
            lr_min_frac=diffusion_pipeline.lr_min_frac,
            scheduler_type=self.scheduler_type,
            lr_decay_milestones=diffusion_pipeline.lr_decay_milestones,
            split_weight_decay=diffusion_pipeline.split_weight_decay,
            val_ap_every_n_epochs=diffusion_pipeline.val_ap_every_n_epochs,
            val_ann_file=kwargs.get("val_ann_file"),
            diffusion_noise_seed=diffusion_pipeline.diffusion_noise_seed,
            deterministic_seed=diffusion_pipeline.deterministic_seed,
            flow_padding_policy=self.flow_padding_policy,
            flow_sampling_steps=self.flow_sampling_steps,
            corner_order_loss_weight=self.corner_order_loss_weight,
            center_loss_factor=self.center_loss_factor,
            decoder_loss_mode=self.decoder_loss_mode,
            lr_cycle_restart_epoch=self.lr_cycle_restart_epoch,
            lr_cycle_first_min=self.lr_cycle_first_min,
            lr_cycle_restart_peak=self.lr_cycle_restart_peak,
            lr_cycle_final_min=self.lr_cycle_final_min,
            lr_main_warmup_epochs=self.lr_main_warmup_epochs,
            lr_backbone_warmup_epochs=self.lr_backbone_warmup_epochs,
        )


class PoseFlowV7Pipeline(PoseFlowV6Pipeline):
    """V6 training contract with a smaller Tiny DINOv3 branch."""


class PoseFlowV7PipelineConfig(PoseFlowV6PipelineConfig):
    """Distinct V7 checkpoint family using DINOv3 ConvNeXt-Tiny stages 2 and 4."""

    num_feature_levels: Literal[2] = 2
    dino_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    dino_hidden_state_indices: tuple[int, ...] = (2, 4)
    dino_feature_channels: tuple[int, ...] = (192, 768)

    def _build_poseflow_model(self) -> PoseFlowV7Model:
        return PoseFlowV7Model(
            **self._poseflow_model_kwargs(),
            dino_model_name=self.dino_model_name,
            dino_frozen=self.dino_frozen,
            dino_hidden_state_indices=self.dino_hidden_state_indices,
            dino_feature_channels=self.dino_feature_channels,
        )

    def build(self, **kwargs) -> PoseFlowV7Pipeline:
        diffusion_pipeline = DiffusionRegPosePipelineConfig.build(self, **kwargs)
        model = self._build_poseflow_model()
        load_result = model.load_state_dict(diffusion_pipeline.model.state_dict(), strict=False)
        expected_missing = {
            f"{prefix}.{name}"
            for prefix in (
                "hypothesis_stage",
                "initial_coordinate_head",
                "initial_class_head",
                "v5_box_heads",
                "v4_class_heads",
                "v4_pose_heads",
                "v4_pose_hw_heads",
                "dino_backbone",
                "dino_input_projections",
                "dino_decoder_fusion",
            )
            for name in getattr(model, prefix).state_dict()
        }
        if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
            raise RuntimeError("PoseFlow V7 initialization changed shared model tensors")
        return PoseFlowV7Pipeline(
            model,
            diffusion_pipeline.criterion,
            lr=diffusion_pipeline.lr,
            lr_backbone=diffusion_pipeline.lr_backbone,
            weight_decay=diffusion_pipeline.weight_decay,
            max_epochs=diffusion_pipeline.max_epochs,
            lr_warmup_epochs=diffusion_pipeline.lr_warmup_epochs,
            lr_min_frac=diffusion_pipeline.lr_min_frac,
            scheduler_type=self.scheduler_type,
            lr_decay_milestones=diffusion_pipeline.lr_decay_milestones,
            split_weight_decay=diffusion_pipeline.split_weight_decay,
            val_ap_every_n_epochs=diffusion_pipeline.val_ap_every_n_epochs,
            val_ann_file=kwargs.get("val_ann_file"),
            diffusion_noise_seed=diffusion_pipeline.diffusion_noise_seed,
            deterministic_seed=diffusion_pipeline.deterministic_seed,
            flow_padding_policy=self.flow_padding_policy,
            flow_sampling_steps=self.flow_sampling_steps,
            corner_order_loss_weight=self.corner_order_loss_weight,
            center_loss_factor=self.center_loss_factor,
            decoder_loss_mode=self.decoder_loss_mode,
            lr_cycle_restart_epoch=self.lr_cycle_restart_epoch,
            lr_cycle_first_min=self.lr_cycle_first_min,
            lr_cycle_restart_peak=self.lr_cycle_restart_peak,
            lr_cycle_final_min=self.lr_cycle_final_min,
            lr_main_warmup_epochs=self.lr_main_warmup_epochs,
            lr_backbone_warmup_epochs=self.lr_backbone_warmup_epochs,
        )


class PoseFlowV8Pipeline(PoseFlowV7Pipeline):
    """V7 training contract with independent DINO coarse/fine feature adapters."""


class PoseFlowV8PipelineConfig(PoseFlowV7PipelineConfig):
    """DINO-only four-level feature pyramid with separate coarse and fine paths."""

    num_feature_levels: Literal[4] = 4
    num_encoder_feature_levels: Literal[3] = 3
    dino_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    dino_hidden_state_indices: tuple[int, ...] = (1, 2, 3, 4)
    dino_feature_channels: tuple[int, ...] = (96, 192, 384, 768)
    dino_coarse_hidden_state_indices: tuple[int, ...] = (2, 3, 4)

    @model_validator(mode="after")
    def _validate_v8_dino_stages(self) -> "PoseFlowV8PipelineConfig":
        if self.dino_hidden_state_indices != (1, 2, 3, 4):
            raise ValueError("PoseFlow V8 requires all four DINO ConvNeXt stages")
        if self.dino_feature_channels != (96, 192, 384, 768):
            raise ValueError("PoseFlow V8 requires DINO stage channels [96, 192, 384, 768]")
        if self.dino_coarse_hidden_state_indices != (2, 3, 4):
            raise ValueError("PoseFlow V8 coarse features must use DINO stages [2, 3, 4]")
        return self

    def _build_poseflow_model(self) -> PoseFlowV8Model:
        model_kwargs = self._poseflow_model_kwargs()
        model_kwargs["backbone_pretrained"] = False
        model_kwargs["num_encoder_feature_levels"] = self.num_encoder_feature_levels
        return PoseFlowV8Model(
            **model_kwargs,
            dino_model_name=self.dino_model_name,
            dino_frozen=self.dino_frozen,
            dino_hidden_state_indices=self.dino_hidden_state_indices,
            dino_feature_channels=self.dino_feature_channels,
            dino_coarse_hidden_state_indices=self.dino_coarse_hidden_state_indices,
        )

    def build(self, **kwargs) -> PoseFlowV8Pipeline:
        source_config = self.model_copy(update={"backbone_pretrained": False})
        diffusion_pipeline = DiffusionRegPosePipelineConfig.build(source_config, **kwargs)
        model = self._build_poseflow_model()
        source_state = {
            key: value
            for key, value in diffusion_pipeline.model.state_dict().items()
            if not key.startswith(("backbone.", "input_proj."))
        }
        target_state = model.state_dict()
        source_state, adapted_encoder_keys = _adapt_v8_encoder_attention_levels(
            source_state,
            target_state,
            num_heads=self.num_heads,
        )
        load_result = model.load_state_dict(source_state, strict=False)
        expected_missing = {
            f"{prefix}.{name}"
            for prefix in (
                "hypothesis_stage",
                "initial_coordinate_head",
                "initial_class_head",
                "v5_box_heads",
                "v4_class_heads",
                "v4_pose_heads",
                "v4_pose_hw_heads",
                "dino_backbone",
                "dino_coarse_input_projections",
                "dino_fine_input_projections",
            )
            for name in getattr(model, prefix).state_dict()
        }
        if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
            raise RuntimeError("PoseFlow V8 initialization changed shared model tensors")
        if len(adapted_encoder_keys) != self.num_encoder_layers * 4:
            raise RuntimeError(
                "PoseFlow V8 expected to adapt four level-specific attention tensors "
                "per encoder layer"
            )
        return PoseFlowV8Pipeline(
            model,
            diffusion_pipeline.criterion,
            lr=diffusion_pipeline.lr,
            lr_backbone=diffusion_pipeline.lr_backbone,
            weight_decay=diffusion_pipeline.weight_decay,
            max_epochs=diffusion_pipeline.max_epochs,
            lr_warmup_epochs=diffusion_pipeline.lr_warmup_epochs,
            lr_min_frac=diffusion_pipeline.lr_min_frac,
            scheduler_type=self.scheduler_type,
            lr_decay_milestones=diffusion_pipeline.lr_decay_milestones,
            split_weight_decay=diffusion_pipeline.split_weight_decay,
            val_ap_every_n_epochs=diffusion_pipeline.val_ap_every_n_epochs,
            val_ann_file=kwargs.get("val_ann_file"),
            diffusion_noise_seed=diffusion_pipeline.diffusion_noise_seed,
            deterministic_seed=diffusion_pipeline.deterministic_seed,
            flow_padding_policy=self.flow_padding_policy,
            flow_sampling_steps=self.flow_sampling_steps,
            corner_order_loss_weight=self.corner_order_loss_weight,
            center_loss_factor=self.center_loss_factor,
            decoder_loss_mode=self.decoder_loss_mode,
            lr_cycle_restart_epoch=self.lr_cycle_restart_epoch,
            lr_cycle_first_min=self.lr_cycle_first_min,
            lr_cycle_restart_peak=self.lr_cycle_restart_peak,
            lr_cycle_final_min=self.lr_cycle_final_min,
            lr_main_warmup_epochs=self.lr_main_warmup_epochs,
            lr_backbone_warmup_epochs=self.lr_backbone_warmup_epochs,
        )
