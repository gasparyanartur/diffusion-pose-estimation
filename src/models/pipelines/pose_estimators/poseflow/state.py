"""Pure state operations for the joint box-and-pose PoseFlow variant."""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import Tensor


def fourier_encode_coordinates(coordinates: Tensor, *, num_bands: int) -> Tensor:
    """Encode each coordinate with fixed sine/cosine Fourier bands.

    Coordinates are intentionally neither clamped nor reordered: intermediate
    flow states may be out of frame or have crossed box corners.
    """
    if coordinates.ndim < 1:
        raise ValueError("coordinates must have a coordinate dimension")
    if num_bands < 1:
        raise ValueError("num_bands must be positive")
    frequencies = coordinates.new_tensor(
        [math.pi * (2**band) for band in range(num_bands)]
    )
    phases = coordinates[..., None] * frequencies
    return torch.cat((phases.sin(), phases.cos()), dim=-1).flatten(-2)


def cxcywh_to_corners(boxes: Tensor) -> Tensor:
    """Convert normalized ``cxcywh`` boxes to their two transported corners."""
    if boxes.shape[-1] != 4:
        raise ValueError("boxes must end in four cxcywh values")
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2.0, center + size / 2.0), dim=-1)


def corners_to_cxcywh(corners: Tensor) -> Tensor:
    """Derive decoder geometry without sorting, clamping, or projecting corners."""
    if corners.shape[-1] != 4:
        raise ValueError("corners must end in two xy points")
    corner_a, corner_b = corners[..., :2], corners[..., 2:]
    return torch.cat(((corner_a + corner_b) / 2.0, corner_b - corner_a), dim=-1)


def pack_person_state(corners: Tensor, keypoints: Tensor) -> Tensor:
    """Pack two box corners and flattened keypoint XY into one person state."""
    if corners.shape[:-1] != keypoints.shape[:-1]:
        raise ValueError("corners and keypoints must share batch and slot dimensions")
    if corners.shape[-1] != 4:
        raise ValueError("corners must end in two xy points")
    if keypoints.shape[-1] == 0 or keypoints.shape[-1] % 2:
        raise ValueError("keypoints must contain one or more xy pairs")
    return torch.cat((corners, keypoints), dim=-1)


def unpack_person_state(state: Tensor, num_body_points: int) -> tuple[Tensor, Tensor]:
    """Split a packed person state into corners and flattened keypoint XY."""
    expected_width = 4 + 2 * num_body_points
    if num_body_points < 1 or state.shape[-1] != expected_width:
        raise ValueError("state width must equal four box values plus keypoint XY")
    return state[..., :4], state[..., 4:]


def interpolate_person_state(source: Tensor, endpoint: Tensor, times: Tensor) -> Tensor:
    """Linearly interpolate paired source and endpoint states at per-image time."""
    if source.shape != endpoint.shape or source.ndim != 3:
        raise ValueError("source and endpoint must have matching [B, N, D] shapes")
    if times.shape != (source.shape[0],):
        raise ValueError("times must have one value per batch element")
    return (1.0 - times[:, None, None]) * source + times[:, None, None] * endpoint


def euler_update_person_state(
    state: Tensor, endpoint: Tensor, source: Tensor, *, sampling_steps: int
) -> Tensor:
    """Advance fixed-source endpoint flow by one forward-Euler step."""
    if sampling_steps < 1:
        raise ValueError("sampling_steps must be positive")
    if state.shape != endpoint.shape or state.shape != source.shape:
        raise ValueError("state, endpoint, and source must have matching shapes")
    return state + (endpoint - source) / sampling_steps


def merge_person_slot_fields(
    active: Mapping[str, Tensor], fresh: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """Append fresh slots to active slots while preserving paired field order."""
    if active.keys() != fresh.keys() or not active:
        raise ValueError("active and fresh fields must have the same non-empty keys")
    merged: dict[str, Tensor] = {}
    for name, active_values in active.items():
        fresh_values = fresh[name]
        if active_values.ndim < 2 or fresh_values.ndim != active_values.ndim:
            raise ValueError(f"{name} must have matching batch-and-slot tensor ranks")
        if active_values.shape[0] != fresh_values.shape[0] or active_values.shape[2:] != fresh_values.shape[2:]:
            raise ValueError(f"{name} must match outside the slot dimension")
        merged[name] = torch.cat((active_values, fresh_values), dim=1)
    return merged


def gather_person_slot_fields(
    fields: Mapping[str, Tensor], indices: Tensor
) -> dict[str, Tensor]:
    """Gather every per-slot field with one shared top-k selection."""
    if not fields:
        raise ValueError("fields must not be empty")
    if indices.ndim != 2 or indices.dtype != torch.long:
        raise ValueError("indices must be a [B, S] long tensor")
    gathered: dict[str, Tensor] = {}
    for name, values in fields.items():
        if values.ndim < 2 or values.shape[:2] != (indices.shape[0], values.shape[1]):
            raise ValueError(f"{name} must have matching batch and slot dimensions")
        if indices.numel() and (indices.min() < 0 or indices.max() >= values.shape[1]):
            raise ValueError("indices must select existing slots")
        index = indices[(...,) + (None,) * (values.ndim - 2)].expand(
            indices.shape + values.shape[2:]
        )
        gathered[name] = torch.gather(values, 1, index)
    return gathered


def state_reference_gate(times: Tensor) -> Tensor:
    """Use state geometry from 0% at t=0 through 50% at t=1."""
    if times.ndim != 1 or torch.any((times < 0.0) | (times > 1.0)):
        raise ValueError("times must be a [B] tensor in [0, 1]")
    return 0.5 * times


def blend_box_references(image_references: Tensor, state_references: Tensor, times: Tensor) -> Tensor:
    """Blend image and state box geometry while retaining image evidence at t=1."""
    if image_references.shape != state_references.shape or image_references.ndim != 3:
        raise ValueError("references must have matching [B, N, 4] shapes")
    if image_references.shape[-1] != 4:
        raise ValueError("references must end in cxcywh")
    gate = state_reference_gate(times)
    return (1.0 - gate[:, None, None]) * image_references + gate[:, None, None] * state_references
