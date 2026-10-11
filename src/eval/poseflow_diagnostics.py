"""Detached, fixed-checkpoint diagnostics for PoseFlow evaluation."""

from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import Tensor
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


FEATURE_PATHS = (
    "hypothesis_coordinate",
    "hypothesis_time",
    "hypothesis_image_update",
    "refiner_image_cross_attention",
)


@torch.no_grad()
def summarize_v10_latent_update(before: Tensor, after: Tensor,
                               previous_coordinates: Tensor, logits: Tensor) -> dict[str, float]:
    before, after = before.detach().float(), after.detach().float()
    coordinates = logits.detach().float().sigmoid()
    update_rms = (after - before).square().mean().sqrt()
    input_rms = before.square().mean().sqrt()
    centered = after - after.mean(dim=1, keepdim=True)
    corners = coordinates[..., :4]
    sizes = (corners[..., 2:] - corners[..., :2]).abs()
    return {
        "person_count": float(before.shape[0] * before.shape[1]),
        "input_rms": float(input_rms),
        "update_rms": float(update_rms),
        "update_input_rms_ratio": float(update_rms / input_rms.clamp_min(1e-8)),
        "person_centered_rms": float(centered.square().mean().sqrt()),
        "geometry_logit_saturation_fraction": float((logits.detach().abs() > 5).float().mean()),
        "box_point_movement_l1": float((coordinates[..., :6] - previous_coordinates[..., :6]).abs().mean()),
        "pose_movement_l1": float((coordinates[..., 6:] - previous_coordinates[..., 6:]).abs().mean()),
        "tiny_box_fraction": float((sizes < 1e-3).any(dim=-1).float().mean()),
    }


@contextmanager
def capture_v10_latents(model):
    """Observe clean-person updates during evaluation; restore hooks on exit."""
    if model.training:
        raise ValueError("V10 latent capture requires evaluation mode without DN queries")
    records = []
    pending = {}
    geometry = {}
    hooks = []

    def capture_input(layer_index):
        def capture(module, args):
            pending.update(layer=layer_index, before=args[0].detach().transpose(0, 1))
        return capture

    def capture_output(module, args, output):
        pending["after"] = output.detach().transpose(0, 1)

    def capture_selection(module, args, scores):
        if "initial" not in geometry:
            return
        coordinates = geometry.pop("initial")
        selected = scores.detach().max(-1).values.topk(model.num_proposals, dim=1).indices
        geometry["previous"] = coordinates.gather(
            1, selected[..., None].expand(-1, -1, coordinates.shape[-1]),
        )

    def capture_geometry(module, args, logits):
        if "after" not in pending:
            geometry["initial"] = logits.detach().sigmoid()
            return
        values = summarize_v10_latent_update(
            pending["before"], pending["after"], geometry["previous"], logits,
        )
        records.append({"step": len(records) // len(model.transformer.decoder.layers),
                        "layer": pending["layer"], **values})
        geometry["previous"] = logits.detach().sigmoid()
        pending.clear()

    for index, layer in enumerate(model.transformer.decoder.layers):
        hooks.append(layer.register_forward_pre_hook(capture_input(index)))
        hooks.append(layer.register_forward_hook(capture_output))
    hooks.append(model.initial_coordinate_head.register_forward_hook(capture_geometry))
    hooks.append(model.initial_class_head.register_forward_hook(capture_selection))
    try:
        yield records
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def capture_v9_attention(model):
    decoder = model.transformer.decoder
    group_count = decoder.num_group
    group_size = decoder.num_body_points + 1
    records: list[dict[str, float | int]] = []
    hooks = []

    def make_hook(layer_index: int):
        def capture(module, inputs, kwargs):
            query = inputs[0].detach().transpose(0, 1)
            mask = kwargs.get("attn_mask")
            batch_size, _, width = query.shape
            head_count = module.num_heads
            query_count = group_count * group_size
            if not isinstance(mask, Tensor) or mask.shape[-2:] != (query_count, query_count):
                raise ValueError(
                    "V9 attention probe requires the clean grouped attention mask; "
                    f"got {None if not isinstance(mask, Tensor) else tuple(mask.shape)}"
                )
            if mask.ndim == 2:
                mask_by_head = mask[None, None].expand(batch_size, head_count, -1, -1)
            elif mask.ndim == 3 and mask.shape[0] == batch_size * head_count:
                mask_by_head = mask.reshape(batch_size, head_count, query_count, query_count)
            else:
                raise ValueError(f"unexpected V9 attention mask shape: {tuple(mask.shape)}")
            head_width = width // head_count
            bias = module.in_proj_bias
            query_bias = None if bias is None else bias[:width]
            key_bias = None if bias is None else bias[width : 2 * width]
            with torch.no_grad():
                query_projection = F.linear(query, module.in_proj_weight[:width], query_bias)
                key_projection = F.linear(query, module.in_proj_weight[width : 2 * width], key_bias)
                query_heads = query_projection.reshape(
                    batch_size, group_count, group_size, head_count, head_width,
                ).permute(0, 1, 3, 2, 4)
                key_heads = key_projection.reshape(
                    batch_size, group_count, group_size, head_count, head_width,
                ).permute(0, 1, 3, 2, 4)
                raw_logits = torch.matmul(
                    query_heads[..., 1:, :], key_heads.transpose(-2, -1),
                ).float() / math.sqrt(head_width)
                group_masks = torch.stack([
                    mask_by_head[:, :,
                                 group * group_size + 1 : (group + 1) * group_size,
                                 group * group_size : (group + 1) * group_size]
                    for group in range(group_count)
                ], dim=1)
                off_edge_masks = torch.stack([
                    model._off_edge_pairs[
                        group * group_size + 1 : (group + 1) * group_size,
                        group * group_size : (group + 1) * group_size,
                    ]
                    for group in range(group_count)
                ])
                probabilities = (raw_logits + group_masks).softmax(-1)
                nonedge_mass = (
                    probabilities * off_edge_masks[None, :, None]
                ).sum(-1).mean()
                records.append({
                    "step": len(records) // len(decoder.layers),
                    "layer": layer_index,
                    "off_edge_bias": float(model.off_edge_attention_bias[layer_index].item()),
                    "raw_logit_std": float(raw_logits.std(dim=-1, unbiased=False).mean().item()),
                    "nonedge_attention_mass": float(nonedge_mass.item()),
                })
        return capture

    for layer_index, layer in enumerate(decoder.layers):
        hooks.append(layer.self_attn.register_forward_pre_hook(
            make_hook(layer_index), with_kwargs=True,
        ))
    try:
        yield records
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def ablate_poseflow_path(model, path: str):
    if path == "hypothesis_coordinate":
        modules = [model.hypothesis_stage.coordinate_projection]
        if hasattr(model, "joint_conditioning"):
            modules.append(model.joint_conditioning)
    elif path == "hypothesis_time":
        modules = [model.hypothesis_stage.time_projection]
    elif path == "hypothesis_image_update":
        modules = [model.hypothesis_stage.cross_attention]
    elif path == "refiner_image_cross_attention":
        modules = [layer.cross_attn for layer in model.transformer.decoder.layers]
    else:
        raise ValueError(f"unknown PoseFlow feature path: {path}")

    def zero_output(_module, _inputs, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]), *output[1:])
        return torch.zeros_like(output)

    hooks = [module.register_forward_hook(zero_output) for module in modules]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def selected_stage_outputs(
    sample_output: dict,
) -> list[tuple[str, dict[str, Tensor]]]:
    auxiliary = sample_output.get("aux_outputs")
    selected = sample_output.get("selected_person_indices")
    if not isinstance(auxiliary, list) or not auxiliary or not isinstance(selected, Tensor):
        raise ValueError("stage probe requires auxiliary predictions and selected indices")
    names = ("pred_logits", "pred_boxes", "pred_keypoints")
    initial = {name: auxiliary[0][name] for name in names if name in auxiliary[0]}
    selected_initial = {
        name: torch.gather(
            initial[name], 1,
            selected[..., None].expand(-1, -1, initial[name].shape[-1]),
        )
        for name in initial
    }
    stages = [("hypothesis_all", initial), ("hypothesis_selected", selected_initial)]
    for index, output in enumerate([*auxiliary[1:], sample_output]):
        stages.append((f"refiner_{index}", {name: output[name] for name in names}))
    return stages


def pairwise_pose_oks(
    prediction: Tensor,
    target: Tensor,
    visibility: Tensor,
    *,
    area: Tensor,
    image_size: tuple[int, int],
    sigmas: Tensor,
) -> Tensor:
    """Training-style, bbox-area OKS for normalized predicted joint XY."""
    if prediction.ndim != 3 or target.ndim != 3 or prediction.shape[1:] != target.shape[1:]:
        raise ValueError("prediction and target must be [people, joints, 2]")
    if visibility.shape != target.shape[:2] or area.shape != target.shape[:1]:
        raise ValueError("visibility and area must match target people")
    if sigmas.shape != (target.shape[1],):
        raise ValueError("sigmas must have one value per joint")
    if prediction.shape[0] == 0 or target.shape[0] == 0:
        return prediction.new_zeros((prediction.shape[0], target.shape[0]))
    height, width = image_size
    pixel_prediction = prediction * prediction.new_tensor([width, height])
    squared_distance = (pixel_prediction[:, None] - target[None]).square().sum(-1)
    denominator = 2 * area.clamp_min(1.0)[None, :, None] * (2 * sigmas.to(prediction)).square()
    joint_oks = torch.exp(-squared_distance / denominator)
    valid = (visibility > 0).to(prediction)
    return (joint_oks * valid[None]).sum(-1) / valid.sum(-1).clamp_min(1)[None]


def stage_pose_oks(
    output: dict[str, Tensor],
    sample: dict,
    sigmas: Tensor,
) -> Tensor:
    joint_count = sigmas.numel()
    predicted = output["pred_keypoints"][0, :, : 2 * joint_count].reshape(-1, joint_count, 2)
    boxes = sample["gt_boxes_pixel"].to(predicted)
    area = (boxes[:, 2] * boxes[:, 3]).clamp_min(1)
    return pairwise_pose_oks(
        predicted,
        sample["keypoints"].to(predicted),
        sample["visibility"].to(predicted),
        area=area,
        image_size=sample["img_hw"],
        sigmas=sigmas,
    )


def fixed_final_pairs(final_oks: Tensor) -> tuple[Tensor, Tensor]:
    if not final_oks.numel():
        empty = torch.empty(0, dtype=torch.long, device=final_oks.device)
        return empty, empty
    queries, targets = linear_sum_assignment(-final_oks.detach().cpu().numpy())
    return (
        torch.as_tensor(queries, dtype=torch.long, device=final_oks.device),
        torch.as_tensor(targets, dtype=torch.long, device=final_oks.device),
    )


def stage_quality_counts(
    oks: Tensor,
    *,
    fixed_pairs: tuple[Tensor, Tensor] | None = None,
) -> dict[str, float | int | None]:
    target_count = oks.shape[1]
    best = oks.amax(dim=0) if oks.shape[0] else oks.new_zeros(target_count)
    paired_oks = None
    paired_count = 0
    if fixed_pairs is not None:
        query_indices, target_indices = fixed_pairs
        paired_count = query_indices.numel()
        paired_oks = float(oks[query_indices, target_indices].sum().item())
    return {
        "target_count": target_count,
        "oracle_oks_sum": float(best.sum().item()),
        "covered_oks_50": int((best >= 0.5).sum().item()),
        "covered_oks_75": int((best >= 0.75).sum().item()),
        "paired_count": paired_count,
        "paired_oks_sum": paired_oks,
    }


def compare_feature_predictions(
    baseline: dict,
    perturbed: dict,
    sample: dict,
    sigmas: Tensor,
) -> dict[str, float | int | None]:
    baseline_stages = selected_stage_outputs(baseline)
    perturbed_stages = selected_stage_outputs(perturbed)
    base_initial = baseline_stages[0][1]
    changed_initial = perturbed_stages[0][1]
    base_final = baseline_stages[-1][1]
    changed_final = perturbed_stages[-1][1]
    base_indices = baseline["selected_person_indices"][0].tolist()
    changed_lookup = {
        slot_id: index
        for index, slot_id in enumerate(perturbed["selected_person_indices"][0].tolist())
    }
    shared = [(index, changed_lookup[slot_id]) for index, slot_id in enumerate(base_indices)
              if slot_id in changed_lookup]

    def mean_changes(first: dict[str, Tensor], second: dict[str, Tensor],
                     first_indices, second_indices) -> dict[str, float | None]:
        if not first_indices:
            return {"box_l1": None, "pose_l1": None, "score_abs": None}
        first_boxes = first["pred_boxes"][0, first_indices]
        second_boxes = second["pred_boxes"][0, second_indices]
        joint_count = sigmas.numel()
        pose_change = None
        if "pred_keypoints" in first and "pred_keypoints" in second:
            first_pose = first["pred_keypoints"][0, first_indices, : 2 * joint_count]
            second_pose = second["pred_keypoints"][0, second_indices, : 2 * joint_count]
            pose_change = float((first_pose - second_pose).abs().mean().item())
        first_scores = first["pred_logits"][0, first_indices].sigmoid().amax(dim=-1)
        second_scores = second["pred_logits"][0, second_indices].sigmoid().amax(dim=-1)
        return {
            "box_l1": float((first_boxes - second_boxes).abs().mean().item()),
            "pose_l1": pose_change,
            "score_abs": float((first_scores - second_scores).abs().mean().item()),
        }

    initial_changes = mean_changes(
        base_initial, changed_initial,
        list(range(base_initial["pred_boxes"].shape[1])),
        list(range(changed_initial["pred_boxes"].shape[1])),
    )
    final_changes = mean_changes(
        base_final, changed_final,
        [pair[0] for pair in shared], [pair[1] for pair in shared],
    )
    base_initial_quality = changed_initial_quality = None
    if "pred_keypoints" in base_initial:
        base_initial_quality = stage_quality_counts(stage_pose_oks(base_initial, sample, sigmas))
        changed_initial_quality = stage_quality_counts(stage_pose_oks(changed_initial, sample, sigmas))
    base_final_quality = stage_quality_counts(stage_pose_oks(base_final, sample, sigmas))
    changed_final_quality = stage_quality_counts(stage_pose_oks(changed_final, sample, sigmas))
    return {
        "target_count": base_final_quality["target_count"],
        "selected_count": len(base_indices),
        "shared_selected_count": len(shared),
        **{f"initial_{name}": value for name, value in initial_changes.items()},
        **{f"final_shared_{name}": value for name, value in final_changes.items()},
        "initial_oracle_oks_change": (
            (changed_initial_quality["oracle_oks_sum"] - base_initial_quality["oracle_oks_sum"])
            if base_initial_quality is not None else None
        ),
        "final_oracle_oks_change": (
            changed_final_quality["oracle_oks_sum"] - base_final_quality["oracle_oks_sum"]
        ),
        "initial_covered_oks_50_change": (
            (changed_initial_quality["covered_oks_50"] - base_initial_quality["covered_oks_50"])
            if base_initial_quality is not None else None
        ),
        "final_covered_oks_50_change": (
            changed_final_quality["covered_oks_50"] - base_final_quality["covered_oks_50"]
        ),
    }
