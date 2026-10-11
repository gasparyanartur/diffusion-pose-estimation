"""Bounded validation summaries for future PoseFlow training runs."""

from __future__ import annotations

import torch
from torch import Tensor

from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import (
    CROWDPOSE_POSE_NMS_SCORE_THRESHOLD,
)
from .poseflow_diagnostics import (
    compare_feature_predictions, fixed_final_pairs, selected_stage_outputs, stage_pose_oks,
)


def _image_targets(batch: dict[str, Tensor], index: int) -> dict:
    valid = batch["person_mask"][index].bool()
    height, width = batch["img_hw"][index].tolist()
    points = (batch["gt_keypoints"][index, valid] + 1.0) / 2.0
    return {
        "keypoints": points * points.new_tensor([width, height]),
        "visibility": batch["gt_labeled"][index, valid],
        "gt_boxes_pixel": batch["gt_boxes_pixel"][index, valid],
        "img_hw": (int(height), int(width)),
    }


def _image_stage(stage: dict[str, Tensor], index: int) -> dict[str, Tensor]:
    return {name: value[index : index + 1] for name, value in stage.items()}


def _image_output(output: dict, index: int) -> dict:
    fields = ("pred_logits", "pred_boxes", "pred_keypoints")
    return {
        **{name: output[name][index : index + 1] for name in fields},
        "selected_person_indices": output["selected_person_indices"][index : index + 1],
        "aux_outputs": [
            {name: stage[name][index : index + 1] for name in fields if name in stage}
            for stage in output["aux_outputs"]
        ],
    }


@torch.no_grad()
def summarize_feature_intervention(
    batch: dict[str, Tensor], baseline: dict, changed: dict, sigmas: Tensor,
) -> dict[str, float]:
    """Compare paired validation predictions from one disabled feature path."""
    totals = {"gt_count": 0.0, "selected_count": 0.0, "shared_selected_count": 0.0,
              "initial_oracle_oks_change": 0.0, "final_oracle_oks_change": 0.0}
    if "pred_keypoints" not in baseline["aux_outputs"][0]:
        totals.pop("initial_oracle_oks_change")
        totals["initial_box_l1"] = 0.0
        totals["initial_score_abs"] = 0.0
    paired_quality: dict[str, float] = {}
    shared_pair_count = 0
    for index in range(batch["image"].shape[0]):
        original = _image_output(baseline, index)
        perturbed = _image_output(changed, index)
        target = _image_targets(batch, index)
        comparison = compare_feature_predictions(
            original, perturbed, target, sigmas,
        )
        totals["gt_count"] += float(comparison["target_count"])
        for name in totals:
            if name == "gt_count":
                continue
            totals[name] += float(comparison[name])
        original_stages = selected_stage_outputs(original)[1:]
        perturbed_stages = selected_stage_outputs(perturbed)[1:]
        final_oks = stage_pose_oks(original_stages[-1][1], target, sigmas)
        fixed_queries, fixed_targets = fixed_final_pairs(final_oks)
        changed_slots = {
            int(slot): query for query, slot in enumerate(
                perturbed["selected_person_indices"][0].tolist()
            )
        }
        shared = [
            (int(query), changed_slots[int(original["selected_person_indices"][0, query])],
             int(gt))
            for query, gt in zip(fixed_queries.tolist(), fixed_targets.tolist())
            if int(original["selected_person_indices"][0, query]) in changed_slots
        ]
        shared_pair_count += len(shared)
        if shared:
            before_queries, after_queries, target_indices = zip(*shared)
            for (name, before), (_, after) in zip(original_stages, perturbed_stages):
                if "pred_keypoints" not in before:
                    continue
                before_oks = stage_pose_oks(before, target, sigmas)
                after_oks = stage_pose_oks(after, target, sigmas)
                change = (
                    after_oks[list(after_queries), list(target_indices)]
                    - before_oks[list(before_queries), list(target_indices)]
                ).sum().item()
                paired_quality[name] = paired_quality.get(name, 0.0) + change
    gt_count = max(totals["gt_count"], 1)
    selected_count = max(totals["selected_count"], 1)
    result = {
        "gt_count": totals["gt_count"],
        "selected_count": totals["selected_count"],
        "shared_fraction": totals["shared_selected_count"] / selected_count,
        "shared_pair_count": float(shared_pair_count),
        **({"initial_oracle_oks_change_per_gt": totals["initial_oracle_oks_change"] / gt_count}
           if "initial_oracle_oks_change" in totals else {
               "initial_box_l1": totals["initial_box_l1"] / max(batch["image"].shape[0], 1),
               "initial_score_abs": totals["initial_score_abs"] / max(batch["image"].shape[0], 1),
           }),
        "final_oracle_oks_change_per_gt": totals["final_oracle_oks_change"] / gt_count,
    }
    for name, change in paired_quality.items():
        result[f"{name}/paired_oks_change"] = change / max(shared_pair_count, 1)
    return result


def _paired_box_iou(boxes: Tensor, target_boxes: Tensor, image_size: tuple[int, int]) -> Tensor:
    height, width = image_size
    center = boxes[..., :2] * boxes.new_tensor([width, height])
    half_size = boxes[..., 2:] * boxes.new_tensor([width, height]) / 2
    prediction_min, prediction_max = center - half_size, center + half_size
    target_min = target_boxes[..., :2]
    target_max = target_min + target_boxes[..., 2:]
    overlap = (torch.minimum(prediction_max, target_max)
               - torch.maximum(prediction_min, target_min)).clamp_min(0).prod(-1)
    prediction_area = (prediction_max - prediction_min).clamp_min(0).prod(-1)
    target_area = target_boxes[..., 2:].clamp_min(0).prod(-1)
    return overlap / (prediction_area + target_area - overlap).clamp_min(1e-6)


@torch.no_grad()
def summarize_validation_solution(
    batch: dict[str, Tensor],
    output: dict,
    postprocessed: list[dict[str, Tensor]],
    sigmas: Tensor,
) -> dict[str, float]:
    """Measure one validation batch without changing sampled predictions."""
    stages = selected_stage_outputs(output)
    if len(postprocessed) != batch["image"].shape[0]:
        raise ValueError("postprocessed results must match validation batch size")
    totals: dict[str, float] = {
        "image_count": float(len(postprocessed)), "gt_count": 0.0, "paired_count": 0.0,
    }
    matched_final_scores: list[Tensor] = []
    unmatched_final_scores: list[Tensor] = []

    def add(name: str, value: float) -> None:
        totals[name] = totals.get(name, 0.0) + float(value)

    for image_index, kept in enumerate(postprocessed):
        target = _image_targets(batch, image_index)
        gt_count = target["keypoints"].shape[0]
        add("gt_count", gt_count)
        stage_oks = [stage_pose_oks(_image_stage(stage, image_index), target, sigmas)
                     if "pred_keypoints" in stage else None for _, stage in stages]
        fixed_queries, fixed_targets = fixed_final_pairs(stage_oks[-1])
        add("paired_count", fixed_queries.numel())
        for (stage_name, stage), oks in zip(stages, stage_oks):
            if oks is not None:
                best = oks.amax(0) if oks.shape[0] else oks.new_zeros(gt_count)
                add(f"stage/{stage_name}/oracle_oks_sum", best.sum().item())
                add(f"stage/{stage_name}/covered_oks50_count", (best >= 0.5).sum().item())
            else:
                boxes = stage["pred_boxes"][image_index]
                iou = _paired_box_iou(boxes[:, None], target["gt_boxes_pixel"][None].to(boxes), target["img_hw"])
                best = iou.amax(0) if iou.shape[0] else iou.new_zeros(gt_count)
                add(f"stage/{stage_name}/oracle_box_iou_sum", best.sum().item())
                add(f"stage/{stage_name}/covered_box50_count", (best >= .5).sum().item())
            if stage_name != "hypothesis_all":
                if oks is not None:
                    add(f"stage/{stage_name}/paired_oks_sum",
                        oks[fixed_queries, fixed_targets].sum().item())
                if fixed_queries.numel():
                    boxes = stage["pred_boxes"][image_index, fixed_queries]
                    targets = target["gt_boxes_pixel"][fixed_targets].to(boxes)
                    add(f"stage/{stage_name}/paired_box_iou_sum",
                        _paired_box_iou(boxes, targets, target["img_hw"]).sum().item())
                scores = stage["pred_logits"][image_index].sigmoid().amax(-1)
                if fixed_queries.numel():
                    matched_scores = scores[fixed_queries]
                    add(f"stage/{stage_name}/matched_score_sum", matched_scores.sum().item())
                    rank = (scores[:, None] > matched_scores[None]).sum(0) + 1
                    add(f"stage/{stage_name}/matched_rank_sum", rank.sum().item())
                if stage_name == stages[-1][0]:
                    matched_final_scores.append(scores[fixed_queries])
                    unmatched = torch.ones_like(scores, dtype=torch.bool)
                    unmatched[fixed_queries] = False
                    unmatched_final_scores.append(scores[unmatched])
        if fixed_queries.numel():
            all_scores = stages[0][1]["pred_logits"][image_index].sigmoid().amax(-1)
            selected_ids = output["selected_person_indices"][image_index, fixed_queries]
            selected_scores = all_scores[selected_ids]
            rank = (all_scores[:, None] > selected_scores[None]).sum(0) + 1
            add("stage/hypothesis_all/selected_pair_rank_sum", rank.sum().item())
        final_oks = stage_oks[-1]
        original_size = batch.get("original_img_hw", batch["img_hw"])[image_index]
        normalized_kept = kept["keypoints"].reshape(-1, sigmas.numel(), 3)[..., :2]
        normalized_kept = normalized_kept / normalized_kept.new_tensor(
            [original_size[1], original_size[0]]
        )
        kept_oks = stage_pose_oks(
            {"pred_keypoints": normalized_kept.flatten(1)[None]}, target, sigmas,
        )
        best_raw = final_oks.amax(0) if final_oks.shape[0] else final_oks.new_zeros(gt_count)
        best_kept = kept_oks.amax(0) if kept_oks.shape[0] else kept_oks.new_zeros(gt_count)
        add("postprocess/covered_pre_count", (best_raw >= 0.5).sum().item())
        add("postprocess/covered_post_count", (best_kept >= 0.5).sum().item())
        add("postprocess/covered_lost_count",
            ((best_raw >= 0.5) & (best_kept < 0.5)).sum().item())
        add("postprocess/candidates_before_nms", final_oks.shape[0])
        add("postprocess/candidates_after_nms", kept["scores"].numel())

        steps = output.get("flow_step_diagnostics")
        if isinstance(steps, list):
            for step_index, step in enumerate(steps):
                step_oks = stage_pose_oks(
                    {"pred_keypoints": step["pred_keypoints"][image_index : image_index + 1]},
                    target, sigmas,
                )
                best_step = step_oks.amax(0) if step_oks.shape[0] else step_oks.new_zeros(gt_count)
                add(f"flow/step_{step_index}/oracle_oks_sum", best_step.sum().item())
                add(f"flow/step_{step_index}/covered_oks50_count",
                    (best_step >= 0.5).sum().item())

    gt_count = totals["gt_count"]
    paired_count = totals["paired_count"]
    totals["postprocess/score_threshold"] = CROWDPOSE_POSE_NMS_SCORE_THRESHOLD
    totals["postprocess/class_count"] = float(stages[-1][1]["pred_logits"].shape[-1])
    for label, tensors in (("matched", matched_final_scores),
                           ("unmatched", unmatched_final_scores)):
        if tensors and any(tensor.numel() for tensor in tensors):
            scores = torch.cat(tensors)
            totals[f"postprocess/{label}_score_count"] = float(scores.numel())
            for quantile, name in ((0.1, "p10"), (0.5, "median"), (0.9, "p90")):
                totals[f"postprocess/{label}_score_{name}"] = float(
                    torch.quantile(scores.float(), quantile).item()
                )
        else:
            totals[f"postprocess/{label}_score_count"] = 0.0
    for name in list(totals):
        if name.endswith(("/oracle_oks_sum", "/oracle_box_iou_sum")):
            totals[name.replace("_sum", "")] = totals.pop(name) / max(gt_count, 1)
        elif name.endswith(("/covered_oks50_count", "/covered_box50_count")):
            totals[name.replace("_count", "")] = totals[name] / max(gt_count, 1)
        elif name.endswith("/paired_oks_sum"):
            totals[name.replace("_sum", "")] = totals.pop(name) / max(paired_count, 1)
        elif name.endswith("/paired_box_iou_sum") or name.endswith("/selected_pair_rank_sum"):
            totals[name.replace("_sum", "")] = totals.pop(name) / max(paired_count, 1)
        elif name.endswith("/matched_score_sum") or name.endswith("/matched_rank_sum"):
            totals[name.replace("_sum", "")] = totals.pop(name) / max(paired_count, 1)
    for name in ("covered_pre", "covered_post", "covered_lost"):
        totals[f"postprocess/{name}_fraction"] = (
            totals[f"postprocess/{name}_count"] / max(gt_count, 1)
        )
    totals["postprocess/candidate_keep_fraction"] = (
        totals["postprocess/candidates_after_nms"]
        / max(totals["postprocess/candidates_before_nms"], 1)
    )
    steps = output.get("flow_step_diagnostics")
    if isinstance(steps, list):
        for step_index, step in enumerate(steps):
            prefix = f"flow/step_{step_index}"
            fresh = step["selected_from_fresh"]
            totals[f"{prefix}/fresh_fraction"] = float(fresh.float().mean().item())
            totals[f"{prefix}/selected_count"] = float(fresh.numel())
            totals[f"{prefix}/state_source_l1"] = float(
                (step["selected_state"] - step["selected_source"]).abs().mean().item()
            )
            if "state_update_l1" in step:
                totals[f"{prefix}/state_update_l1"] = float(
                    step["state_update_l1"].mean().item()
                )
            if step_index:
                prior_ids = steps[step_index - 1]["selected_slot_ids"]
                retained = (step["selected_slot_ids"][..., None] == prior_ids[:, None]).any(-1)
                totals[f"{prefix}/retained_fraction"] = float(retained.float().mean().item())
    return totals
