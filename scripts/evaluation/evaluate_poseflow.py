#!/usr/bin/env python3
"""Evaluate a trained PoseFlow checkpoint on CrowdPose."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data.datasets import PoseDataset
from src.data.io import read_image
from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import (
    crowdpose_result_records,
    evaluate_crowdpose_results,
    postprocess_crowdpose,
    postprocess_crowdpose_step_ensemble,
    resize_crowdpose_eval_image,
    write_crowdpose_results,
)
from src.models.pipelines.pose_estimators.poseflow import (
    PoseFlowPipelineConfig,
    PoseFlowV3Model,
    PoseFlowV5Model,
    PoseFlowV9Model,
    PoseFlowV12Model,
)
from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPoseMatcher
from src.eval.poseflow_diagnostics import (
    FEATURE_PATHS,
    ablate_poseflow_path,
    capture_v9_attention,
    compare_feature_predictions,
    fixed_final_pairs,
    selected_stage_outputs,
    stage_pose_oks,
    stage_quality_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--flow-sampling-steps",
        type=int,
        default=None,
        help="Override the checkpoint configuration for this evaluation only.",
    )
    parser.add_argument("--flow-renewal", action="store_true")
    parser.add_argument("--flow-renewal-score-threshold", type=float, default=None)
    parser.add_argument("--flow-ensemble", action="store_true")
    parser.add_argument(
        "--flow-seed",
        type=int,
        default=None,
        help="Optional per-image flow-source seed; use one common value for step comparisons.",
    )
    parser.add_argument(
        "--step-probe-dir",
        type=Path,
        default=None,
        help="Write V5/V9 per-step CrowdPose metrics and slot trajectories here.",
    )
    parser.add_argument(
        "--stage-probe-dir",
        type=Path,
        default=None,
        help="Write V5/V9 hypothesis-selection and refiner quality by stage here.",
    )
    parser.add_argument(
        "--feature-probe-dir",
        type=Path,
        default=None,
        help="Write fixed-source feature-path perturbation measurements here.",
    )
    parser.add_argument(
        "--attention-probe-dir",
        type=Path,
        default=None,
        help="Write V9 bias, unbiased logit scale, and nonedge attention mass.",
    )
    return parser.parse_args()


def load_poseflow_model(checkpoint_path: Path, config_path: Path):
    resolved_config = OmegaConf.load(config_path)
    pipeline_config = PoseFlowPipelineConfig.from_hydra_config(resolved_config.pipeline)
    evaluation_config = pipeline_config.model_copy(update={
        "backbone_pretrained": False,
        "pretrained_checkpoint": None,
    })
    model = evaluation_config._build_poseflow_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state:
        raise ValueError(f"checkpoint has no PoseFlow model state: {checkpoint_path}")
    model.load_state_dict(model_state, strict=True)
    return model, pipeline_config


def apply_flow_sampling_steps(model, sampling_steps: int | None) -> int:
    """Override sampler calls: V12/V13 latent DDIM, or earlier Euler flow."""
    if sampling_steps is None:
        return model.flow_sampling_steps
    if sampling_steps < 1:
        raise ValueError("flow sampling steps must be positive")
    model.flow_sampling_steps = sampling_steps
    return sampling_steps


def apply_flow_sampler_overrides(model, args: argparse.Namespace) -> None:
    if isinstance(model, PoseFlowV12Model) and (
        args.flow_renewal or args.flow_renewal_score_threshold is not None or args.flow_ensemble
    ):
        raise ValueError(f"{type(model).__name__} uses fixed-group latent DDIM without renewal or ensemble")
    if isinstance(model, PoseFlowV3Model) and (
        args.flow_renewal or args.flow_renewal_score_threshold is not None
    ):
        raise ValueError("PoseFlow V3 has fixed 50-plus-fresh refill and no score-renewal override")
    if args.flow_renewal:
        model.flow_renewal = True
    if args.flow_renewal_score_threshold is not None:
        if not 0.0 <= args.flow_renewal_score_threshold <= 1.0:
            raise ValueError("flow renewal score threshold must be in [0, 1]")
        model.flow_renewal_score_threshold = args.flow_renewal_score_threshold
    if args.flow_ensemble:
        model.flow_ensemble = True


def seed_flow_sampling(*, flow_seed: int | None, image_id: int) -> None:
    """Reset flow sampling RNG per image so Euler-step runs share their start."""
    if flow_seed is not None:
        torch.manual_seed(flow_seed + image_id)


def summarize_step_trajectory(steps: list[dict[str, torch.Tensor]]) -> list[dict]:
    """Summarize one image's selected slots across a V5/V9 rollout."""
    summaries = []
    previous = None
    for step_index, step in enumerate(steps):
        if step["selected_slot_ids"].shape[0] != 1:
            raise ValueError("step probe expects one image at a time")
        selected_ids = step["selected_slot_ids"][0].tolist()
        scores = step["pred_logits"][0].sigmoid().amax(dim=-1)
        current = {
            "step": step_index,
            "time": float(step["time"][0].item()),
            "selected_slot_ids": selected_ids,
            "selected_from_fresh": int(step["selected_from_fresh"].sum().item()),
            "selected_from_carried": int((~step["selected_from_fresh"]).sum().item()),
            "mean_score": float(scores.mean().item()),
            "mean_state_source_l1": float(
                (step["selected_state"] - step["selected_source"]).abs().mean().item()
            ),
            "paired_count": 0,
            "paired_score_change": None,
            "paired_box_l1": None,
            "paired_pose_l1": None,
        }
        if previous is not None:
            previous_indices = {
                slot_id: index for index, slot_id in enumerate(previous["selected_slot_ids"][0].tolist())
            }
            paired = [
                (index, previous_indices[slot_id])
                for index, slot_id in enumerate(selected_ids)
                if slot_id in previous_indices
            ]
            current["paired_count"] = len(paired)
            if paired:
                now, before = zip(*paired)
                previous_scores = previous["pred_logits"][0].sigmoid().amax(dim=-1)
                current["paired_score_change"] = float(
                    (scores[list(now)] - previous_scores[list(before)]).mean().item()
                )
                current["paired_box_l1"] = float(
                    (step["pred_boxes"][0, list(now)] - previous["pred_boxes"][0, list(before)])
                    .abs().mean().item()
                )
                joint_count = step["pred_keypoints"].shape[-1] // 3
                current["paired_pose_l1"] = float(
                    (step["pred_keypoints"][0, list(now), : 2 * joint_count]
                     - previous["pred_keypoints"][0, list(before), : 2 * joint_count])
                    .abs().mean().item()
                )
        summaries.append(current)
        previous = step
    return summaries


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.annotations.is_file() or not args.image_root.is_dir():
        raise FileNotFoundError("CrowdPose annotations or image root is missing")

    model, pipeline_config = load_poseflow_model(args.checkpoint, args.config)
    sampling_steps = apply_flow_sampling_steps(model, args.flow_sampling_steps)
    apply_flow_sampler_overrides(model, args)
    if any(directory is not None for directory in (
        args.step_probe_dir, args.stage_probe_dir, args.feature_probe_dir,
    )):
        if type(model) not in (PoseFlowV5Model, PoseFlowV9Model):
            raise ValueError("diagnostic probes currently support V5 and V9 checkpoints")
        if args.step_probe_dir is not None and model.flow_ensemble:
            raise ValueError("step probe requires ensemble disabled for final-step AP parity")
    if args.step_probe_dir is not None and args.step_probe_dir == args.stage_probe_dir:
        raise ValueError("step and stage probe directories must differ")
    if args.feature_probe_dir is not None and args.flow_seed is None:
        raise ValueError("feature probe requires --flow-seed to pair source draws")
    if args.attention_probe_dir is not None and type(model) is not PoseFlowV9Model:
        raise ValueError("attention probe requires a V9 checkpoint")
    probe_directories = [directory for directory in (
        args.step_probe_dir, args.stage_probe_dir,
        args.feature_probe_dir, args.attention_probe_dir,
    ) if directory is not None]
    if len(set(probe_directories)) != len(probe_directories):
        raise ValueError("diagnostic probe directories must be distinct")
    model.to(args.device).eval()
    data_config = OmegaConf.load(args.config).data
    dataset = PoseDataset.from_path(
        args.annotations,
        img_root=args.image_root,
        num_keypoints=pipeline_config.num_body_points,
    )
    records = []
    image_ids = []
    renewed_slots_by_step = torch.zeros(max(sampling_steps - 1, 0), dtype=torch.long)
    candidates_before_nms = 0
    candidates_after_nms = 0
    probe_records = [[] for _ in range(sampling_steps)] if args.step_probe_dir is not None else None
    probe_trajectories: list[dict] = []
    probe_counts_before = [0] * sampling_steps
    probe_counts_after = [0] * sampling_steps
    stage_records: dict[str, list[dict]] = {}
    stage_counts: dict[str, dict[str, float]] = {}
    stage_by_image: list[dict] = []
    feature_by_image: list[dict] = []
    attention_by_image: list[dict] = []
    stage_sigmas = (
        DiffusionRegPoseMatcher(num_body_points=pipeline_config.num_body_points).sigmas
        if args.stage_probe_dir is not None else None
    )
    feature_sigmas = None
    if args.feature_probe_dir is not None:
        feature_sigmas = (
            stage_sigmas if stage_sigmas is not None else
            DiffusionRegPoseMatcher(num_body_points=pipeline_config.num_body_points).sigmas
        )
    samples = sorted(dataset.samples, key=lambda sample: sample["image_id"])
    if args.limit is not None:
        samples = samples[:args.limit]
    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            image = resize_crowdpose_eval_image(
                read_image(sample["img_path"]).float().div(255.0),
                shortest_side=int(data_config.eval_resize_scale),
                max_size=int(data_config.eval_max_size),
            ).to(args.device)
            seed_flow_sampling(flow_seed=args.flow_seed, image_id=sample["image_id"])
            attention_context = (
                capture_v9_attention(model)
                if args.attention_probe_dir is not None else nullcontext(None)
            )
            with attention_context as attention_rows:
                output = model.sample_keypoints(
                    image[None], collect_step_diagnostics=True,
                ) if probe_records is not None else model.sample_keypoints(image[None])
            if attention_rows is not None:
                expected_rows = sampling_steps * len(model.transformer.decoder.layers)
                if len(attention_rows) != expected_rows:
                    raise ValueError("V9 attention probe missed a refiner call")
                attention_by_image.append({
                    "image_id": sample["image_id"], "layers_by_step": attention_rows,
                })
            renewed_slots = output.get("flow_renewed_slots")
            if isinstance(renewed_slots, torch.Tensor):
                renewed_slots_by_step += renewed_slots.detach().cpu().sum(dim=1)
            original_size = torch.tensor([sample["img_hw"]], dtype=torch.float32, device=args.device)
            nms_size = torch.tensor([image.shape[-2:]], dtype=torch.float32, device=args.device)
            if probe_records is not None:
                steps = output["flow_step_diagnostics"]
                if not isinstance(steps, list) or len(steps) != sampling_steps:
                    raise ValueError("sampler did not return every step diagnostic")
                trajectory = summarize_step_trajectory(steps)
                for step_index, step in enumerate(steps):
                    step_result = postprocess_crowdpose(
                        step, original_size, nms_target_sizes=nms_size,
                    )
                    probe_counts_before[step_index] += step["pred_logits"].shape[1]
                    probe_counts_after[step_index] += step_result[0]["scores"].numel()
                    probe_records[step_index].extend(
                        crowdpose_result_records(sample["image_id"], step_result)
                    )
                    trajectory[step_index]["candidates_before_nms"] = step["pred_logits"].shape[1]
                    trajectory[step_index]["candidates_after_nms"] = step_result[0]["scores"].numel()
                probe_trajectories.append({"image_id": sample["image_id"], "steps": trajectory})
            if stage_sigmas is not None:
                stages = selected_stage_outputs(output)
                fixed_pairs = fixed_final_pairs(stage_pose_oks(stages[-1][1], sample, stage_sigmas))
                image_stages = {}
                for stage_name, stage_output in stages:
                    oks = stage_pose_oks(stage_output, sample, stage_sigmas)
                    counts = stage_quality_counts(
                        oks,
                        fixed_pairs=None if stage_name == "hypothesis_all" else fixed_pairs,
                    )
                    image_stages[stage_name] = counts
                    totals = stage_counts.setdefault(stage_name, {})
                    for name, value in counts.items():
                        if value is not None:
                            totals[name] = totals.get(name, 0.0) + value
                    stage_result = postprocess_crowdpose(
                        stage_output, original_size, nms_target_sizes=nms_size,
                    )
                    stage_records.setdefault(stage_name, []).extend(
                        crowdpose_result_records(sample["image_id"], stage_result)
                    )
                    if stage_name == stages[-1][0]:
                        initial_scores_output = {
                            **stage_output,
                            "pred_logits": stages[1][1]["pred_logits"],
                        }
                        for variant_name, variant_output, use_pose_nms in (
                            ("final_hypothesis_scores", initial_scores_output, True),
                            ("final_no_pose_nms", stage_output, False),
                            ("final_hypothesis_scores_no_pose_nms", initial_scores_output, False),
                        ):
                            variant_result = postprocess_crowdpose(
                                variant_output, original_size,
                                nms_target_sizes=nms_size, pose_nms=use_pose_nms,
                            )
                            stage_records.setdefault(variant_name, []).extend(
                                crowdpose_result_records(sample["image_id"], variant_result)
                            )
                        serialized = stage_result[0]["keypoints"].reshape(
                            -1, pipeline_config.num_body_points, 3,
                        )[..., :2]
                        normalized = serialized / serialized.new_tensor(
                            [sample["img_hw"][1], sample["img_hw"][0]]
                        )
                        post_nms_output = {"pred_keypoints": normalized.flatten(1)[None]}
                        post_nms_counts = stage_quality_counts(
                            stage_pose_oks(post_nms_output, sample, stage_sigmas)
                        )
                        image_stages["final_post_nms"] = post_nms_counts
                        post_totals = stage_counts.setdefault("final_post_nms", {})
                        for name, value in post_nms_counts.items():
                            if value is not None:
                                post_totals[name] = post_totals.get(name, 0.0) + value
                stage_by_image.append({"image_id": sample["image_id"], "stages": image_stages})
            if feature_sigmas is not None:
                comparisons = {}
                for path in FEATURE_PATHS:
                    seed_flow_sampling(flow_seed=args.flow_seed, image_id=sample["image_id"])
                    with ablate_poseflow_path(model, path):
                        changed_output = model.sample_keypoints(image[None])
                    comparisons[path] = compare_feature_predictions(
                        output, changed_output, sample, feature_sigmas,
                    )
                feature_by_image.append({"image_id": sample["image_id"], "paths": comparisons})
            if model.flow_ensemble:
                step_predictions = output.get("flow_step_predictions")
                if not isinstance(step_predictions, list):
                    raise ValueError("ensemble sampling did not return per-step predictions")
                results = postprocess_crowdpose_step_ensemble(
                    step_predictions,
                    original_size,
                    nms_target_sizes=nms_size,
                )
                candidates_before_nms += sum(
                    prediction["pred_logits"].shape[1] for prediction in step_predictions
                )
            else:
                results = (
                    model.postprocess(output, original_size, nms_target_sizes=nms_size)
                    if isinstance(model, PoseFlowV12Model)
                    else postprocess_crowdpose(output, original_size, nms_target_sizes=nms_size)
                )
            candidates_after_nms += sum(result["scores"].numel() for result in results)
            records.extend(crowdpose_result_records(sample["image_id"], results))
            image_ids.append(sample["image_id"])
            print(f"[{index}/{len(samples)}] image_id={sample['image_id']}")

    write_crowdpose_results(records, args.output)
    metrics = evaluate_crowdpose_results(args.output, args.annotations, image_ids=image_ids)
    if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
        raise ValueError("CrowdPose evaluation returned a non-finite metric")
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(args.checkpoint),
        "sampler": "latent DDIM eta0" if isinstance(model, PoseFlowV12Model) else "Euler flow",
        "flow_sampling_steps": sampling_steps,
        "flow_seed": args.flow_seed,
        "renewal": model.flow_renewal,
        "renewal_score_threshold": model.flow_renewal_score_threshold,
        "ensemble": model.flow_ensemble,
        "ensemble_candidates_before_nms": candidates_before_nms if model.flow_ensemble else None,
        "ensemble_candidates_after_nms": candidates_after_nms if model.flow_ensemble else None,
        "renewed_slots_by_step": renewed_slots_by_step.tolist(),
        "metrics": metrics,
    }
    args.metrics.write_text(json.dumps(result, indent=2))
    if probe_records is not None:
        args.step_probe_dir.mkdir(parents=True, exist_ok=True)
        step_metrics = []
        for step_index, step_records in enumerate(probe_records):
            record_path = args.step_probe_dir / f"step_{step_index:02d}_predictions.json"
            write_crowdpose_results(step_records, record_path)
            step_metrics.append(evaluate_crowdpose_results(
                record_path, args.annotations, image_ids=image_ids,
            ))
        (args.step_probe_dir / "trajectories.json").write_text(json.dumps(probe_trajectories))
        (args.step_probe_dir / "summary.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "annotations": str(args.annotations),
            "image_count": len(image_ids),
            "flow_seed": args.flow_seed,
            "flow_sampling_steps": sampling_steps,
            "eval_resize_scale": int(data_config.eval_resize_scale),
            "eval_max_size": int(data_config.eval_max_size),
            "evaluator": "crowdposetools COCOeval via evaluate_crowdpose_results",
            "steps": [
                {
                    "step": step_index,
                    "time": step_index / sampling_steps,
                    "candidates_before_nms": probe_counts_before[step_index],
                    "candidates_after_nms": probe_counts_after[step_index],
                    "metrics": metrics,
                }
                for step_index, metrics in enumerate(step_metrics)
            ],
        }, indent=2))
        if step_metrics[-1] != metrics:
            raise RuntimeError("step probe final AP disagrees with ordinary evaluation")
    if stage_sigmas is not None:
        args.stage_probe_dir.mkdir(parents=True, exist_ok=True)
        stage_results = {}
        for stage_name, stage_predictions in stage_records.items():
            record_path = args.stage_probe_dir / f"{stage_name}_predictions.json"
            write_crowdpose_results(stage_predictions, record_path)
            stage_results[stage_name] = {
                "metrics": evaluate_crowdpose_results(
                    record_path, args.annotations, image_ids=image_ids,
                ),
                "quality_counts": stage_counts.get(stage_name),
            }
        stage_results["final_post_nms"] = {"quality_counts": stage_counts["final_post_nms"]}
        (args.stage_probe_dir / "by_image.json").write_text(json.dumps(stage_by_image))
        (args.stage_probe_dir / "summary.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "annotations": str(args.annotations),
            "image_count": len(image_ids),
            "flow_seed": args.flow_seed,
            "flow_sampling_steps": sampling_steps,
            "eval_resize_scale": int(data_config.eval_resize_scale),
            "eval_max_size": int(data_config.eval_max_size),
            "quality_metric": "bbox-area OKS on labeled joints; oracle coverage at 0.50 and 0.75",
            "paired_metric": "one fixed one-to-one final-refiner OKS assignment reused across selected stages",
            "stages": stage_results,
        }, indent=2))
    if feature_sigmas is not None:
        args.feature_probe_dir.mkdir(parents=True, exist_ok=True)
        (args.feature_probe_dir / "by_image.json").write_text(json.dumps(feature_by_image))
        path_summaries = {}
        for path in FEATURE_PATHS:
            rows = [image["paths"][path] for image in feature_by_image]
            totals = {name: sum(row[name] for row in rows) for name in (
                "target_count", "selected_count", "shared_selected_count",
                "initial_oracle_oks_change", "final_oracle_oks_change",
                "initial_covered_oks_50_change", "final_covered_oks_50_change",
            )}
            path_summaries[path] = {
                **totals,
                "mean_initial_box_l1": sum(row["initial_box_l1"] for row in rows) / len(rows),
                "mean_initial_pose_l1": sum(row["initial_pose_l1"] for row in rows) / len(rows),
                "mean_initial_score_abs": sum(row["initial_score_abs"] for row in rows) / len(rows),
                "mean_final_shared_box_l1": sum(
                    row["final_shared_box_l1"] for row in rows
                    if row["final_shared_box_l1"] is not None
                ) / max(sum(row["final_shared_box_l1"] is not None for row in rows), 1),
                "mean_final_shared_pose_l1": sum(
                    row["final_shared_pose_l1"] for row in rows
                    if row["final_shared_pose_l1"] is not None
                ) / max(sum(row["final_shared_pose_l1"] is not None for row in rows), 1),
            }
        (args.feature_probe_dir / "summary.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "annotations": str(args.annotations),
            "image_count": len(image_ids),
            "flow_seed": args.flow_seed,
            "flow_sampling_steps": sampling_steps,
            "eval_resize_scale": int(data_config.eval_resize_scale),
            "eval_max_size": int(data_config.eval_max_size),
            "interpretation": "inference-only path perturbations; prediction sensitivity, not a retrained ablation",
            "paths": path_summaries,
        }, indent=2))
    if args.attention_probe_dir is not None:
        args.attention_probe_dir.mkdir(parents=True, exist_ok=True)
        (args.attention_probe_dir / "by_image.json").write_text(json.dumps(attention_by_image))
        layer_summaries = []
        for step_index in range(sampling_steps):
            for layer_index in range(len(model.transformer.decoder.layers)):
                rows = [
                    row for image_row in attention_by_image
                    for row in image_row["layers_by_step"]
                    if row["step"] == step_index and row["layer"] == layer_index
                ]
                layer_summaries.append({
                    "step": step_index,
                    "layer": layer_index,
                    **{
                        name: sum(row[name] for row in rows) / len(rows)
                        for name in ("off_edge_bias", "raw_logit_std", "nonedge_attention_mass")
                    },
                })
        (args.attention_probe_dir / "summary.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "annotations": str(args.annotations),
            "image_count": len(image_ids),
            "flow_seed": args.flow_seed,
            "flow_sampling_steps": sampling_steps,
            "eval_resize_scale": int(data_config.eval_resize_scale),
            "eval_max_size": int(data_config.eval_max_size),
            "measure": "within-person clean joint-query QK logits; pre-dropout softmax",
            "layers_by_step": layer_summaries,
        }, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
