#!/usr/bin/env python3
"""Evaluate a trained PoseFlow checkpoint on CrowdPose."""

from __future__ import annotations

import argparse
import json
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
from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig, PoseFlowV3Model


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
    """Apply a validated evaluator-only Euler-step override."""
    if sampling_steps is None:
        return model.flow_sampling_steps
    if sampling_steps < 1:
        raise ValueError("flow sampling steps must be positive")
    model.flow_sampling_steps = sampling_steps
    return sampling_steps


def apply_flow_sampler_overrides(model, args: argparse.Namespace) -> None:
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
            output = model.sample_keypoints(image[None])
            renewed_slots = output["flow_renewed_slots"]
            if isinstance(renewed_slots, torch.Tensor):
                renewed_slots_by_step += renewed_slots.detach().cpu().sum(dim=1)
            original_size = torch.tensor([sample["img_hw"]], dtype=torch.float32, device=args.device)
            nms_size = torch.tensor([image.shape[-2:]], dtype=torch.float32, device=args.device)
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
                results = postprocess_crowdpose(output, original_size, nms_target_sizes=nms_size)
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
        "flow_sampling_steps": sampling_steps,
        "renewal": model.flow_renewal,
        "renewal_score_threshold": model.flow_renewal_score_threshold,
        "ensemble": model.flow_ensemble,
        "ensemble_candidates_before_nms": candidates_before_nms if model.flow_ensemble else None,
        "ensemble_candidates_after_nms": candidates_after_nms if model.flow_ensemble else None,
        "renewed_slots_by_step": renewed_slots_by_step.tolist(),
        "metrics": metrics,
    }
    args.metrics.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
