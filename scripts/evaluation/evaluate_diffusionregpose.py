#!/usr/bin/env python3
"""Evaluate the native DiffusionRegPose checkpoint on CrowdPose."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data.datasets import PoseDataset
from src.data.io import read_image
from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import (
    crowdpose_result_records,
    evaluate_crowdpose_results,
    postprocess_crowdpose,
    resize_crowdpose_eval_image,
    write_crowdpose_results,
)
from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
from src.models.pipelines.pose_estimators.diffusionregpose import DiffusionRegPosePipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Resolved Hydra config used to construct a non-default model.",
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampling-timesteps", type=int, default=1)
    parser.add_argument("--shortest-side", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    return parser.parse_args()


def extract_model_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Extract native model weights from external or Lightning checkpoints."""
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_state = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
        if model_state:
            return model_state
    raise KeyError("checkpoint contains neither an external 'model' nor Lightning 'state_dict' model weights")


def build_evaluation_model(
    pipeline_config: DiffusionRegPosePipelineConfig,
    *,
    sampling_timesteps: int,
) -> DiffusionRegPoseModel:
    """Build the configured architecture without loading its training initializer."""
    evaluation_config = pipeline_config.model_copy(update={
        "backbone_pretrained": False,
        "sampling_timesteps": sampling_timesteps,
    })
    return evaluation_config.build(
        max_epochs=1,
        load_pretrained_checkpoint=False,
    ).model


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    # Keep the external evaluator's cuDNN setting; it only disables TF32 for
    # matrix multiplications explicitly.
    torch.backends.cudnn.allow_tf32 = True
    args = parse_args()
    # The original evaluation seeds PyTorch before model construction and
    # consumes the global CUDA generator for each diffusion proposal tensor.
    torch.manual_seed(args.seed)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.annotations.exists():
        raise FileNotFoundError(args.annotations)
    if not args.image_root.exists():
        raise FileNotFoundError(args.image_root)
    if args.config is not None and not args.config.exists():
        raise FileNotFoundError(args.config)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.config is None:
        model = DiffusionRegPoseModel(
            backbone_pretrained=False,
            sampling_timesteps=args.sampling_timesteps,
        )
    else:
        resolved_config = OmegaConf.load(args.config)
        pipeline_config = DiffusionRegPosePipelineConfig.from_hydra_config(
            resolved_config.pipeline
        )
        # The checkpoint below supplies the learned model. Do not load the
        # training-time initializer as an intermediate state. Keeping the
        # selected-layer fields allows build() to construct the separate
        # expansion pose head required by that architecture.
        model = build_evaluation_model(
            pipeline_config,
            sampling_timesteps=args.sampling_timesteps,
        )
    state_dict = extract_model_state_dict(checkpoint)
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped_keys = set(state_dict) - set(compatible_state)
    incompatible = model.load_state_dict(compatible_state, strict=False)
    parameter_names = set(dict(model.named_parameters()))
    missing_parameters = [
        key for key in incompatible.missing_keys if key in parameter_names
    ]
    missing_buffers = [
        key for key in incompatible.missing_keys if key not in parameter_names
    ]
    if missing_parameters or incompatible.unexpected_keys or skipped_keys:
        warnings.warn(
            f"Loaded {args.checkpoint} with {len(missing_parameters)} missing "
            f"parameters, {len(incompatible.unexpected_keys)} unexpected, and "
            f"{len(skipped_keys)} shape-incompatible keys.",
            stacklevel=2,
        )
    if missing_buffers:
        warnings.warn(
            f"Checkpoint omits {len(missing_buffers)} deterministic diffusion "
            "schedule buffers; using the model-generated values.",
            stacklevel=2,
        )
    model.to(args.device).eval()

    dataset = PoseDataset.from_path(args.annotations, img_root=args.image_root, num_keypoints=14)
    records = []
    evaluated_image_ids = []
    ordered_samples = sorted(dataset.samples, key=lambda sample: sample["image_id"])
    samples = ordered_samples if args.limit is None else ordered_samples[: args.limit]
    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            if sample["img_hw"] is None:
                raise ValueError(f"annotation is missing image dimensions: {sample['image_id']}")
            image = resize_crowdpose_eval_image(
                read_image(sample["img_path"]).float().div(255.0),
                shortest_side=args.shortest_side,
                max_size=args.max_size,
            )
            image = image.to(args.device)
            output = model.sample_keypoints(image[None])
            target_size = torch.tensor([sample["img_hw"]], dtype=torch.float32, device=args.device)
            nms_target_size = torch.tensor([image.shape[-2:]], dtype=torch.float32, device=args.device)
            results = postprocess_crowdpose(output, target_size, nms_target_sizes=nms_target_size)
            records.extend(crowdpose_result_records(sample["image_id"], results))
            evaluated_image_ids.append(sample["image_id"])
            print(f"[{index}/{len(samples)}] image_id={sample['image_id']}")

    write_crowdpose_results(records, args.output)
    print(f"Wrote {len(records)} predictions to {args.output}")
    if args.metrics is not None:
        metrics = evaluate_crowdpose_results(
            args.output, args.annotations, image_ids=evaluated_image_ids
        )
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2))
        print(metrics)


if __name__ == "__main__":
    main()
