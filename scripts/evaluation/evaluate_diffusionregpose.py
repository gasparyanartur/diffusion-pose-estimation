#!/usr/bin/env python3
"""Evaluate the native DiffusionRegPose checkpoint on CrowdPose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampling-timesteps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    return parser.parse_args()


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

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DiffusionRegPoseModel(
        backbone_pretrained=False,
        sampling_timesteps=args.sampling_timesteps,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device).eval()

    dataset = PoseDataset.from_path(args.annotations, img_root=args.image_root, num_keypoints=14)
    records = []
    ordered_samples = sorted(dataset.samples, key=lambda sample: sample["image_id"])
    samples = ordered_samples if args.limit is None else ordered_samples[: args.limit]
    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            if sample["img_hw"] is None:
                raise ValueError(f"annotation is missing image dimensions: {sample['image_id']}")
            image = resize_crowdpose_eval_image(read_image(sample["img_path"]).float().div(255.0))
            image = image.to(args.device)
            output = model.sample_keypoints(image[None])
            target_size = torch.tensor([sample["img_hw"]], dtype=torch.float32, device=args.device)
            nms_target_size = torch.tensor([image.shape[-2:]], dtype=torch.float32, device=args.device)
            results = postprocess_crowdpose(output, target_size, nms_target_sizes=nms_target_size)
            records.extend(crowdpose_result_records(sample["image_id"], results))
            print(f"[{index}/{len(samples)}] image_id={sample['image_id']}")

    write_crowdpose_results(records, args.output)
    print(f"Wrote {len(records)} predictions to {args.output}")
    if args.metrics is not None:
        metrics = evaluate_crowdpose_results(args.output, args.annotations)
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2))
        print(metrics)


if __name__ == "__main__":
    main()
