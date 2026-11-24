#!/usr/bin/env python3
"""Run pinned GroupPose on CrowdPose with native prediction formatting."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from src.data.io import load_gt_annotations, read_image
from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import (
    evaluate_crowdpose_results,
    resize_crowdpose_eval_image,
    write_crowdpose_results,
)

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REVISION = "127fe0e2e7b54d7a43420015a2e90e456a8ea1fb"
DEFAULT_OPS = ROOT / "external/DiffusionRegPose/models/diffusionregpose/ops"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_images(annotations: Path, manifest: Path | None) -> list[dict]:
    images = json.loads(annotations.read_text())["images"]
    indexed = {image["id"]: image for image in images}
    if not indexed or len(indexed) != len(images):
        raise ValueError("annotations must contain a nonempty set of unique image IDs")
    if manifest is None:
        return [indexed[image_id] for image_id in sorted(indexed)]
    subset = json.loads(manifest.read_text())
    if subset["annotation_sha256"] != sha256(annotations):
        raise ValueError("subset annotation hash does not match requested annotations")
    image_ids = subset["image_ids"]
    if not image_ids or len(set(image_ids)) != len(image_ids) or not set(image_ids) <= indexed.keys():
        raise ValueError("subset must contain nonempty, unique, known image IDs")
    return [indexed[image_id] for image_id in sorted(image_ids)]


def build_native(upstream: Path, ops_root: Path, preset: dict, device: torch.device):
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if revision != UPSTREAM_REVISION or changes:
        raise ValueError("GroupPose must use the clean pinned upstream checkout")

    # Upstream imports its top-level models/util modules and the CUDA extension by name.
    sys.path[:0] = [str(upstream.resolve()), str(ops_root.resolve())]
    import MultiScaleDeformableAttention as extension
    from models.grouppose import grouppose
    from models.grouppose.backbones import backbone

    config = runpy.run_path(str(upstream / "config/grouppose.py"))
    options = {name: value for name, value in config.items() if not name.startswith("_")}
    args = Namespace(**(options | preset | {"device": str(device)}))

    def build_swin_for_restore(options):
        # A full strict checkpoint provides all weights. Skip upstream's separate
        # Swin classification initializer while retaining its backbone definition.
        position = backbone.build_position_encoding(options)
        network = backbone.build_swin_transformer(
            options.backbone, pretrain_img_size=384,
            out_indices=tuple(options.return_interm_indices),
            dilation=options.dilation, use_checkpoint=options.use_checkpoint,
        )
        model = backbone.Joiner(network, position)
        model.num_channels = network.num_features[4 - len(options.return_interm_indices):]
        return model

    if args.backbone == "swin_L_384_22k":
        with patch.object(grouppose, "build_backbone", build_swin_for_restore):
            model, criterion, postprocessors = grouppose.build_grouppose(args)
    elif args.backbone == "resnet50":
        with patch.object(backbone, "is_main_process", return_value=False):
            model, criterion, postprocessors = grouppose.build_grouppose(args)
    else:
        raise ValueError(f"Unsupported GroupPose backbone: {args.backbone}")
    kernel = Path(extension.__file__).resolve()
    return model.to(device), criterion, postprocessors["keypoints"], kernel


def read_image_for_model(image_root: Path, image: dict, preset: dict, device: torch.device):
    pixels = read_image(image_root / image["file_name"])
    size = (image["height"], image["width"])
    if tuple(pixels.shape[-2:]) != size:
        raise ValueError(f"Image dimensions differ from the annotation for {image['id']}")
    resized = resize_crowdpose_eval_image(
        pixels.float().div(255), shortest_side=preset["shortest_side"], max_size=preset["max_size"]
    )
    return resized.to(device)


def native_records(image_id: int, prediction: dict[str, torch.Tensor]) -> list[dict]:
    keypoints = prediction["keypoints"]
    if keypoints.ndim != 2 or keypoints.shape[1] != 42:
        raise ValueError("CrowdPose predictions require 14 ordered keypoint triples")
    return [
        {"image_id": image_id, "category_id": label, "score": score, "keypoints": pose}
        for label, score, pose in zip(
            prediction["labels"].tolist(), prediction["scores"].tolist(),
            keypoints.tolist(), strict=True,
        )
    ]


def evaluate(model, postprocess, args, preset: dict, provenance: dict) -> None:
    images = select_images(args.annotations, args.subset_manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    with (args.output / "images.jsonl").open("w") as ledger, torch.inference_mode():
        model.eval()
        for image in images:
            ident = image["id"]
            image_tensor = read_image_for_model(args.image_root, image, preset, args.device)
            size = image_tensor.new_tensor([[image["height"], image["width"]]])
            prediction = postprocess(model([image_tensor]), size)[0]
            detections = native_records(ident, prediction)
            ledger.write(json.dumps({"image_id": ident, "predictions": detections}) + "\n")
            records.extend(detections)

    result_file = args.output / "predictions.json"
    write_crowdpose_results(records, result_file)
    metrics = evaluate_crowdpose_results(
        result_file, args.annotations, image_ids=[image["id"] for image in images],
        include_pose_thresholds=True, pose_only=True,
    )
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    provenance["evaluated_images"] = len(images)
    provenance["subset_manifest_sha256"] = sha256(args.subset_manifest) if args.subset_manifest else None
    provenance["checkpoint_role"] = args.checkpoint_role
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(metrics)


def smoke_update(model, criterion, postprocess, args, preset: dict, provenance: dict) -> None:
    images = select_images(args.annotations, None)
    annotations = load_gt_annotations(args.annotations)
    image = next(
        image for image in images
        if any(
            ann["num_keypoints"] > 0 and not ann["iscrowd"] and min(ann["bbox"][2:]) > 0
            for ann in annotations.get(image["id"], [])
        )
    )
    people = [
        ann for ann in annotations[image["id"]]
        if ann["num_keypoints"] > 0 and not ann["iscrowd"] and min(ann["bbox"][2:]) > 0
    ]
    pixels = read_image_for_model(args.image_root, image, preset, args.device)
    scale = pixels.new_tensor([image["width"], image["height"]])
    points = pixels.new_tensor([ann["keypoints"] for ann in people]).reshape(-1, 14, 3)
    joints = (points[..., :2] / scale).flatten(1)
    boxes = pixels.new_tensor([ann["bbox"] for ann in people]) / scale.repeat(2)
    target = {
        "labels": torch.tensor([ann["category_id"] for ann in people], device=args.device),
        "keypoints": torch.cat((joints, points[..., 2].gt(0).float()), dim=1),
        "boxes": boxes,
        "area": boxes[:, 2:].prod(dim=1),
    }
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad), lr=1e-4,
    )
    model.train()
    losses = criterion(model([pixels]), [target])
    loss = sum(value * criterion.weight_dict[name] for name, value in losses.items()
               if name in criterion.weight_dict)
    if not torch.isfinite(loss):
        raise RuntimeError("GroupPose smoke loss is nonfinite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 0.1, error_if_nonfinite=True
    )
    if gradient_norm.item() == 0:
        raise RuntimeError("GroupPose smoke has zero gradients")
    optimizer.step()
    with torch.inference_mode():
        size = pixels.new_tensor([[image["height"], image["width"]]])
        prediction = postprocess(model.eval()([pixels]), size)[0]
        if not all(torch.isfinite(value).all() for value in prediction.values()):
            raise RuntimeError("GroupPose smoke prediction is nonfinite")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "preset": preset, "role": "smoke", "provenance": provenance,
            "image_id": image["id"], "loss": loss.item(), "gradient_norm": gradient_norm.item(),
        }, handle)
    print(f"One update on image {image['id']}, loss {loss.item():.4f}, gradient {gradient_norm.item():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("evaluate", "smoke-train"))
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--ops-root", type=Path, default=DEFAULT_OPS)
    parser.add_argument("--preset", type=Path, default=ROOT / "conf/group_pose/reduced.yaml")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-role", choices=("smoke", "released", "trained"))
    parser.add_argument("--subset-manifest", type=Path)
    parser.add_argument("--shortest-side", type=int)
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    args = parser.parse_args()
    preset = yaml.safe_load(args.preset.read_text())
    if args.shortest_side is not None:
        preset["shortest_side"] = args.shortest_side
    if args.max_size is not None:
        preset["max_size"] = args.max_size
    if min(preset["shortest_side"], preset["max_size"]) <= 0 or preset["num_body_points"] != 14:
        parser.error("CrowdPose requires positive resize dimensions and 14 joints")
    if args.device.type != "cuda":
        parser.error("the pinned native GroupPose backend requires CUDA")
    if args.action == "evaluate" and (args.checkpoint is None or args.checkpoint_role is None):
        parser.error("evaluate requires --checkpoint and --checkpoint-role")
    if args.action == "smoke-train" and (
        args.checkpoint or args.checkpoint_role or args.subset_manifest or preset["backbone"] != "resnet50"
    ):
        parser.error("smoke-train takes a ResNet-50 preset and training annotations")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    model, criterion, postprocess, kernel = build_native(args.upstream, args.ops_root, preset, args.device)
    provenance = {
        "upstream_revision": UPSTREAM_REVISION, "preset": preset, "seed": args.seed,
        "annotations": str(args.annotations.resolve()), "annotation_sha256": sha256(args.annotations),
        "trainable_parameters": sum(param.numel() for param in model.parameters() if param.requires_grad),
        "cuda_extension": str(kernel), "cuda_extension_sha256": sha256(kernel),
        "torch_version": str(torch.__version__), "device": torch.cuda.get_device_name(args.device),
        "cudnn_tf32": torch.backends.cudnn.allow_tf32, "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    if args.action == "smoke-train":
        smoke_update(model, criterion, postprocess, args, preset, provenance)
        return

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("role") == "smoke" and args.checkpoint_role != "smoke":
        parser.error("a smoke checkpoint must be labeled as such")
    model.load_state_dict(checkpoint["model"], strict=True)
    provenance["checkpoint_sha256"] = sha256(args.checkpoint)
    evaluate(model, postprocess, args, preset, provenance)


if __name__ == "__main__":
    main()
