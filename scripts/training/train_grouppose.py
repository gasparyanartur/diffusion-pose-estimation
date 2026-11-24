#!/usr/bin/env python3
"""Train reduced GroupPose on CrowdPose TRAIN+VAL and monitor fixed32 TEST."""

from __future__ import annotations

import argparse
import json
import os
import random
from argparse import Namespace
from itertools import islice
from pathlib import Path
from uuid import uuid4

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet50_Weights

from scripts.evaluation.evaluate_grouppose import (
    DEFAULT_OPS,
    ROOT,
    UPSTREAM_REVISION,
    build_native,
    evaluate,
    select_images,
    sha256,
)

BATCH_SIZE = 4
ACCUMULATE = 2
SEED = 42


def validate_protocol(data_root: Path, manifest: Path) -> tuple[Path, Path]:
    train = data_root / "json/crowdpose_trainval.json"
    test = data_root / "json/crowdpose_test.json"
    train_images = select_images(train, None)
    test_images = select_images(test, None)
    overlap = {image["id"] for image in train_images} & {image["id"] for image in test_images}
    if len(train_images) != 12000 or len(test_images) != 8000 or overlap:
        raise ValueError("expected official disjoint CrowdPose TRAIN+VAL (12000) and TEST (8000)")
    subset = select_images(test, manifest)
    if len(subset) != 32 or json.loads(manifest.read_text())["seed"] != SEED:
        raise ValueError("expected the shared seed-42 fixed32 TEST manifest")
    return train, test


def build_train_data(data_root: Path):
    from datasets.crowdpose import CocoDetection, make_coco_transforms
    from util.misc import collate_fn

    augmentation = Namespace(
        data_aug_scales=[256], data_aug_max_size=427,
        data_aug_scales2_resize=[128, 160, 192], data_aug_scales2_crop=[123, 192],
    )
    transforms = make_coco_transforms("train", args=augmentation)
    dataset = CocoDetection(data_root, "train", transforms)
    return dataset, collate_fn


def initialize_backbone(model) -> None:
    weights = ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=False, check_hash=True)
    weights.pop("fc.weight")
    weights.pop("fc.bias")
    model.backbone[0].body.load_state_dict(weights, strict=True)


def train_epoch(model, criterion, optimizer, loader, writer, step, device, limit):
    model.train()
    size = len(loader) if limit is None else min(limit, len(loader))
    total = 0.0
    optimizer.zero_grad(set_to_none=True)
    for index, (images, targets) in enumerate(islice(loader, size)):
        images = images.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        losses = criterion(model(images), targets)
        loss = sum(value * criterion.weight_dict[name] for name, value in losses.items()
                   if name in criterion.weight_dict)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at optimizer step {step}")
        group_size = min(ACCUMULATE, size - index // ACCUMULATE * ACCUMULATE)
        (loss / group_size).backward()
        total += loss.detach().item() / group_size
        if (index + 1) % ACCUMULATE and index + 1 != size:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        writer.add_scalar("train/loss_step", total, step)
        if step <= 3 or step % 20 == 0:
            print(f"optimizer_step={step} loss={total:.4f} grad_norm={grad_norm.item():.4f}", flush=True)
        total = 0.0
    return step


def quick_test(model, postprocess, args, preset, test, identity, epoch, step):
    output = args.output / "quick_test" / f"step{step:08d}_{uuid4().hex[:8]}"
    evaluation = Namespace(
        annotations=test,
        image_root=args.data_root / "images",
        subset_manifest=args.subset_manifest,
        output=output,
        device=args.device,
        checkpoint_role="smoke" if args.limit_train_batches else "trained",
    )
    provenance = dict(identity, epoch=epoch, optimizer_step=step, initial_weights_only=epoch == 0)
    evaluate(model, postprocess, evaluation, preset, provenance)
    return json.loads((output / "metrics.json").read_text())["AP"]


def save_checkpoint(output: Path, checkpoint: dict, is_best: bool) -> None:
    temporary = output / f".checkpoint_{uuid4().hex}.pth"
    try:
        torch.save(checkpoint, temporary)
        if is_best:
            staged_best = temporary.with_suffix(".best")
            os.link(temporary, staged_best)
            os.replace(staged_best, output / "best.pth")
        os.replace(temporary, output / "last.pth")
    finally:
        temporary.unlink(missing_ok=True)
        temporary.with_suffix(".best").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device.type != "cuda" or args.epochs < 1 or args.workers < 0:
        parser.error("training needs CUDA, positive epochs and a nonnegative worker count")
    if args.limit_train_batches is not None and args.limit_train_batches < 1:
        parser.error("--limit-train-batches must be positive")

    train, test = validate_protocol(args.data_root, args.subset_manifest)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    preset = yaml.safe_load((ROOT / "conf/group_pose/reduced.yaml").read_text())
    model, criterion, postprocess, kernel = build_native(args.upstream, DEFAULT_OPS, preset, args.device)
    from util.get_param_dicts import get_param_dict

    dataset, collate_fn = build_train_data(args.data_root)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers,
        generator=generator, collate_fn=collate_fn,
    )
    optimizer = torch.optim.AdamW(
        get_param_dict(Namespace(lr_backbone=1e-5), model), lr=1e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 70)
    script = Path(__file__).resolve()
    identity = {
        "preset": preset, "seed": SEED, "batch_size": BATCH_SIZE, "accumulation": ACCUMULATE,
        "upstream_revision": UPSTREAM_REVISION, "native_kernel_sha256": sha256(kernel),
        "train_annotation_sha256": sha256(train), "test_annotation_sha256": sha256(test),
        "subset_sha256": sha256(args.subset_manifest), "train_dataset_size": len(dataset),
        "limit_train_batches": args.limit_train_batches, "workers": args.workers,
        "backbone_initialization": ResNet50_Weights.IMAGENET1K_V1.url,
        "training_script_sha256": sha256(script),
        "eval_runner_sha256": sha256(ROOT / "scripts/evaluation/evaluate_grouppose.py"),
        "evaluator_sha256": sha256(ROOT / "src/models/pipelines/pose_estimators/diffusionregpose/evaluation.py"),
    }

    start_epoch, step, best_ap = 0, 0, -1.0
    if args.resume:
        checkpoint = torch.load(args.output / "last.pth", map_location="cpu", weights_only=True)
        if checkpoint["identity"] != identity:
            raise ValueError("resume configuration differs from saved training configuration")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"], args.device)
        random.setstate(checkpoint["python_rng"])
        generator.set_state(checkpoint["loader_rng"])
        start_epoch, step, best_ap = checkpoint["epoch"], checkpoint["step"], checkpoint["best_ap"]
        if start_epoch >= args.epochs:
            parser.error("requested epochs are already complete")
        del checkpoint
    else:
        initialize_backbone(model)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "run.json").write_text(json.dumps(identity | {"planned_epochs": args.epochs}, indent=2) + "\n")
    with SummaryWriter(str(args.output / "tb_logs"), purge_step=step + 1 if args.resume else None) as writer:
        if not args.resume:
            initial_ap = quick_test(model, postprocess, args, preset, test, identity, 0, step)
            writer.add_scalar("quick-test/pose_AP", initial_ap, step)
            writer.flush()
            print(f"initial_fixed32_pose_AP={initial_ap:.6f}", flush=True)
        for epoch in range(start_epoch + 1, args.epochs + 1):
            step = train_epoch(
                model, criterion, optimizer, loader, writer, step, args.device, args.limit_train_batches
            )
            scheduler.step()
            ap = quick_test(model, postprocess, args, preset, test, identity, epoch, step)
            is_best = ap > best_ap
            best_ap = max(ap, best_ap)
            save_checkpoint(args.output, {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "identity": identity,
                "epoch": epoch, "step": step, "best_ap": best_ap, "preset": preset,
                "role": "smoke" if args.limit_train_batches else "trained",
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(args.device),
                "python_rng": random.getstate(), "loader_rng": generator.get_state(),
            }, is_best)
            writer.add_scalar("quick-test/pose_AP", ap, step)
            writer.flush()
            print(f"epoch={epoch}/{args.epochs} optimizer_step={step} fixed32_pose_AP={ap:.6f}", flush=True)


if __name__ == "__main__":
    main()
