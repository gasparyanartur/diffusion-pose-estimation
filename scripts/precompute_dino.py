"""Precompute DINO backbone features for all images in a CrowdPose split.

ViT backbone
  Saves {image_id: Tensor(G*G, hidden_dim)} — one flat patch-token tensor per image.

ConvNeXt backbone
  Saves {image_id: {'spatial':  Tensor(96, H/4, W/4),   # stage1 — high-res spatial (H,W vary with aspect ratio)
                    'semantic': Tensor(768, H/32, W/32), # stage4 — semantic bottleneck
                    'global':   Tensor(768)}}            # pooler_output — per-image summary

Output path: results/cache/crowdpose/<model_slug>/<split>.pt
where <model_slug> is the model name with '/' replaced by '_'.

Usage
-----
python scripts/precompute_dino.py \\
    --ann-path  data/crowdpose/annotations/ \\
    --img-root  data/crowdpose/images \\
    --split     all \\
    --model-name facebook/dinov3-convnext-small-pretrain-lvd1689m \\
    --backbone-type convnext

# Single split
python scripts/precompute_dino.py \\
    --ann-path  data/crowdpose/annotations/ \\
    --img-root  data/crowdpose/images \\
    --split     train \\
    --model-name facebook/dinov3-convnext-small-pretrain-lvd1689m \\
    --backbone-type convnext

# Legacy: pass a specific annotation file directly
python scripts/precompute_dino.py \\
    --ann-path  data/crowdpose/annotations/crowdpose_train.json \\
    --img-root  data/crowdpose/images \\
    --split     train \\
    --model-name facebook/dinov3-convnext-small-pretrain-lvd1689m \\
    --backbone-type convnext
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute DINO patch features.")
    p.add_argument("--ann-path",   default="data/crowdpose/annotations",
                   help="COCO-style annotation JSON, or a directory containing crowdpose_*.json files")
    p.add_argument("--img-root",   default="data/crowdpose/images",
                   help="Root directory for images")
    p.add_argument("--split",      default="all",
                   help="Split name (train / val / test) or 'all' to process every crowdpose_*.json found")
    p.add_argument("--model-name",    default="facebook/dinov3-convnext-small-pretrain-lvd1689m",
                   help="HuggingFace model ID for the DINO backbone")
    p.add_argument("--backbone-type", default="auto", choices=["auto", "vit", "convnext"],
                   help="Backbone type. 'auto' detects from the model class name.")
    p.add_argument("--image-size",    type=int, default=224,
                   help="Resolution to which images are resized before feeding DINO")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output-dir", default="results/cache/crowdpose",
                   help="Root output directory (model slug and split are appended)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_image_paths(ann_path: str, img_root: str) -> dict[int, str]:
    """Return {image_id: absolute_img_path} from a COCO-style JSON."""
    with open(ann_path) as fh:
        data = json.load(fh)
    root = Path(img_root)
    return {img["id"]: str(root / img["file_name"]) for img in data["images"]}


@torch.no_grad()
def extract_features(
    image_ids: list[int],
    paths: list[str],
    processor: AutoImageProcessor,
    model: AutoModel,
    device: str,
) -> dict[int, torch.Tensor]:
    """Run backbone on a batch; returns {image_id: Tensor(G*G, D)} on CPU."""
    pil_imgs = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # Drop CLS token and register tokens (e.g. 4 for DINOv3) → (B, G*G, D)
    n_registers = getattr(model.config, "num_register_tokens", 0)
    n_skip = 1 + n_registers
    patch_feats = outputs.last_hidden_state[:, n_skip:].cpu()
    return {iid: patch_feats[i] for i, iid in enumerate(image_ids)}


@torch.no_grad()
def extract_features_convnext(
    image_ids: list[int],
    paths: list[str],
    processor: AutoImageProcessor,
    model: AutoModel,
    device: str,
) -> dict[int, dict]:
    """Run ConvNeXt backbone one image at a time.

    Images with different aspect ratios would be padded when batched, which
    corrupts the pixel→feature-map coordinate mapping. Processing one image
    at a time avoids that entirely.

    Returns {image_id: {'spatial': Tensor(96, H/4, W/4),
                        'semantic': Tensor(768, H/32, W/32),
                        'global': Tensor(768)}} on CPU.
    """
    result = {}
    for iid, path in zip(image_ids, paths):
        pil_img = Image.open(path).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        # hidden_states index: 0=input pixels, 1=stage1, 2=stage2, 3=stage3, 4=stage4
        stage1 = outputs.hidden_states[1][0].cpu()  # (96, H/4, W/4)
        stage4 = outputs.hidden_states[4][0].cpu()  # (768, H/32, W/32)
        pooler = outputs.pooler_output[0].cpu()     # (768,)
        result[iid] = {"spatial": stage1, "semantic": stage4, "global": pooler}
    return result


def resolve_splits(ann_path: str, split: str) -> list[tuple[str, Path]]:
    """Return [(split_name, annotation_file)] to process.

    If ann_path is a directory:
      - split='all'  → find every crowdpose_*.json in the directory
      - split=<name> → use crowdpose_<name>.json inside the directory
    If ann_path is a file, use it directly with the given split name.
    """
    p = Path(ann_path)
    if p.is_dir():
        if split == "all":
            files = sorted(p.glob("crowdpose_*.json"))
            if not files:
                raise FileNotFoundError(f"No crowdpose_*.json files found in {p}")
            return [(f.stem.removeprefix("crowdpose_"), f) for f in files]
        else:
            return [(split, p / f"crowdpose_{split}.json")]
    else:
        return [(split, p)]


def run_split(
    split_name: str,
    ann_file: Path,
    args: argparse.Namespace,
    out_dir: Path,
    extractor,
    processor: AutoImageProcessor,
    backbone: AutoModel,
) -> None:
    split_dir = out_dir / split_name
    if split_dir.exists() and any(split_dir.iterdir()):
        print(f"[{split_name}] Cache directory already exists and is non-empty: {split_dir} — delete it to recompute.")
        return
    split_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_image_paths(str(ann_file), args.img_root)
    image_ids   = sorted(image_paths.keys())
    print(f"[{split_name}] {len(image_ids)} images  ({ann_file.name})")

    bs = args.batch_size
    sample = None
    for start in tqdm(range(0, len(image_ids), bs), unit="batch", desc=split_name):
        batch_ids   = image_ids[start : start + bs]
        batch_paths = [image_paths[iid] for iid in batch_ids]
        batch = extractor(batch_ids, batch_paths, processor, backbone, args.device)
        for iid, feats in batch.items():
            torch.save(feats, split_dir / f"{iid}.pt")
            sample = feats

    print(f"[{split_name}] Saved {len(image_ids)} embeddings → {split_dir}/")
    if sample is not None and isinstance(sample, dict):
        print(f"  keys per image: { {k: tuple(v.shape) for k, v in sample.items()} }")


def main() -> None:
    args = parse_args()

    splits = resolve_splits(args.ann_path, args.split)
    print(f"Splits to process: {[s for s, _ in splits]}")

    # ---- output directory ---------------------------------------------------
    model_slug = args.model_name.replace("/", "_")
    out_dir = Path(args.output_dir) / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load backbone once -------------------------------------------------
    print(f"Loading backbone: {args.model_name}")
    processor = AutoImageProcessor.from_pretrained(
        args.model_name, use_fast=True,
        do_center_crop=False, size={"shortest_edge": args.image_size},
    )
    backbone = AutoModel.from_pretrained(args.model_name).to(args.device).eval()

    if args.backbone_type == "auto":
        backbone_type = "convnext" if "convnext" in type(backbone).__name__.lower() else "vit"
        print(f"Auto-detected backbone type: {backbone_type}")
    else:
        backbone_type = args.backbone_type

    extractor = extract_features_convnext if backbone_type == "convnext" else extract_features

    # ---- process each split -------------------------------------------------
    for split_name, ann_file in splits:
        run_split(split_name, ann_file, args, out_dir, extractor, processor, backbone)


if __name__ == "__main__":
    main()
