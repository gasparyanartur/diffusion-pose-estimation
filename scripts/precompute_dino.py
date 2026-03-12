"""Precompute DINO patch features for all images in a CrowdPose split.

Saves a dict  {image_id: Tensor(G*G, hidden_dim)}  as a .pt file so that
training / inference never has to run the DINO backbone again.

Output path: results/cache/crowdpose/<model_slug>/<split>.pt
where <model_slug> is the model name with '/' replaced by '_'.

Usage
-----
python scripts/precompute_dino.py \\
    --ann-path  data/crowdpose/annotations/crowdpose_train.json \\
    --img-root  data/crowdpose/images \\
    --split     train \\
    --model-name facebook/dinov3-vits16-pretrain-lvd1689m

python scripts/precompute_dino.py \\
    --ann-path  data/crowdpose/annotations/crowdpose_val.json \\
    --img-root  data/crowdpose/images \\
    --split     val \\
    --model-name facebook/dinov3-vits16-pretrain-lvd1689m
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
    p.add_argument("--ann-path",   required=True, help="COCO-style annotation JSON")
    p.add_argument("--img-root",   required=True, help="Root directory for images")
    p.add_argument("--split",      required=True, help="Split name, e.g. train / val / test")
    p.add_argument("--model-name", default="facebook/dinov2-small",
                   help="HuggingFace model ID for the DINO backbone")
    p.add_argument("--image-size", type=int, default=224,
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
    # Drop CLS token → (B, G*G, D)
    patch_feats = outputs.last_hidden_state[:, 1:].cpu()
    return {iid: patch_feats[i] for i, iid in enumerate(image_ids)}


def main() -> None:
    args = parse_args()

    # ---- output path --------------------------------------------------------
    model_slug = args.model_name.replace("/", "_")
    out_dir = Path(args.output_dir) / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.pt"

    if out_path.exists():
        print(f"Cache already exists at {out_path} — delete it to recompute.")
        return

    # ---- load image list ----------------------------------------------------
    image_paths = load_image_paths(args.ann_path, args.img_root)
    image_ids   = sorted(image_paths.keys())
    print(f"Found {len(image_ids)} images for split '{args.split}'")

    # ---- load DINO ----------------------------------------------------------
    print(f"Loading backbone: {args.model_name}")
    device    = args.device
    processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True,
                                                   size=args.image_size)
    backbone  = AutoModel.from_pretrained(args.model_name).to(device).eval()

    # ---- iterate in batches -------------------------------------------------
    cache: dict[int, torch.Tensor] = {}
    bs = args.batch_size

    for start in tqdm(range(0, len(image_ids), bs), unit="batch", desc="Extracting"):
        batch_ids   = image_ids[start : start + bs]
        batch_paths = [image_paths[iid] for iid in batch_ids]
        cache.update(extract_features(batch_ids, batch_paths, processor, backbone, device))

    # ---- save ---------------------------------------------------------------
    torch.save(cache, out_path)
    sample_tensor = next(iter(cache.values()))
    print(f"Saved {len(cache)} embeddings → {out_path}")
    print(f"  shape per image: {tuple(sample_tensor.shape)}  "
          f"(G*G={sample_tensor.shape[0]}, hidden_dim={sample_tensor.shape[1]})")


if __name__ == "__main__":
    main()
