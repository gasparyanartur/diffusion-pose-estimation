"""Generate YOLO person bounding boxes for CrowdPose (train / val / test).

Output (per split):
    data/crowdpose/annotations/boxes_crowdpose_{split}.json
    → list of {"bbox": [x, y, w, h], "category_id": 1, "image_id": int, "score": float}

Usage:
    # Generate all splits (default thresholds: conf=0.5, nms=0.45)
    python scripts/generate_boxes_yolo.py

    # Generate only test split
    python scripts/generate_boxes_yolo.py --split test

    # Match reference file thresholds (det_for_crowd_test_0.1_0.5.json) and verify
    python scripts/generate_boxes_yolo.py --split test \\
        --conf-threshold 0.1 --nms-threshold 0.5 --verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from src.configs import defaults

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PERSON_CLASS_ID = 0  # COCO class 0 = person
_DEFAULT_MODEL   = "yolo26l.pt"

_IOU_THRESHOLDS = [0.3, 0.5, 0.75, 0.9]
_SIZE_BUCKETS   = [("small  (h< 50px)", 0, 50),
                   ("medium (50–150px)", 50, 150),
                   ("large  (h>150px)", 150, float("inf"))]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate YOLO bounding boxes for CrowdPose.")
    p.add_argument("--model", default=_DEFAULT_MODEL,
                   help="Model name or path (default: yolo26l.pt, auto-downloaded by ultralytics)")
    p.add_argument("--split", default="all", choices=["train", "val", "test", "all"],
                   help="Which split(s) to process (default: all)")
    p.add_argument("--ann-dir", default=str(defaults.DATA_ROOT / "crowdpose" / "annotations"))
    p.add_argument("--img-root", default=str(defaults.DATA_ROOT / "crowdpose" / "images"))
    p.add_argument("--weights-dir", default=str(defaults.CHECKPOINTS_DIR),
                   help="Directory checked first for the model file; falls back to ultralytics cache")
    p.add_argument("--out-dir", default=str(defaults.DATA_ROOT / "crowdpose" / "annotations"))
    p.add_argument("--conf-threshold", type=float, default=0.5,
                   help="Confidence threshold (reference file used 0.1)")
    p.add_argument("--nms-threshold", type=float, default=0.45,
                   help="NMS IoU threshold (reference file used 0.5)")
    p.add_argument("--subset", type=int, default=None,
                   help="Process only this many images per split (for quick testing)")
    p.add_argument("--verify", action="store_true",
                   help="After generating test split, compare against det_for_crowd_test_0.1_0.5.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_yolo_model(weights_dir: Path, model_name: str) -> YOLO:
    local = weights_dir / model_name
    return YOLO(str(local) if local.exists() else model_name)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_persons(
    model: YOLO,
    image_path: Path,
    conf_threshold: float,
    nms_threshold: float,
) -> list[dict]:
    """Run YOLO inference on one image; return person detections.

    Returns list of {"bbox": [x, y, w, h], "score": float} in pixel coordinates.
    """
    results = model.predict(
        str(image_path),
        conf=conf_threshold,
        iou=nms_threshold,
        classes=[_PERSON_CLASS_ID],
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
    confs      = results[0].boxes.conf.cpu().numpy()

    return [
        {
            "bbox":  [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(conf),
        }
        for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs)
    ]


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def load_image_records(ann_json: Path) -> list[tuple[int, str]]:
    with open(ann_json) as fh:
        data = json.load(fh)
    return [(img["id"], img["file_name"]) for img in data["images"]]


def run_split(
    split: str,
    model: YOLO,
    ann_dir: Path,
    img_root: Path,
    conf_threshold: float,
    nms_threshold: float,
    out_dir: Path,
    subset: int | None = None,
) -> tuple[Path, set[int]]:
    ann_json = ann_dir / f"crowdpose_{split}.json"
    records  = load_image_records(ann_json)
    if subset is not None and subset < len(records):
        records = records[:subset]
        print(f"  [subset] {len(records)} / {len(load_image_records(ann_json))} images")

    detections = []
    for image_id, file_name in tqdm(records, desc=f"  {split}"):
        image_path = img_root / file_name
        if not image_path.exists():
            print(f"\n  [warn] image not found: {image_path}")
            continue

        persons = detect_persons(model, image_path, conf_threshold, nms_threshold)
        for det in persons:
            detections.append({
                "bbox":        det["bbox"],
                "category_id": 1,
                "image_id":    image_id,
                "score":       det["score"],
            })

    out_path = out_dir / f"boxes_crowdpose_{split}.json"
    with open(out_path, "w") as fh:
        json.dump(detections, fh)

    n_imgs = len(records)
    n_det  = len(detections)
    print(f"  [ok] {out_path.name}  —  {n_det} detections across {n_imgs} images")
    processed_ids = {image_id for image_id, _ in records}
    return out_path, processed_ids


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _box_iou(a: list[float], b: list[float]) -> float:
    """IoU of two COCO [x, y, w, h] boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _group_by_image(detections: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for det in detections:
        grouped.setdefault(det["image_id"], []).append(det)
    return grouped


def compare_with_reference(
    generated_path: Path,
    reference_path: Path,
    subset_image_ids: set[int] | None = None,
) -> None:
    """Compare generated boxes against the reference detection file.

    subset_image_ids: when provided, only these images are compared — so
    verification covers exactly the same images that were just generated.
    """
    with open(generated_path) as fh:
        generated = json.load(fh)
    with open(reference_path) as fh:
        reference = json.load(fh)

    gen_by_img = _group_by_image(generated)
    ref_by_img = _group_by_image(reference)

    gen_ids    = set(gen_by_img)
    ref_ids    = set(ref_by_img)
    common_ids = gen_ids & ref_ids
    if subset_image_ids is not None:
        common_ids &= subset_image_ids

    print(f"\n--- Verification against {reference_path.name} ---")
    print(f"  Generated  : {len(generated):6d} detections, {len(gen_ids):5d} images")
    print(f"  Reference  : {len(reference):6d} detections, {len(ref_ids):5d} images")
    suffix = " (subset)" if subset_image_ids is not None else ""
    print(f"  Comparing  : {len(common_ids):5d} common images{suffix}")

    if not common_ids:
        print("  [warn] No common image IDs — cannot compare further.")
        return

    count_diffs = [
        abs(len(gen_by_img.get(img_id, [])) - len(ref_by_img.get(img_id, [])))
        for img_id in common_ids
    ]
    print(f"  Mean |Δ detections per image| : {np.mean(count_diffs):.2f}")

    sample_ids    = sorted(common_ids)
    thresh_counts = {t: 0 for t in _IOU_THRESHOLDS}
    size_counts   = {label: [0, 0] for label, *_ in _SIZE_BUCKETS}  # [matched@0.5, total]
    total_ref     = 0

    for img_id in sample_ids:
        ref_boxes = [d["bbox"] for d in ref_by_img[img_id]]
        gen_boxes = [d["bbox"] for d in gen_by_img[img_id]]
        total_ref += len(ref_boxes)

        for rb in ref_boxes:
            best_iou = max((_box_iou(rb, gb) for gb in gen_boxes), default=0.0)
            for t in _IOU_THRESHOLDS:
                if best_iou > t:
                    thresh_counts[t] += 1

            h = rb[3]
            for label, lo, hi in _SIZE_BUCKETS:
                if lo <= h < hi:
                    size_counts[label][1] += 1
                    if best_iou > 0.5:
                        size_counts[label][0] += 1
                    break

    print(f"\n  IoU match rates (sample {len(sample_ids)} images, per ref box):")
    for t in _IOU_THRESHOLDS:
        rate = thresh_counts[t] / total_ref if total_ref else 0.0
        print(f"    IoU > {t:.2f} : {thresh_counts[t]:5d} / {total_ref} = {rate:.1%}")

    print(f"\n  IoU>0.5 match rate by box height:")
    for label, (matched, total) in size_counts.items():
        rate = matched / total if total else 0.0
        print(f"    {label} : {matched:4d} / {total:4d} = {rate:.1%}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    weights_dir = Path(args.weights_dir)
    ann_dir     = Path(args.ann_dir)
    img_root    = Path(args.img_root)
    out_dir     = Path(args.out_dir)

    print("=== YOLO bounding box generation for CrowdPose ===")
    print(f"  model          : {args.model}")
    print(f"  conf threshold : {args.conf_threshold}")
    print(f"  nms threshold  : {args.nms_threshold}")
    print()

    print("Loading model …")
    model = load_yolo_model(weights_dir, args.model)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        print(f"\nProcessing: {split}")
        out_path, processed_ids = run_split(
            split, model,
            ann_dir, img_root,
            args.conf_threshold, args.nms_threshold,
            out_dir,
            subset=args.subset,
        )

        if args.verify and split == "test":
            reference = ann_dir / "det_for_crowd_test_0.1_0.5.json"
            if reference.exists():
                compare_with_reference(out_path, reference,
                                       subset_image_ids=processed_ids)
            else:
                print(f"  [skip verify] reference not found: {reference}")

    print("Done.")


if __name__ == "__main__":
    main()
