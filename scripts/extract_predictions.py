#!/usr/bin/env python3
"""
Extract and cache predictions from any MMPose model.

Works with both top-down (RTMPose) and bottom-up (DEKR) models.

Usage:
    source .env && python scripts/extract_predictions.py \\
        rtmpose-m_8xb64-210e_crowdpose-256x192

    source .env && python scripts/extract_predictions.py \\
        dekr-hrnet-w32_8xb10-300e_crowdpose-512x512

    # With custom cache path
    source .env && python scripts/extract_predictions.py \\
        rtmpose-m_8xb64-210e_crowdpose-256x192 \\
        --cache results/cache/my_custom_name.pkl

    # Force re-extraction (overwrite existing cache)
    source .env && python scripts/extract_predictions.py \\
        rtmpose-m_8xb64-210e_crowdpose-256x192 \\
        --force
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import defaults
from src.configs import REGISTRY
from src.benchmark import download_checkpoint


def _group_predictions_by_image(predictions: list[dict]) -> list[dict]:
    """
    Group predictions by image_id for top-down models.
    
    Top-down models produce one prediction per person (N=1 each).
    This groups them into one prediction per image with all people stacked.
    
    Args:
        predictions: List of per-person predictions
        
    Returns:
        List of per-image predictions with stacked people
    """
    from collections import defaultdict
    
    # Group by image_id
    grouped = defaultdict(list)
    for pred in predictions:
        grouped[pred['image_id']].append(pred)
    
    # Combine predictions for each image
    combined = []
    for image_id, preds in sorted(grouped.items()):
        # Stack all people's data
        combined_pred = {
            'image_id': image_id,
            'img_path': preds[0]['img_path'],  # Same for all
            'keypoints': np.stack([p['keypoints'][0] for p in preds], axis=0),  # (N, K, 2)
            'keypoint_scores': np.stack([p['keypoint_scores'][0] for p in preds], axis=0),  # (N, K)
        }
        
        # Optional fields (only present in top-down)
        if 'bboxes' in preds[0]:
            combined_pred['bboxes'] = np.stack([p['bboxes'][0] for p in preds], axis=0)  # (N, 4/5)
        
        if 'bbox_scores' in preds[0]:
            combined_pred['bbox_scores'] = np.stack([p['bbox_scores'][0] for p in preds], axis=0)  # (N,)
        
        combined.append(combined_pred)
    
    return combined


def _get_cache_name(model_name: str) -> str:
    """Convert long model name to shorter cache name."""
    parts = model_name.split('_')
    if 'rtmpose' in model_name:
        variant = parts[0].split('-')[1]  # rtmpose-m → m
        return f"rtmpose-{variant}_crowdpose"
    elif 'dekr' in model_name:
        backbone = parts[0].split('-')[-1]  # dekr-hrnet-w32 → w32
        return f"dekr-{backbone}_crowdpose"
    else:
        # Fallback: truncate and sanitize
        return model_name.replace('_', '-')[:40]


# Map split name -> (ann_file, bbox_file_or_None) relative to data_root
# bbox_file is model-type-dependent so we use a sentinel and resolve below.
_SPLIT_ANN_FILES: dict[str, str] = {
    "train": "crowdpose/annotations/crowdpose_train.json",
    "val":   "crowdpose/annotations/crowdpose_val.json",
    "test":  "crowdpose/annotations/crowdpose_test.json",
}
# RTMPose (top-down) needs a bbox file per split; bottom-up ignores this.
# Only the test split bbox file is distributed with CrowdPose; val/train
# fall back to GT bounding boxes (oracle mode).
_SPLIT_BBOX_FILES: dict[str, str | None] = {
    "train": None,  # not publicly released
    "val":   None,  # not publicly released
    "test":  "crowdpose/annotations/det_for_crowd_test_0.1_0.5.json",
}


def extract_predictions(
    model_name: str,
    cache_path: Path,
    force: bool = False,
    include_features: bool = False,
    split: str = "test",
) -> None:
    """Run base model and cache predictions."""
    if cache_path.exists() and not force:
        print(f"[cache] Using existing predictions: {cache_path}")
        print(f"[cache] Use --force to re-extract")
        return

    cfg = dict(REGISTRY[model_name])

    # Override annotation / bbox file for the requested split
    if split not in _SPLIT_ANN_FILES:
        sys.exit(f"[error] Unknown split '{split}'. Choose from: {list(_SPLIT_ANN_FILES)}")
    cfg["ann_file"] = _SPLIT_ANN_FILES[split]
    # Only override bbox_file for top-down models (bottom-up has bbox_file=None).
    # If the split has no bbox file (val/train), set to None → GT oracle boxes.
    if cfg.get("bbox_file") is not None:
        bbox_override = _SPLIT_BBOX_FILES[split]
        cfg["bbox_file"] = bbox_override
        if bbox_override is None:
            print(f"[info] No detector bbox file available for split='{split}'; "
                  f"using GT bounding boxes (oracle mode).")
    
    # Download checkpoint if needed
    if cfg.get("checkpoint_url"):
        download_checkpoint(cfg["checkpoint_url"], cfg["checkpoint"])
    
    # Add MMPose to path (must be done before importing mmengine modules)
    mmpose_dir = cfg["mmpose_dir"]
    if str(mmpose_dir) not in sys.path:
        sys.path.insert(0, str(mmpose_dir))
    
    # Verify MMPose directory exists
    if not mmpose_dir.exists():
        sys.exit(f"[error] MMPose directory not found: {mmpose_dir}\n"
                 f"        Please clone MMPose: git clone https://github.com/open-mmlab/mmpose.git external/mmpose")
    
    from mmengine.config import Config
    from mmengine.runner import Runner
    
    print(f"[model] Loading: {model_name}  (split={split})")
    
    # Load config
    config = Config.fromfile(str(cfg["config_path"]))
    
    # Patch config (similar to src.benchmark._patch_config)
    data_root_str = str(cfg["data_root"]) + "/"
    ann_file = cfg["ann_file"]
    img_prefix = cfg["img_prefix"]
    metainfo = dict(from_file=str(cfg["metainfo_config"]))
    bbox_file = cfg.get("bbox_file")
    
    for dl in (config.test_dataloader,):
        dl.dataset.data_root = data_root_str
        dl.dataset.ann_file = ann_file
        dl.dataset.data_prefix = dict(img=img_prefix)
        dl.dataset.metainfo = metainfo
        if bbox_file is not None:
            dl.dataset.bbox_file = data_root_str + bbox_file
        elif "bbox_file" in dl.dataset:
            del dl.dataset["bbox_file"]
    
    config.test_evaluator.ann_file = data_root_str + ann_file
    config.load_from = str(cfg["checkpoint"])
    config.launcher = "none"
    config.work_dir = str(defaults.CACHE_DIR / "work_dir")
    
    # Build runner and model
    runner = Runner.from_cfg(config)
    model = runner.model

    # MMEngine only loads the checkpoint when the test loop starts (runner.test()),
    # NOT when you access runner.model directly. Load explicitly here.
    from mmengine.runner.checkpoint import load_checkpoint
    print(f"[model] Loading checkpoint: {cfg['checkpoint']}")
    load_checkpoint(model, str(cfg["checkpoint"]), map_location="cpu")

    model.eval()

    arch_type = cfg.get("architecture_type", "top-down" if bbox_file is not None else "bottom-up")
    is_bottom_up = arch_type == "bottom-up"
    print(f"[model] Architecture: {arch_type}")
    print(f"[extracting] Running inference on {len(runner.test_dataloader)} batches...")
    
    predictions = []
    
    with torch.no_grad():
        for idx, batch in enumerate(runner.test_dataloader):
            if idx % 100 == 0 and idx > 0:
                print(f"  Progress: {idx}/{len(runner.test_dataloader)} batches")
            
            # Get data samples
            data_samples = batch['data_samples']
            
            # Forward pass
            results = model.test_step(batch)
            
            # Extract predictions - format depends on architecture
            for i, result in enumerate(results):
                pred_instances = result.pred_instances
                
                # Get image ID (handle different field names)
                img_id = data_samples[i].get('img_id')
                if img_id is None:
                    img_id = data_samples[i].get('id', -1)
                
                pred_data = {
                    'image_id': int(img_id),
                    'img_path': data_samples[i].get('img_path', ''),
                    'keypoints': pred_instances.keypoints,  # (N, K, 2)
                    'keypoint_scores': pred_instances.keypoint_scores,  # (N, K)
                }
                
                # Add bbox info if available (top-down models)
                if hasattr(pred_instances, 'bboxes'):
                    pred_data['bboxes'] = pred_instances.bboxes  # (N, 4 or 5)
                
                if hasattr(pred_instances, 'bbox_scores'):
                    pred_data['bbox_scores'] = pred_instances.bbox_scores
                
                # Optionally extract intermediate features
                # This would require hooks on the model - TODO for future
                if include_features:
                    pass  # Reserved for future feature extraction
                
                predictions.append(pred_data)
    
    print(f"[extracting] Completed. Collected {len(predictions)} raw predictions.")
    
    # Convert tensors to numpy before grouping
    def clean_predict(pred):
        """Convert tensors to numpy arrays for caching."""
        cleaned = {}
        for k, v in pred.items():
            if isinstance(v, torch.Tensor):
                cleaned[k] = v.cpu().numpy()
            else:
                cleaned[k] = v
        return cleaned
    
    predictions = [clean_predict(p) for p in predictions]
    
    # Convert to per-image format
    if arch_type == "top-down":
        # Group predictions by image_id
        print(f"[grouping] Grouping {len(predictions)} predictions by image...")
        predictions = _group_predictions_by_image(predictions)
        print(f"[grouping] Grouped into {len(predictions)} images")
    else:
        # Bottom-up already produces per-image predictions, just verify
        image_ids = [p['image_id'] for p in predictions]
        unique_images = len(set(image_ids))
        if unique_images != len(predictions):
            print(f"[warning] Bottom-up has duplicate image_ids: {unique_images} unique vs {len(predictions)} total")
    
    # Compute statistics
    people_per_image = [p['keypoints'].shape[0] for p in predictions]
    
    print(f"\n[stats] Per-image statistics:")
    print(f"        Total images: {len(predictions)}")
    print(f"        People per image: min={min(people_per_image)}, max={max(people_per_image)}, mean={np.mean(people_per_image):.2f}")
    print(f"        Total people: {sum(people_per_image)}")
    
    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    cache_data = {
        'model_name': model_name,
        'architecture_type': arch_type,
        'num_keypoints': predictions[0]['keypoints'].shape[1] if predictions else 0,
        'format': 'per-image',  # One entry per image, keypoints shape (N_people, K, 2)
        'predictions': predictions,
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    file_size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"\n[cache] Saved to: {cache_path}")
    print(f"[cache] File size: {file_size_mb:.2f} MB")
    print(f"[cache] Format: per-image (universal for both top-down and bottom-up)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "model",
        nargs='?',
        help="Model name from REGISTRY (e.g., rtmpose-m_8xb64-210e_crowdpose-256x192)",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Cache output path (default: results/cache/<model>_predictions.pkl)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if cache exists",
    )
    p.add_argument(
        "--features",
        action="store_true",
        help="Extract intermediate features (experimental, not yet implemented)",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to run inference on (default: %(default)s).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all available models and exit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.list:
        print("Available models:")
        for name in REGISTRY:
            print(f"  {name}")
        return
    
    if not args.model:
        sys.exit("[error] model argument is required (or use --list to see available models)")
    
    if args.model not in REGISTRY:
        names = "\n  ".join(REGISTRY)
        sys.exit(f"[error] Unknown model: {args.model}\nAvailable:\n  {names}")
    
    cache_name = _get_cache_name(args.model)
    cache_path = args.cache or (defaults.CACHE_DIR / f"{cache_name}_{args.split}_predictions.pkl")

    extract_predictions(args.model, cache_path, args.force, args.features, split=args.split)


if __name__ == "__main__":
    main()
