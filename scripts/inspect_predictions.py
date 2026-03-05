#!/usr/bin/env python3
"""
Inspect cached predictions to understand the structure.

Usage:
    python scripts/inspect_predictions.py \\
        results/cache/rtmpose-m_crowdpose_predictions.pkl

    python scripts/inspect_predictions.py \\
        results/cache/dekr-w32_crowdpose_predictions.pkl

    # Inspect multiple caches
    python scripts/inspect_predictions.py \\
        results/cache/*.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def inspect_cache(cache_path: Path) -> None:
    """Load and print statistics about cached predictions."""
    if not cache_path.exists():
        print(f"[error] File not found: {cache_path}")
        return
    
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\n{'='*70}")
    print(f"Cache: {cache_path.name}")
    print(f"{'='*70}\n")
    
    print(f"Model: {data['model_name']}")
    print(f"Architecture: {data['architecture_type']}")
    print(f"Format: {data.get('format', 'unknown')}")
    print(f"Num keypoints: {data['num_keypoints']}")
    print(f"Total images: {len(data['predictions'])}\n")
    
    # Inspect first prediction structure
    if data['predictions']:
        pred = data['predictions'][0]
        print("First image structure:")
        for key, val in pred.items():
            if isinstance(val, np.ndarray):
                print(f"  {key:20s}: shape={str(val.shape):15s} dtype={val.dtype}")
            elif isinstance(val, (int, float)):
                print(f"  {key:20s}: {type(val).__name__} = {val}")
            else:
                val_str = str(val)[:50] + ('...' if len(str(val)) > 50 else '')
                print(f"  {key:20s}: {type(val).__name__} = {val_str}")
        
        # Check if bboxes are present
        has_bboxes = 'bboxes' in pred
        print(f"\n{'Optional fields:':30s}")
        print(f"  {'Bounding boxes:':20s} {'Yes' if has_bboxes else 'No (bottom-up model)'}")
        
        # Statistics
        all_kpts = [p['keypoints'] for p in data['predictions']]
        num_people_per_image = [kpts.shape[0] for kpts in all_kpts]
        total_people = sum(num_people_per_image)
        
        print(f"\n{'People per image statistics:':30s}")
        print(f"  {'Total people:':20s} {total_people}")
        print(f"  {'Min per image:':20s} {min(num_people_per_image)}")
        print(f"  {'Max per image:':20s} {max(num_people_per_image)}")
        print(f"  {'Mean per image:':20s} {np.mean(num_people_per_image):.2f}")
        print(f"  {'Median per image:':20s} {np.median(num_people_per_image):.1f}")
        print(f"  {'Std per image:':20s} {np.std(num_people_per_image):.2f}")
        
        # Check keypoint score distribution
        all_scores = np.concatenate([p['keypoint_scores'].flatten() 
                                     for p in data['predictions']])
        print(f"\n{'Keypoint score statistics:':30s}")
        print(f"  {'Min:':20s} {all_scores.min():.4f}")
        print(f"  {'Max:':20s} {all_scores.max():.4f}")
        print(f"  {'Mean:':20s} {all_scores.mean():.4f}")
        print(f"  {'Median:':20s} {np.median(all_scores):.4f}")
        print(f"  {'Std:':20s} {all_scores.std():.4f}")
        
        # Keypoint coordinate ranges (sanity check)
        all_coords = np.concatenate([p['keypoints'].reshape(-1, 2) 
                                     for p in data['predictions']])
        print(f"\n{'Keypoint coordinate ranges:':30s}")
        print(f"  {'X range:':20s} [{all_coords[:, 0].min():.1f}, {all_coords[:, 0].max():.1f}]")
        print(f"  {'Y range:':20s} [{all_coords[:, 1].min():.1f}, {all_coords[:, 1].max():.1f}]")
        
        # File size
        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"\n{'File size:':30s} {file_size_mb:.2f} MB")
        
        # Memory estimate per prediction
        mem_per_pred = file_size_mb * 1024 / len(data['predictions'])
        print(f"{'Avg size per prediction:':30s} {mem_per_pred:.2f} KB")
    else:
        print("[warning] No predictions found in cache!")
    
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "cache",
        type=Path,
        nargs='+',
        help="Path to cached predictions (.pkl files)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Expand globs if needed
    cache_files = []
    for path in args.cache:
        if '*' in str(path) or '?' in str(path):
            # Glob pattern
            parent = path.parent if path.parent != Path('.') else Path.cwd()
            pattern = path.name
            cache_files.extend(parent.glob(pattern))
        else:
            cache_files.append(path)
    
    if not cache_files:
        print("[error] No cache files found")
        return
    
    for cache_path in sorted(cache_files):
        inspect_cache(cache_path)


if __name__ == "__main__":
    main()
