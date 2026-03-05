#!/usr/bin/env python3
"""
Example: how to use cached predictions with a refiner.

This shows how the universal per-image format works with both
top-down and bottom-up models.
"""

import pickle
import sys
from pathlib import Path

import numpy as np


def example_graph_refiner(keypoints, keypoint_scores):
    """
    Example refiner that processes all poses in an image together.
    
    Args:
        keypoints: (N, K, 2) - N people, K keypoints, 2 coords
        keypoint_scores: (N, K) - confidence scores
        
    Returns:
        refined_keypoints: (N, K, 2) - refined poses
    """
    N, K, _ = keypoints.shape
    
    print(f"  Processing {N} people with {K} keypoints each")
    
    # Example: simple refinement (add small noise for demo)
    # Your actual refiner would:
    # 1. Build a graph connecting keypoints within/across people
    # 2. Use GNN or attention to refine based on context
    # 3. Return improved keypoint estimates
    
    refined = keypoints.copy()
    
    # Single person case (N=1): trivial graph
    if N == 1:
        print(f"    Single person - no inter-person context")
    else:
        print(f"    Multiple people - using inter-person relationships")
        # Could compute pairwise distances, build interaction graph, etc.
    
    return refined


def process_cache(cache_path: Path):
    """Load cache and process with refiner."""
    
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nModel: {data['model_name']}")
    print(f"Architecture: {data['architecture_type']}")
    print(f"Format: {data['format']}")
    print(f"Total images: {len(data['predictions'])}\n")
    
    # Process first 3 images as example
    for i, pred in enumerate(data['predictions'][:3]):
        print(f"Image {i+1}: ID={pred['image_id']}")
        
        # Extract data (works for both top-down and bottom-up!)
        keypoints = pred['keypoints']  # (N, K, 2)
        scores = pred['keypoint_scores']  # (N, K)
        
        # Optional: bounding boxes (only for top-down)
        if 'bboxes' in pred:
            bboxes = pred['bboxes']
            print(f"  Input: {keypoints.shape[0]} people with bboxes")
        else:
            print(f"  Input: {keypoints.shape[0]} people (no bboxes, bottom-up)")
        
        # Run refiner
        refined_keypoints = example_graph_refiner(keypoints, scores)
        
        # Refinement done! Shape is preserved
        assert refined_keypoints.shape == keypoints.shape
        print(f"  Output: {refined_keypoints.shape}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/example_refiner.py <cache.pkl>")
        print("\nExample:")
        print("  python scripts/example_refiner.py results/cache/rtmpose-m_crowdpose_predictions.pkl")
        print("  python scripts/example_refiner.py results/cache/dekr-w32_crowdpose_predictions.pkl")
        sys.exit(1)
    
    cache_path = Path(sys.argv[1])
    
    if not cache_path.exists():
        print(f"Error: Cache not found: {cache_path}")
        sys.exit(1)
    
    process_cache(cache_path)


if __name__ == "__main__":
    main()
