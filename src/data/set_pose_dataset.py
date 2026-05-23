"""Backward-compatible re-exports. Content moved to :mod:`src.data.datasets`."""

from src.data.datasets import SetPoseDataset, compute_person_scale, set_pose_collate

__all__ = ["SetPoseDataset", "set_pose_collate", "compute_person_scale"]
