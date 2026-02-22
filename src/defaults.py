"""
Central defaults for pose-graph-diff.

All paths derive from two environment variables (set via .env):
    PROJECT_DIR  – repo root
    MMPOSE       – mmpose source clone (external/mmpose)

Both fall back to sensible relative paths if the env vars are not set.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
MMPOSE_DIR = Path(os.environ.get("MMPOSE", PROJECT_DIR / "external" / "mmpose"))

# ---------------------------------------------------------------------------
# MMPose configs
# ---------------------------------------------------------------------------

RTMPOSE_CROWDPOSE_CONFIG = (
    MMPOSE_DIR
    / "configs/body_2d_keypoint/rtmpose/crowdpose"
    / "rtmpose-m_8xb64-210e_crowdpose-256x192.py"
)
CROWDPOSE_METAINFO_CONFIG = MMPOSE_DIR / "configs/_base_/datasets/crowdpose.py"

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"

RTMPOSE_CROWDPOSE_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth"
)
RTMPOSE_CROWDPOSE_CHECKPOINT = CHECKPOINTS_DIR / "rtmpose-m_crowdpose.pth"

# ---------------------------------------------------------------------------
# CrowdPose data  (all annotation/image paths are relative to DATA_ROOT)
# ---------------------------------------------------------------------------

DATA_ROOT = PROJECT_DIR / "data"

CROWDPOSE_ANN_FILE = "crowdpose/annotations/crowdpose_test.json"
CROWDPOSE_IMG_PREFIX = "crowdpose/images/"

# Pre-generated YOLOv3 detector boxes – required for official benchmark numbers.
# Run  scripts/prepare_crowdpose.py  to download.
CROWDPOSE_BBOX_FILE = "crowdpose/annotations/det_for_crowd_test_0.1_0.5.json"

CROWDPOSE_ANNOTATIONS_URL = (
    "https://download.openmmlab.com/mmpose/datasets/crowdpose_annotations.tar"
)

# ---------------------------------------------------------------------------
# Work / output directories
# ---------------------------------------------------------------------------

RESULTS_DIR = PROJECT_DIR / "rtmpose_crowdpose_benchmark"
RTMPOSE_BENCHMARK_RESULTS = RESULTS_DIR / "results.json"