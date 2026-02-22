"""
Central path defaults for pose-graph-diff.

All paths derive from two environment variables (set via .env):
    PROJECT_DIR  – repo root
    MMPOSE       – mmpose source clone (external/mmpose)

Both fall back to sensible relative paths if the env vars are not set.

For predefined model + dataset configurations, see src/configs.py.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
MMPOSE_DIR = Path(os.environ.get("MMPOSE", PROJECT_DIR / "external" / "mmpose"))
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
DATA_ROOT = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

# ---------------------------------------------------------------------------
# Download URLs  (used by prepare_* scripts)
# ---------------------------------------------------------------------------

CROWDPOSE_ANNOTATIONS_URL = (
    "https://download.openmmlab.com/mmpose/datasets/crowdpose_annotations.tar"
)