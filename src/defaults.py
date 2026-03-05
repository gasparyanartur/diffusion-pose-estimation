"""
Central path defaults for pose-graph-diff.

Paths are auto-detected relative to the project root. Optionally, you can
override them via .env file (loaded with python-dotenv).

For predefined model + dataset configurations, see src/configs.py.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env from project root
    _env_path = Path(__file__).resolve().parents[1] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    # python-dotenv not installed, fall back to os.environ
    pass

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------

# Auto-detect project root from this file's location
_DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]

# Get from env (if set and not empty), otherwise use auto-detected path
_env_project_dir = os.environ.get("PROJECT_DIR", "").strip()
PROJECT_DIR = Path(_env_project_dir) if _env_project_dir else _DEFAULT_PROJECT_DIR

_env_mmpose = os.environ.get("MMPOSE", "").strip()
MMPOSE_DIR = Path(_env_mmpose) if _env_mmpose else (PROJECT_DIR / "external" / "mmpose")

CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
DATA_ROOT = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"

# ---------------------------------------------------------------------------
# Download URLs  (used by prepare_* scripts)
# ---------------------------------------------------------------------------

CROWDPOSE_ANNOTATIONS_URL = (
    "https://download.openmmlab.com/mmpose/datasets/crowdpose_annotations.tar"
)