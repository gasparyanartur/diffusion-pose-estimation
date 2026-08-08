#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-$ROOT_DIR/venv}"
PYTHON_VERSION="3.10"
PYTHON="$VENV_DIR/bin/python"
UV="$(command -v uv || true)"

if [[ -z "$UV" ]]; then
  echo "error: uv is required; install it before running this script." >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Creating virtual environment at $VENV_DIR"
  "$UV" venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi

if [[ "$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "$PYTHON_VERSION" ]]; then
  echo "error: $VENV_DIR does not use Python $PYTHON_VERSION" >&2
  exit 1
fi

pip_install() {
  "$UV" pip install --python "$PYTHON" "$@"
}

pip_install --upgrade pip setuptools wheel ninja packaging numpy==1.26.4 cython

# Install the repository's tested Torch pair before packages that inspect Torch
# while building. The external requirements are deliberately not installed
# wholesale because they contain broad, legacy Torch constraints.
pip_install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip_install mmcv==2.2.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

# Install pinned root requirements except entries that need controlled handling.
FILTERED_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$FILTERED_REQUIREMENTS"' EXIT
awk '
  $0 ~ /^mmpose[[:space:]]*@/ { next }
  $0 ~ /^torch==/ { next }
  $0 ~ /^torchvision==/ { next }
  $0 ~ /^mmcv==/ { next }
  $0 ~ /^chumpy==/ { next }
  $0 ~ /^torch-linear-assignment([[:space:]]|$)/ { next }
  $0 ~ /^-e[[:space:]]+\.$/ { next }
  { print }
' "$ROOT_DIR/requirements.txt" > "$FILTERED_REQUIREMENTS"
pip_install --no-build-isolation -r "$FILTERED_REQUIREMENTS"

# Local OpenMMLab packages must be installed after Torch/MMCV and without
# dependency resolution from their legacy metadata.
pip_install --no-build-isolation --no-deps \
  -e "$ROOT_DIR/external/mmpose" \
  -e "$ROOT_DIR/external/mmdet"
pip_install --no-build-isolation -e "$ROOT_DIR" --no-deps

# Mandatory CrowdPose and Detectron2 dependencies.
pip_install pycocotools xtcocotools --no-deps
pip_install --no-build-isolation \
  "git+https://github.com/cocodataset/panopticapi.git#egg=panopticapi"
pip_install --no-build-isolation --no-deps \
  "git+https://github.com/Jeff-sjtu/CrowdPose.git#subdirectory=crowdpose-api/PythonAPI&egg=crowdposetools"
pip_install --no-build-isolation --no-deps \
  "git+https://github.com/facebookresearch/detectron2.git"
pip_install fvcore iopath yacs tabulate cloudpickle pydot future omegaconf hydra-core

# DiffusionRegPose-specific runtime packages. chumpy has broken isolated build
# metadata and therefore needs the already-provisioned environment.
pip_install chumpy json_tricks munkres scipy shapely terminaltables tqdm timm \
  submitit tensorboardX addict yapf
pip_install --no-build-isolation chumpy==0.70

CUDA_HOME="${CUDA_HOME:-$($PYTHON -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or "")')}"
if [[ -z "$CUDA_HOME" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "error: CUDA_HOME must point to a complete CUDA toolkit containing bin/nvcc." >&2
  exit 1
fi
export CUDA_HOME
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-2}"

# torch-linear-assignment has the same strict CUDA-version check as the
# DiffusionRegPose operator. Temporarily relax that check inside the disposable
# target environment while building, then restore the installed PyTorch file.
CPP_EXTENSION_FILE="$($PYTHON -c 'import torch.utils.cpp_extension as cpp; print(cpp.__file__)')"
CPP_EXTENSION_BACKUP="$(mktemp)"
cp "$CPP_EXTENSION_FILE" "$CPP_EXTENSION_BACKUP"
restore_cpp_extension() {
  cp "$CPP_EXTENSION_BACKUP" "$CPP_EXTENSION_FILE"
  rm -f "$CPP_EXTENSION_BACKUP"
}
trap restore_cpp_extension EXIT
$PYTHON - "$CPP_EXTENSION_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needle = "raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))"
if needle not in text:
    raise SystemExit("CUDA version guard format was not found in torch cpp_extension.py")
path.write_text(text.replace(needle, "warnings.warn(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))"))
PY
pip_install --no-build-isolation --no-deps torch-linear-assignment==0.0.6
restore_cpp_extension
trap - EXIT

OPS_DIR="$ROOT_DIR/external/DiffusionRegPose/models/diffusionregpose/ops"
if [[ ! -f "$OPS_DIR/setup.py" ]]; then
  echo "error: DiffusionRegPose CUDA extension source is missing: $OPS_DIR" >&2
  exit 1
fi

# PyTorch 2.4.1+cu121 rejects nvcc 13.0 before compilation. The source builds
# and its operator tests pass with the available CUDA 13.0 toolkit, so bypass
# only this version guard. A matching CUDA 12.1 toolkit remains preferable.
(
  cd "$OPS_DIR"
  rm -rf build MultiScaleDeformableAttention*.so
  "$PYTHON" -c 'import runpy; import torch.utils.cpp_extension as cpp; cpp._check_cuda_version = lambda *args, **kwargs: None; runpy.run_path("setup.py", run_name="__main__")' build_ext --inplace
)

# Register the source-tree extension directory in this environment so imports
# work from the repository root without requiring a manual PYTHONPATH export.
# The extension itself embeds the selected Torch library path at link time, so
# direct imports also work without requiring `import torch` first.
SITE_PACKAGES="$($PYTHON -c 'import site; print(site.getsitepackages()[0])')"
mkdir -p "$SITE_PACKAGES"
printf '%s\n' "$OPS_DIR" > "$SITE_PACKAGES/pose_graph_diffusionregpose_ops.pth"

PYTHONPATH="$ROOT_DIR:$ROOT_DIR/external/DiffusionRegPose" "$PYTHON" - <<'PY'
import importlib

for name in (
    "torch", "torchvision", "mmcv", "mmengine", "mmpose", "mmdet",
    "pycocotools", "crowdposetools", "panopticapi", "detectron2",
  "torch_linear_assignment", "MultiScaleDeformableAttention",
):
    importlib.import_module(name)

import main
print("Shared environment installation and import checks passed.")
PY

echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
