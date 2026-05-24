#!/usr/bin/env bash
# Train DiffusionDet on CrowdPose (person detection).
#
# Usage:
#   bash scripts/train_diffusiondet_crowdpose.sh
#
# Options (env vars or flags):
#   NUM_GPUS        number of GPUs (default: 1)
#   CROWDPOSE_ROOT  path to CrowdPose data dir (default: data/crowdpose)
#   RESUME          resume from latest checkpoint (default: false)
#   EVAL_ONLY       run evaluation only; set MODEL_WEIGHTS too (default: false)
#   MODEL_WEIGHTS   checkpoint for eval-only mode
#   OUTPUT_DIR      where Detectron2 writes outputs (default: results/diffusiondet_crowdpose)
#
# Examples:
#   # Standard single-GPU training:
#   bash scripts/train_diffusiondet_crowdpose.sh
#
#   # Resume interrupted run:
#   RESUME=true bash scripts/train_diffusiondet_crowdpose.sh
#
#   # Evaluate a finished checkpoint:
#   EVAL_ONLY=true MODEL_WEIGHTS=results/diffusiondet_crowdpose/model_final.pth \
#       bash scripts/train_diffusiondet_crowdpose.sh
#
# After training, copy the final checkpoint:
#   cp results/diffusiondet_crowdpose/model_final.pth checkpoints/diffdet_crowdpose_res50.pth

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIFFUSIONDET_ROOT="$REPO_ROOT/external/diffusiondet"

NUM_GPUS="${NUM_GPUS:-1}"
CROWDPOSE_ROOT="${CROWDPOSE_ROOT:-$REPO_ROOT/data/crowdpose}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/diffusiondet_crowdpose}"
RESUME="${RESUME:-false}"
EVAL_ONLY="${EVAL_ONLY:-false}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-}"

CONFIG_FILE="$DIFFUSIONDET_ROOT/configs/diffdet.crowdpose.res50.yaml"

# Build the argument list.
ARGS=(
    --num-gpus "$NUM_GPUS"
    --config-file "$CONFIG_FILE"
    OUTPUT_DIR "$OUTPUT_DIR"
    CROWDPOSE_ROOT "$CROWDPOSE_ROOT"
)

if [[ "$RESUME" == "true" ]]; then
    ARGS+=(--resume)
fi

if [[ "$EVAL_ONLY" == "true" ]]; then
    ARGS+=(--eval-only)
    if [[ -n "$MODEL_WEIGHTS" ]]; then
        ARGS+=(MODEL.WEIGHTS "$MODEL_WEIGHTS")
    fi
fi

echo "============================================================"
echo " DiffusionDet CrowdPose training"
echo "   GPUs:        $NUM_GPUS"
echo "   Config:      $CONFIG_FILE"
echo "   Output dir:  $OUTPUT_DIR"
echo "   Resume:      $RESUME"
echo "   Eval only:   $EVAL_ONLY"
echo "============================================================"

cd "$DIFFUSIONDET_ROOT"
python train_net.py "${ARGS[@]}"
