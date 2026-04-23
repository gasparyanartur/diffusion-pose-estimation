#!/usr/bin/env bash
# Run inference-parameter ablation for the diffusion refiner.
# Runs the sweep twice: standard DDIM and raw-refiner (single-pass).
#
# Usage:
#   scripts/run_inference_ablation.sh <checkpoint> <pred_cache> <ann_file>
#
# Results:
#   results/inference-ablation/ddim/<task_id>.json   (DDIM mode)
#   results/inference-ablation/raw/<task_id>.json    (raw-refiner mode)
#   results/inference-ablation/ddim/results.csv
#   results/inference-ablation/raw/results.csv
#
# Example:
#   scripts/run_inference_ablation.sh \
#       results/sweeps/arch-message-passing/task_1/20260316_003813/checkpoints/epoch140-running_loss_5=0.0155.ckpt \
#       results/cache/rtmpose-m_crowdpose_val_predictions.pkl \
#       data/crowdpose/annotations/crowdpose_val.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SWEEP_FILE="${PROJECT_ROOT}/sweeps/inference-ablation.sweep"

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <checkpoint> <pred_cache> <ann_file>" >&2
    exit 1
fi

CHECKPOINT="$1"
PRED_CACHE="$2"
ANN_FILE="$3"

TOTAL=$(grep -v '^\s*#' "$SWEEP_FILE" | grep -cv '^\s*$' || true)
echo "Inference ablation: ${TOTAL} tasks × 2 modes (ddim, raw)"
echo ""

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Run all tasks for one mode, then aggregate
# ---------------------------------------------------------------------------
run_mode() {
    local out_dir="$1"
    local mode_label="$2"
    shift 2
    local extra_args=("$@")

    echo "================================================================"
    echo "  Mode      : ${mode_label}"
    echo "  Output    : ${out_dir}"
    echo "================================================================"
    echo ""

    for i in $(seq 1 "$TOTAL"); do
        PARAMS=$(grep -v '^\s*#' "$SWEEP_FILE" | grep -v '^\s*$' | sed -n "${i}p")
        read -ra PARAM_ARRAY <<< "$PARAMS"

        echo "  Task ${i}/${TOTAL}: ${PARAMS}"
        python scripts/benchmark_diffusion_refiner.py \
            --checkpoint "$CHECKPOINT" \
            --pred-cache  "$PRED_CACHE" \
            --ann-file    "$ANN_FILE" \
            --task-id     "$i" \
            --out-dir     "$out_dir" \
            "${PARAM_ARRAY[@]}" \
            "${extra_args[@]}"
        echo ""
    done

    echo "Aggregating ${out_dir}..."
    "${SCRIPT_DIR}/run_benchmark_sweep.sh" "$SWEEP_FILE" --aggregate "$out_dir"
    echo ""
}

run_mode "${PROJECT_ROOT}/results/inference-ablation/ddim" "ddim"
run_mode "${PROJECT_ROOT}/results/inference-ablation/raw"  "raw-refiner" --raw-refiner

echo "================================================================"
echo "  Inference ablation complete."
echo "  DDIM results : results/inference-ablation/ddim/results.csv"
echo "  Raw  results : results/inference-ablation/raw/results.csv"
echo "================================================================"
