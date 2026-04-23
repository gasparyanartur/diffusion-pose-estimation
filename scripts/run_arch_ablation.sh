#!/usr/bin/env bash
# Architecture ablation: message-passing × raw-refiner (2×2 = 4 combinations).
# Inference params fixed at baseline: steps=1, noise-frac=0.2, passes=1, cfg=1.0.
#
# Usage:
#   scripts/run_arch_ablation.sh <ckpt_with_mp> <ckpt_without_mp> <pred_cache> <ann_file>
#
# Results (4 JSONs + aggregated CSV):
#   results/arch-ablation/1.json  (MP=true,  raw_refiner=false)
#   results/arch-ablation/2.json  (MP=true,  raw_refiner=true)
#   results/arch-ablation/3.json  (MP=false, raw_refiner=false)
#   results/arch-ablation/4.json  (MP=false, raw_refiner=true)
#   results/arch-ablation/results.csv
#
# Example:
#   scripts/run_arch_ablation.sh \
#       results/sweeps/arch-message-passing/task_1/20260316_003813/checkpoints/epoch140-running_loss_5=0.0155.ckpt \
#       results/sweeps/arch-message-passing/task_2/20260316_032147/checkpoints/epoch116-running_loss_5=0.0350.ckpt \
#       results/cache/rtmpose-m_crowdpose_val_predictions.pkl \
#       data/crowdpose/annotations/crowdpose_val.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <ckpt_with_mp> <ckpt_without_mp> <pred_cache> <ann_file>" >&2
    exit 1
fi

CKPT_MP="$1"
CKPT_NOMP="$2"
PRED_CACHE="$3"
ANN_FILE="$4"

OUT_DIR="${PROJECT_ROOT}/results/arch-ablation"
BASELINE_ARGS=(--steps 1 --noise-frac 0.2 --passes 1 --cfg-guidance-scale 1.0)

cd "$PROJECT_ROOT"

run_task() {
    local task_id="$1"
    local checkpoint="$2"
    shift 2
    local extra_args=("$@")

    echo "================================================================"
    echo "  Task ${task_id}"
    echo "  Checkpoint : ${checkpoint}"
    [[ ${#extra_args[@]} -gt 0 ]] && echo "  Extra      : ${extra_args[*]}"
    echo "================================================================"

    python scripts/benchmark_diffusion_refiner.py \
        --checkpoint "$checkpoint" \
        --pred-cache  "$PRED_CACHE" \
        --ann-file    "$ANN_FILE" \
        --task-id     "$task_id" \
        --out-dir     "$OUT_DIR" \
        "${BASELINE_ARGS[@]}" \
        "${extra_args[@]}"
    echo ""
}

# Task 1: MP=true,  raw_refiner=false
run_task 1 "$CKPT_MP"
# Task 2: MP=true,  raw_refiner=true
run_task 2 "$CKPT_MP" --raw-refiner
# Task 3: MP=false, raw_refiner=false
run_task 3 "$CKPT_NOMP"
# Task 4: MP=false, raw_refiner=true
run_task 4 "$CKPT_NOMP" --raw-refiner

echo "================================================================"
echo "  All 4 tasks complete. Aggregating results..."
echo "================================================================"

# Re-use the aggregation snippet from run_benchmark_sweep.sh
"${SCRIPT_DIR}/run_benchmark_sweep.sh" \
    "${PROJECT_ROOT}/sweeps/benchmark_refiner.sweep" \
    --aggregate "$OUT_DIR"

echo ""
echo "  Results : ${OUT_DIR}/results.csv"
echo "================================================================"
