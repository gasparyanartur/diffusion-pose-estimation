#!/usr/bin/env bash
# Run tasks from a benchmark_refiner sweep file.
#
# Modes:
#   Single task:
#     scripts/run_benchmark_sweep.sh <sweep_file> <task_id> <checkpoint> <pred_cache> <ann_file> [extra_args...]
#
#   All tasks sequentially (then auto-aggregates into results.csv):
#     scripts/run_benchmark_sweep.sh <sweep_file> --all <checkpoint> <pred_cache> <ann_file> [extra_args...]
#
#   Aggregate existing per-task JSONs into a CSV:
#     scripts/run_benchmark_sweep.sh <sweep_file> --aggregate <results_dir>
#
# Results are written to:
#   results/diffusion-refiner/<run_name>/<task_id>.json
#
# Example (run task 7):
#   scripts/run_benchmark_sweep.sh sweeps/benchmark_refiner.sweep 7 \
#       results/diffusion_training/diffusion/20260313_120000/checkpoints/best.ckpt \
#       results/cache/rtmpose-m_crowdpose_val_predictions.pkl \
#       data/crowdpose/annotations/crowdpose_val.json
#
# Example (run all tasks):
#   scripts/run_benchmark_sweep.sh sweeps/benchmark_refiner.sweep --all \
#       results/diffusion_training/diffusion/20260313_120000/checkpoints/best.ckpt \
#       results/cache/rtmpose-m_crowdpose_val_predictions.pkl \
#       data/crowdpose/annotations/crowdpose_val.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Aggregate mode: merge all per-task JSONs into one CSV
# ---------------------------------------------------------------------------
if [[ "${2:-}" == "--aggregate" ]]; then
    RESULTS_DIR="${3:?Usage: $0 <sweep_file> --aggregate <results_dir>}"
    python - "$RESULTS_DIR" << 'PYEOF'
import csv, json, sys
from pathlib import Path

results_dir = Path(sys.argv[1])
json_files = sorted(results_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)

if not json_files:
    print(f"No JSON files found in {results_dir}")
    sys.exit(0)

rows = []
for jf in json_files:
    with open(jf) as fh:
        rows.append({"task_id": jf.stem, **json.load(fh)})

out_path = results_dir / "results.csv"
with open(out_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# Print aligned table
keys = list(rows[0].keys())
widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in keys}
fmt = "  ".join(f"{{:<{widths[k]}}}" for k in keys)
print(fmt.format(*keys))
print("  ".join("-" * widths[k] for k in keys))
for r in rows:
    print(fmt.format(*[str(r[k]) for k in keys]))
print(f"\nSaved to {out_path}")
PYEOF
    exit 0
fi

# ---------------------------------------------------------------------------
# All-tasks mode: run every task in the sweep file, then aggregate
# ---------------------------------------------------------------------------
if [[ "${2:-}" == "--all" ]]; then
    if [[ $# -lt 5 ]]; then
        echo "Usage: $0 <sweep_file> --all <checkpoint> <pred_cache> <ann_file> [extra_args...]" >&2
        exit 1
    fi
    SWEEP_FILE="$1"
    CHECKPOINT="$3"
    PRED_CACHE="$4"
    ANN_FILE="$5"
    EXTRA_ARGS=("${@:6}")

    if [[ ! -f "$SWEEP_FILE" ]]; then
        echo "Error: sweep file not found: ${SWEEP_FILE}" >&2
        exit 1
    fi

    TOTAL=$(grep -v '^\s*#' "$SWEEP_FILE" | grep -cv '^\s*$' || true)
    echo "Running all ${TOTAL} tasks from $(basename "$SWEEP_FILE")..."
    echo ""

    for i in $(seq 1 "$TOTAL"); do
        "$0" "$SWEEP_FILE" "$i" "$CHECKPOINT" "$PRED_CACHE" "$ANN_FILE" "${EXTRA_ARGS[@]}"
    done

    # Infer results dir from the checkpoint path (mirrors _infer_run_name in Python)
    # _infer_run_name = checkpoint.parent.parent.parent.name
    #   checkpoint -> .../checkpoints/<name>.ckpt
    #   .parent    -> .../checkpoints/
    #   .parent    -> .../<timestamp>/
    #   .parent    -> .../<run_name>/    <- this is the name we want
    RUN_NAME="$(basename "$(dirname "$(dirname "$(dirname "$CHECKPOINT")")")")"
    RESULTS_DIR="${PROJECT_ROOT}/results/diffusion-refiner/${RUN_NAME}"
    echo ""
    echo "All tasks complete. Aggregating results from ${RESULTS_DIR}..."
    "$0" "$SWEEP_FILE" --aggregate "$RESULTS_DIR"
    exit 0
fi

# ---------------------------------------------------------------------------
# Single-task mode
# ---------------------------------------------------------------------------
if [[ $# -lt 5 ]]; then
    echo "Usage: $0 <sweep_file> <task_id> <checkpoint> <pred_cache> <ann_file> [extra_args...]" >&2
    exit 1
fi

SWEEP_FILE="$1"
TASK_ID="$2"
CHECKPOINT="$3"
PRED_CACHE="$4"
ANN_FILE="$5"
EXTRA_ARGS=("${@:6}")

if [[ ! -f "$SWEEP_FILE" ]]; then
    echo "Error: sweep file not found: ${SWEEP_FILE}" >&2
    exit 1
fi

if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || [[ "$TASK_ID" -lt 1 ]]; then
    echo "Error: task_id must be a positive integer, got: ${TASK_ID}" >&2
    exit 1
fi

PARAMS=$(grep -v '^\s*#' "$SWEEP_FILE" | grep -v '^\s*$' | sed -n "${TASK_ID}p")

if [[ -z "$PARAMS" ]]; then
    TOTAL=$(grep -v '^\s*#' "$SWEEP_FILE" | grep -cv '^\s*$' || true)
    echo "Error: task_id=${TASK_ID} is out of range (sweep has ${TOTAL} tasks)." >&2
    exit 1
fi

SWEEP_NAME="$(basename "$SWEEP_FILE" .sweep)"

echo "================================================================"
echo "  Sweep      : ${SWEEP_NAME}"
echo "  Task       : ${TASK_ID}"
echo "  Params     : ${PARAMS}"
echo "  Checkpoint : ${CHECKPOINT}"
echo "================================================================"
echo ""

cd "$PROJECT_ROOT"

read -ra PARAM_ARRAY <<< "$PARAMS"

python scripts/benchmark_diffusion_refiner.py \
    --checkpoint "$CHECKPOINT" \
    --pred-cache "$PRED_CACHE" \
    --ann-file   "$ANN_FILE" \
    --task-id    "$TASK_ID" \
    "${PARAM_ARRAY[@]}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "================================================================"
echo "  Task ${TASK_ID} complete."
echo "================================================================"
