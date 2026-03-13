#!/usr/bin/env bash
# Run a single task from a sweep parameter file.
#
# Usage:
#   scripts/run_sweep.sh <sweep_file> <task_id> [extra_hydra_overrides...]
#
# Arguments:
#   sweep_file              Path to a .sweep file. Each non-blank, non-comment line is one
#                           set of Hydra parameters for a training run.
#   task_id                 1-based index selecting which line (task) to execute.
#   extra_hydra_overrides   Optional extra Hydra overrides appended after the sweep-line
#                           params (last-write wins, so these can override sweep values).
#
# The script creates results/sweeps/<sweep_name>/task_<task_id>/ as the output
# directory for the run.  Training and final test evaluation are handled by
# scripts/train_pose_refinement.py; no separate test step is needed.
#
# Example (run task 3 from sweeps/my_sweep.sweep):
#   scripts/run_sweep.sh sweeps/my_sweep.sweep 3
#
# To run all tasks in a sweep file sequentially:
#   for i in $(seq 1 $(grep -cve '^\s*#\|^\s*$' sweeps/my_sweep.sweep)); do
#       scripts/run_sweep.sh sweeps/my_sweep.sweep $i
#   done

set -euo pipefail

RUN_DATETIME="$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <sweep_file> <task_id> [extra_hydra_overrides...]" >&2
    exit 1
fi

SWEEP_FILE="$1"
TASK_ID="$2"
EXTRA_ARGS=("${@:3}")

if [[ ! -f "$SWEEP_FILE" ]]; then
    echo "Error: sweep file not found: ${SWEEP_FILE}" >&2
    exit 1
fi

if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || [[ "$TASK_ID" -lt 1 ]]; then
    echo "Error: task_id must be a positive integer, got: ${TASK_ID}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Read parameters for the requested task
# (blank lines and lines starting with # are skipped; task_id is 1-based)
# ---------------------------------------------------------------------------
PARAMS=$(grep -v '^\s*#' "$SWEEP_FILE" | grep -v '^\s*$' | sed -n "${TASK_ID}p")

if [[ -z "$PARAMS" ]]; then
    TOTAL=$(grep -v '^\s*#' "$SWEEP_FILE" | grep -cv '^\s*$' || true)
    echo "Error: task_id=${TASK_ID} is out of range (sweep has ${TOTAL} tasks)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve paths and output directory
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SWEEP_NAME="$(basename "$SWEEP_FILE" .sweep)"
SWEEP_DIR="${PROJECT_ROOT}/results/sweeps/${SWEEP_NAME}/${RUN_DATETIME}/task_${TASK_ID}"

mkdir -p "$SWEEP_DIR"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "================================================================"
echo "  Sweep : ${SWEEP_NAME}"  echo "  Run   : ${RUN_DATETIME}"echo "  Task  : ${TASK_ID}"
echo "  Params: ${PARAMS}"
echo "  Extra : ${EXTRA_ARGS[*]}"
echo "  OutDir: ${SWEEP_DIR}"
echo "================================================================"
echo ""

cd "$PROJECT_ROOT"

# Split $PARAMS into an array so that quoted sub-strings are handled correctly.
read -ra PARAM_ARRAY <<< "$PARAMS"

python scripts/train_pose_refinement.py \
    "hydra.run.dir=${SWEEP_DIR}" \
    "${PARAM_ARRAY[@]}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "================================================================"
echo "  Sweep task ${TASK_ID} complete."
echo "  Results: ${SWEEP_DIR}"
echo "================================================================"
