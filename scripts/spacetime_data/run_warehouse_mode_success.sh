#!/usr/bin/env bash
set -euo pipefail

# Reference row, NOT task_id. Doubled source advances by two rows per attempt.
START_INDEX="${START_INDEX:-0}"
# Dynamic-world sampling limit WITHIN one attempt; no planner retries.
MAX_WORLD_ATTEMPTS="${MAX_WORLD_ATTEMPTS:-12}"
# Total reference attempts for EACH mode/count cell (20 cells in this suite).
ATTEMPTS_PER_CASE="${ATTEMPTS_PER_CASE:-100}"

SEED="${SEED:-20261013}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-8}"
SOLVE_SECONDS="${SOLVE_SECONDS:-5}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# New directory avoids mixing upgraded generator code with the paused run.
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_results/warehouse_mode_success/torch-forest-a${ATTEMPTS_PER_CASE}-start${START_INDEX}-worldtries${MAX_WORLD_ATTEMPTS}-seed${SEED}}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd -- "$REPOSITORY_ROOT"

exec "$PYTHON_BIN" -m scripts.spacetime_data.benchmark_warehouse_mode_success \
  --output-root "$OUTPUT_ROOT" \
  --start-index "$START_INDEX" \
  --max-world-attempts "$MAX_WORLD_ATTEMPTS" \
  --attempts "$ATTEMPTS_PER_CASE" \
  --seed "$SEED" --device "$DEVICE" \
  --query-batch-size "$QUERY_BATCH_SIZE" --solve-seconds "$SOLVE_SECONDS" \
  "$@"
