#!/usr/bin/env bash
set -euo pipefail

# X sampled families; each has six target worlds (0,1,2,3,4,5 obstacles).
NUM_FAMILIES="${NUM_FAMILIES:-200}"
DIRECTION="${DIRECTION:-random}"  # random per family, or forward/reverse for all
SEED="${SEED:-20261015}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-128}"
COLLISION_CHUNK="${COLLISION_CHUNK:-32768}"
EDGES_PER_QUERY="${EDGES_PER_QUERY:-16}"
GOAL_ROOTS="${GOAL_ROOTS:-4}"
MAX_WORLD_ATTEMPTS="${MAX_WORLD_ATTEMPTS:-12}"
SOLVE_SECONDS="${SOLVE_SECONDS:-10}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data_trajectories_spacetime/Warehouse-TorchForest-Multiworld-f${NUM_FAMILIES}-${DIRECTION}-seed${SEED}}"
# Empty uses the packaged catalog; no old dynamic dataset directories are read.
FAMILY_CATALOG="${FAMILY_CATALOG:-}"
catalog_args=()
if [[ -n "$FAMILY_CATALOG" ]]; then
  catalog_args=(--family-catalog "$FAMILY_CATALOG")
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd -- "$REPOSITORY_ROOT"

exec "$PYTHON_BIN" -m scripts.spacetime_data.generate_warehouse_multiworld \
  --output-root "$OUTPUT_ROOT" --num-families "$NUM_FAMILIES" \
  --direction "$DIRECTION" --seed "$SEED" --device "$DEVICE" \
  --query-batch-size "$QUERY_BATCH_SIZE" --max-world-attempts "$MAX_WORLD_ATTEMPTS" \
  --collision-chunk "$COLLISION_CHUNK" --edges-per-query "$EDGES_PER_QUERY" --goal-roots "$GOAL_ROOTS" \
  --solve-seconds "$SOLVE_SECONDS" --require-tau-r \
  "${catalog_args[@]}" \
  "$@"
