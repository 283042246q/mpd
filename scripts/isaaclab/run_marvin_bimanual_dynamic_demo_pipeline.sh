#!/usr/bin/env bash
# One command: resident MPD + Marvin fake hardware + moving world + Isaac Lab video.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPD_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AIRUNTIME_ROOT="${AIRUNTIME_ROOT:-/home/eric/Projects/physical_ai_runtime}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/eric/IsaacLab}"
MPD_PYTHON="${MPD_PYTHON:-/home/eric/anaconda3/envs/mpd-splines-public/bin/python}"
CONDA_EXECUTABLE="${CONDA_EXECUTABLE:-/home/eric/anaconda3/bin/conda}"
ISAAC_PYTHON_PREFIX="${ISAAC_PYTHON_PREFIX:-/home/eric/anaconda3/envs/env_isaaclab}"

TASK_MODE="independent"
RUNTIME_MODE="space-time"
EXECUTE=false
DEVICE="cuda:0"
WORLD_SCENARIO="warehouse_core_crossing"
PLANNING_BUDGET_S=60
MAX_REPLANS=5
MAX_HANDOFFS=5
RUN_TIMEOUT_S=420
OUTPUT_DIR=""
ROS_DOMAIN_ID_DEMO=""
VIDEO_FPS=24
WIDTH=1280
HEIGHT=720
SKIP_BUILD=false
SKIP_RENDER=false

usage() {
  printf '%s\n' \
    "Usage: $0 [options]" \
    "  --task-mode MODE       independent or cooperative (default: independent)" \
    "  --runtime-mode MODE    space-time or fixed-time (default: space-time)" \
    "  --execute              Execute atomically through combined 14-DoF JTC" \
    "  --device DEVICE        MPD device (default: cuda:0)" \
    "  --world-scenario NAME  warehouse_core_crossing (default), crossing_three alias, or safe_three" \
    "  --planning-budget S    Planning deadline budget (default: 60)" \
    "  --max-replans N        Maximum latest-only planning attempts (default: 5)" \
    "  --max-handoffs N       Maximum atomic execution commits (default: 5)" \
    "  --timeout-sec S        Whole ROS action timeout (default: 420)" \
    "  --output-dir PATH      Artifact directory (default: timestamped)" \
    "  --ros-domain-id ID     Isolated ROS domain (default: auto)" \
    "  --video-fps FPS        Isaac Lab output FPS (default: 24)" \
    "  --width PX             Video width (default: 1280)" \
    "  --height PX            Video height (default: 720)" \
    "  --skip-build           Reuse the installed ROS overlay" \
    "  --skip-render          Stop after validating the replay artifact" \
    "  -h, --help             Show this help"
}

while (($#)); do
  case "$1" in
    --task-mode) TASK_MODE="$2"; shift 2 ;;
    --runtime-mode) RUNTIME_MODE="$2"; shift 2 ;;
    --execute) EXECUTE=true; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --world-scenario) WORLD_SCENARIO="$2"; shift 2 ;;
    --planning-budget) PLANNING_BUDGET_S="$2"; shift 2 ;;
    --max-replans) MAX_REPLANS="$2"; shift 2 ;;
    --max-handoffs) MAX_HANDOFFS="$2"; shift 2 ;;
    --timeout-sec) RUN_TIMEOUT_S="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --ros-domain-id) ROS_DOMAIN_ID_DEMO="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --skip-render) SKIP_RENDER=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$TASK_MODE" in
  independent|dual_independent)
    TASK_MODE="independent"
    MPD_CONFIG="${MPD_ROOT}/scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml"
    ;;
  cooperative|cooperative_rigid)
    TASK_MODE="cooperative"
    MPD_CONFIG="${MPD_ROOT}/scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-cooperative-independent-prior.yaml"
    ;;
  *) printf 'Unsupported task mode: %s\n' "$TASK_MODE" >&2; exit 2 ;;
esac

case "$RUNTIME_MODE" in
  space-time|space_time)
    RUNTIME_MODE="space-time"
    WORKER_SCRIPT="${MPD_ROOT}/scripts/runtime/infer_space_time_server_marvin_bimanual.py"
    ROS_RUNTIME_MODE="inference_time_optimized"
    SOCKET_NAME="mpd-marvin-space-time.sock"
    WORKER_EXTRA_ARGS=()
    ;;
  fixed-time|fixed_time)
    RUNTIME_MODE="fixed-time"
    WORKER_SCRIPT="${MPD_ROOT}/scripts/runtime/infer_dynamic_server_marvin_bimanual.py"
    ROS_RUNTIME_MODE="fixed_time_dynamic"
    SOCKET_NAME="mpd-marvin-fixed-time.sock"
    WORKER_EXTRA_ARGS=(--runtime-mode fixed_time_dynamic)
    ;;
  *) printf 'Unsupported runtime mode: %s\n' "$RUNTIME_MODE" >&2; exit 2 ;;
esac

case "$WORLD_SCENARIO" in
  safe_three|crossing_three|warehouse_core_crossing) ;;
  *) printf 'Unsupported world scenario: %s\n' "$WORLD_SCENARIO" >&2; exit 2 ;;
esac

if [[ ! "$RUN_TIMEOUT_S" =~ ^[0-9]+$ ]] || ((RUN_TIMEOUT_S < 1)); then
  printf 'Invalid --timeout-sec: %s\n' "$RUN_TIMEOUT_S" >&2
  exit 2
fi
if [[ ! "$MAX_REPLANS" =~ ^[1-9][0-9]*$ ]] || [[ ! "$MAX_HANDOFFS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'max-replans and max-handoffs must be positive integers\n' >&2
  exit 2
fi
if [[ -z "$ROS_DOMAIN_ID_DEMO" ]]; then
  ROS_DOMAIN_ID_DEMO=$((100 + ($$ % 100)))
fi
if [[ ! "$ROS_DOMAIN_ID_DEMO" =~ ^[0-9]+$ ]] || ((10#$ROS_DOMAIN_ID_DEMO > 232)); then
  printf 'Invalid ROS domain ID: %s\n' "$ROS_DOMAIN_ID_DEMO" >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${MPD_ROOT}/scripts/inference/logs/marvin-dynamic-demo/$(date +%Y%m%d-%H%M%S)"
fi
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
PLANNER_RESULTS="${OUTPUT_DIR}/planner-results"
RECORD_ROOT="${OUTPUT_DIR}/replay"
DEMO_RESULT="${OUTPUT_DIR}/demo-result.json"
TIMELINE="${OUTPUT_DIR}/dynamic-timeline.json"
VIDEO_PATH="${OUTPUT_DIR}/marvin-${TASK_MODE}-${RUNTIME_MODE}.mp4"
SCREENSHOT_PATH="${OUTPUT_DIR}/marvin-${TASK_MODE}-${RUNTIME_MODE}.png"
SUMMARY_PATH="${OUTPUT_DIR}/isaac-replay-summary.json"

for required in "$MPD_PYTHON" "$CONDA_EXECUTABLE" "$MPD_CONFIG" "$WORKER_SCRIPT"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path does not exist: %s\n' "$required" >&2
    exit 1
  fi
done
if [[ "$SKIP_RENDER" != true && ! -e "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
  printf 'Isaac Lab launcher does not exist: %s\n' "${ISAACLAB_ROOT}/isaaclab.sh" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$PLANNER_RESULTS" "$RECORD_ROOT" "${OUTPUT_DIR}/ros-home"
SOCKET_DIR="$(mktemp -d "${XDG_RUNTIME_DIR:-/tmp}/marvin-dynamic.XXXXXX")"
SOCKET_PATH="${SOCKET_DIR}/${SOCKET_NAME}"
SERVER_PID=""
ROS_PID=""

cleanup() {
  if [[ -n "$ROS_PID" ]] && kill -0 "$ROS_PID" 2>/dev/null; then
    kill -INT -- "-$ROS_PID" 2>/dev/null || kill -INT "$ROS_PID" 2>/dev/null || true
    wait "$ROS_PID" 2>/dev/null || true
  fi
  if [[ -n "$SOCKET_PATH" && -S "$SOCKET_PATH" ]]; then
    env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" \
      "${MPD_ROOT}/scripts/runtime/infer_dynamic_client.py" \
      --socket "$SOCKET_PATH" --timeout-sec 5 shutdown >/dev/null 2>&1 || true
  fi
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -d "$SOCKET_DIR" ]]; then
    rmdir "$SOCKET_DIR" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

printf '[1/6] Starting resident Marvin %s worker (%s)\n' "$RUNTIME_MODE" "$TASK_MODE"
MPD_ENV_PREFIX="$(dirname "$(dirname "$MPD_PYTHON")")"
env -u PYTHONPATH LD_LIBRARY_PATH="${MPD_ENV_PREFIX}/lib" \
  "$CONDA_EXECUTABLE" run --no-capture-output -n mpd-splines-public \
  python "$WORKER_SCRIPT" \
  --socket "$SOCKET_PATH" \
  --output-root "$PLANNER_RESULTS" \
  --config "$MPD_CONFIG" \
  --device "$DEVICE" \
  --process-acceleration-std 0.0 \
  "${WORKER_EXTRA_ARGS[@]}" >"${OUTPUT_DIR}/mpd-worker.log" 2>&1 &
SERVER_PID=$!

READY=false
for _ in $(seq 1 240); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf 'MPD worker exited during startup; see %s\n' "${OUTPUT_DIR}/mpd-worker.log" >&2
    exit 1
  fi
  if HEALTH_JSON="$(env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" \
    "${MPD_ROOT}/scripts/runtime/infer_dynamic_client.py" \
    --socket "$SOCKET_PATH" --timeout-sec 10 health 2>/dev/null)"; then
    HEALTH_STATE="$(printf '%s' "$HEALTH_JSON" | \
      env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" -c \
      'import json,sys; print(json.load(sys.stdin).get("state", "UNKNOWN"))')"
    if [[ "$HEALTH_STATE" == "READY" ]]; then
      READY=true
      break
    elif [[ "$HEALTH_STATE" == "FAULT" ]]; then
      printf 'MPD worker entered FAULT; health response follows:\n%s\n' \
        "$HEALTH_JSON" >&2
      exit 1
    fi
  fi
  sleep 1
done
if [[ "$READY" != true ]]; then
  printf 'MPD worker did not become ready; see %s\n' "${OUTPUT_DIR}/mpd-worker.log" >&2
  exit 1
fi

printf '[2/6] Building and launching fake hardware, three moving objects, and action client\n'
cd "$AIRUNTIME_ROOT"
if [[ "$SKIP_BUILD" != true ]]; then
  pixi run build --packages-up-to \
    mpd_bimanual_planner_adapter marvin_mpd_bimanual_bringup \
    >"${OUTPUT_DIR}/ros-build.log" 2>&1
fi
EXECUTE_TEXT=false
if [[ "$EXECUTE" == true ]]; then EXECUTE_TEXT=true; fi
setsid pixi run env -u CYCLONEDDS_URI \
  ROS_DOMAIN_ID="$ROS_DOMAIN_ID_DEMO" \
  ROS_HOME="${OUTPUT_DIR}/ros-home" \
  bash -lc 'source install/setup.bash && exec "$@"' bash \
  ros2 launch mpd_bimanual_planner_adapter dynamic_fake_hardware_demo.launch.py \
  "execute:=${EXECUTE_TEXT}" \
  "task_mode:=${TASK_MODE}" \
  "runtime_mode:=${ROS_RUNTIME_MODE}" \
  "worker_socket:=${SOCKET_PATH}" \
  "record_root:=${RECORD_ROOT}" \
  "result_path:=${DEMO_RESULT}" \
  "planning_budget_s:=${PLANNING_BUDGET_S}" \
  "max_replans:=${MAX_REPLANS}" \
  "max_handoffs:=${MAX_HANDOFFS}" \
  "world_scenario:=${WORLD_SCENARIO}" \
  >"${OUTPUT_DIR}/ros-demo.log" 2>&1 &
ROS_PID=$!

for _ in $(seq 1 "$RUN_TIMEOUT_S"); do
  if [[ -f "$DEMO_RESULT" ]]; then break; fi
  if ! kill -0 "$ROS_PID" 2>/dev/null; then
    printf 'ROS launch exited before producing a result; see %s\n' "${OUTPUT_DIR}/ros-demo.log" >&2
    exit 1
  fi
  sleep 1
done
if [[ ! -f "$DEMO_RESULT" ]]; then
  printf 'Demo timed out after %ss; see %s\n' "$RUN_TIMEOUT_S" "${OUTPUT_DIR}/ros-demo.log" >&2
  exit 1
fi
env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("success"), d' \
  "$DEMO_RESULT"

REQUEST_ID="$(env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["request_id"])' "$DEMO_RESULT")"
REPLAY_RECORD="${RECORD_ROOT}/${REQUEST_ID}.json"
if [[ ! -f "$REPLAY_RECORD" ]]; then
  printf 'Replay record missing: %s\n' "$REPLAY_RECORD" >&2
  exit 1
fi

printf '[3/6] Stopping ROS/worker processes and validating the deterministic record\n'
cleanup
ROS_PID=""
SERVER_PID=""
cd "$MPD_ROOT"
env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" \
  scripts/isaaclab/replay_marvin_bimanual_dynamic_log.py \
  --record "$REPLAY_RECORD" --output "$TIMELINE"

readarray -t PLAN_FIELDS < <(env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" -c \
  'import json,sys; from scripts.isaaclab.replay_marvin_bimanual_dynamic_log import selected_plan; p=selected_plan(json.load(open(sys.argv[1]))); print(p["result_path"]); print(p["top_k_index"])' \
  "$REPLAY_RECORD")
ARTIFACT_PATH="${PLAN_FIELDS[0]}"
TRAJECTORY_INDEX="${PLAN_FIELDS[1]}"
if [[ ! -f "$ARTIFACT_PATH" ]]; then
  printf 'Selected MPD artifact missing: %s\n' "$ARTIFACT_PATH" >&2
  exit 1
fi

printf '[4/6] Verifying the selected Marvin artifact\n'
env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" -c \
  'from pathlib import Path; import sys; from scripts.isaaclab.marvin_bimanual_asset import load_inference_artifact; a=load_inference_artifact(Path(sys.argv[1])); print(f"artifact OK: K={a.top_k_positions.shape[0]} T={a.positions.shape[0]}")' \
  "$ARTIFACT_PATH" | tee "${OUTPUT_DIR}/artifact-validation.log"

printf '[5/6] Materializing replay inputs\n'
printf '%s\n' \
  "  action result: $DEMO_RESULT" \
  "  replay record: $REPLAY_RECORD" \
  "  artifact:      $ARTIFACT_PATH" \
  "  top-K index:   $TRAJECTORY_INDEX" \
  "  timeline:      $TIMELINE"

printf '[6/6] Rendering Isaac Lab video\n'
if [[ "$SKIP_RENDER" == true ]]; then
  printf '  skipped by --skip-render\n'
else
  env -u PYTHONPATH -u LD_LIBRARY_PATH CONDA_PREFIX="$ISAAC_PYTHON_PREFIX" \
    "${ISAACLAB_ROOT}/isaaclab.sh" -p \
    scripts/isaaclab/replay_marvin_bimanual_trajectory.py \
    --artifact "$ARTIFACT_PATH" \
    --trajectory-index "$TRAJECTORY_INDEX" \
    --dynamic-record "$REPLAY_RECORD" \
    --output-video "$VIDEO_PATH" \
    --screenshot "$SCREENSHOT_PATH" \
    --output-json "$SUMMARY_PATH" \
    --video-fps "$VIDEO_FPS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --headless >"${OUTPUT_DIR}/isaac-replay.log" 2>&1
fi

trap - EXIT INT TERM
printf '%s\n' \
  "Done." \
  "  mode:     $TASK_MODE / $RUNTIME_MODE / execute=$EXECUTE_TEXT" \
  "  output:   $OUTPUT_DIR" \
  "  record:   $REPLAY_RECORD"
if [[ "$SKIP_RENDER" != true ]]; then
  printf '%s\n' \
    "  video:    $VIDEO_PATH" \
    "  frame:    $SCREENSHOT_PATH" \
    "  summary:  $SUMMARY_PATH"
fi
