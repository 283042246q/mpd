#!/usr/bin/env bash
# Run a selectable MPD dynamic-world demo and render its exact command trace.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPD_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AIRUNTIME_ROOT="${AIRUNTIME_ROOT:-/home/eric/Projects/physical_ai_runtime}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/eric/IsaacLab}"
MPD_PYTHON="${MPD_PYTHON:-/home/eric/anaconda3/envs/mpd-splines-public/bin/python}"
CONDA_EXECUTABLE="${CONDA_EXECUTABLE:-/home/eric/anaconda3/bin/conda}"
ISAAC_PYTHON_PREFIX="${ISAAC_PYTHON_PREFIX:-/home/eric/anaconda3/envs/env_isaaclab}"

PROFILE="to_drawer"
OUTPUT_DIR=""
RUN_DURATION_S=35
PLAN_RATE_HZ=0.5
VIDEO_FPS=24
WIDTH=1280
HEIGHT=720
SKIP_BUILD=false

usage() {
  printf '%s\n' \
    "Usage: $0 [options]" \
    "  --profile NAME          Environment profile (currently: to_drawer)" \
    "  --output-dir PATH       Artifact directory (default: timestamped log)" \
    "  --duration-sec N        ROS recording duration (default: 35)" \
    "  --plan-rate-hz HZ       Replan rate (default: 0.5)" \
    "  --video-fps FPS         Replay frame rate (default: 24)" \
    "  --width PX              Replay width (default: 1280)" \
    "  --height PX             Replay height (default: 720)" \
    "  --skip-build            Reuse the existing ROS install tree" \
    "  -h, --help              Show this help"
}

while (($#)); do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --duration-sec) RUN_DURATION_S="$2"; shift 2 ;;
    --plan-rate-hz) PLAN_RATE_HZ="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  to_drawer)
    ENV_NAME="EnvOpenDrawerShelf"
    WORLD_SCENARIO="to_drawer_crossing"
    MPD_CONFIG="${MPD_ROOT}/scripts/inference/cfgs/config_EnvOpenDrawerShelf-RobotPanda-runtime-to-drawer.yaml"
    TARGET_POSE="-0.2301621,0.5245667,0.3053491,-0.5024075,0.4807918,0.5121810,0.5040799"
    ;;
  *)
    printf 'Unsupported profile: %s (supported: to_drawer)\n' "$PROFILE" >&2
    exit 2
    ;;
esac

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${MPD_ROOT}/scripts/inference/logs/dynamic-replay-${PROFILE}/$(date +%Y%m%d-%H%M%S)"
fi
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
STATIC_SCENE="${OUTPUT_DIR}/static-scene.json"
RECORD_DIR="${OUTPUT_DIR}/episode"
MANIFEST="${RECORD_DIR}/replay-manifest.json"
SOCKET_PATH="${OUTPUT_DIR}/mpd-dynamic-runtime.sock"
PLANNER_RESULTS="${OUTPUT_DIR}/planner-results"
VIDEO_PATH="${OUTPUT_DIR}/${PROFILE}-dynamic-replay.mp4"
SCREENSHOT_PATH="${OUTPUT_DIR}/${PROFILE}-dynamic-replay-final.png"
SUMMARY_PATH="${OUTPUT_DIR}/${PROFILE}-dynamic-replay-summary.json"

for required in "$MPD_PYTHON" "$CONDA_EXECUTABLE" "${ISAACLAB_ROOT}/isaaclab.sh" "$MPD_CONFIG"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path does not exist: %s\n' "$required" >&2
    exit 1
  fi
done
mkdir -p "$OUTPUT_DIR" "$PLANNER_RESULTS" "$RECORD_DIR"
export ROS_HOME="${OUTPUT_DIR}/ros-home"
mkdir -p "$ROS_HOME"

SERVER_PID=""
cleanup() {
  if [[ -S "$SOCKET_PATH" ]]; then
    env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" \
      "${MPD_ROOT}/scripts/runtime/infer_dynamic_client.py" \
      --socket "$SOCKET_PATH" --timeout-sec 5 shutdown >/dev/null 2>&1 || true
  fi
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

printf '[1/5] Exporting static scene for %s\n' "$ENV_NAME"
cd "$MPD_ROOT"
env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" \
  scripts/isaaclab/export_replay_static_scene.py \
  --profile "$PROFILE" --output "$STATIC_SCENE"

printf '[2/5] Starting resident MPD worker (cold model load occurs once)\n'
env -u PYTHONPATH -u LD_LIBRARY_PATH "$CONDA_EXECUTABLE" run --no-capture-output \
  -n mpd-splines-public python scripts/runtime/infer_dynamic_server.py \
  --socket "$SOCKET_PATH" \
  --output-root "$PLANNER_RESULTS" \
  --config "$MPD_CONFIG" \
  --device cuda:0 >"${OUTPUT_DIR}/mpd-server.log" 2>&1 &
SERVER_PID=$!

READY=false
for _ in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf 'MPD worker exited during startup; see %s\n' "${OUTPUT_DIR}/mpd-server.log" >&2
    exit 1
  fi
  if env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" \
    scripts/runtime/infer_dynamic_client.py \
    --socket "$SOCKET_PATH" --timeout-sec 2 health >/dev/null 2>&1; then
    READY=true
    break
  fi
  sleep 1
done
if [[ "$READY" != true ]]; then
  printf 'MPD worker did not become ready in 180 seconds; see %s\n' "${OUTPUT_DIR}/mpd-server.log" >&2
  exit 1
fi

printf '[3/5] Running fake hardware, moving obstacle, replanning, and passive trace recording\n'
cd "$AIRUNTIME_ROOT"
if [[ "$SKIP_BUILD" != true ]]; then
  pixi run build --packages-up-to mpd_dynamic_planner_adapter \
    >"${OUTPUT_DIR}/ros-build.log" 2>&1
fi
set +e
timeout --signal=INT --kill-after=20s "${RUN_DURATION_S}s" \
  pixi run bash -lc "source install/setup.bash && exec ros2 launch mpd_dynamic_planner_adapter replan_dynamic_fake_hardware.launch.py plan_only:=false plan_rate_hz:=${PLAN_RATE_HZ} world_scenario:=${WORLD_SCENARIO} scene_id:=${ENV_NAME} socket_path:=${SOCKET_PATH} replay_record_dir:=${RECORD_DIR} replay_env_name:=${ENV_NAME} replay_static_scene_json:=${STATIC_SCENE} target_pose_xyzw:=${TARGET_POSE}" \
  >"${OUTPUT_DIR}/ros-replan.log" 2>&1
ROS_STATUS=$?
set -e
if [[ "$ROS_STATUS" -ne 0 && "$ROS_STATUS" -ne 124 ]]; then
  printf 'ROS demo failed with status %s; see %s\n' "$ROS_STATUS" "${OUTPUT_DIR}/ros-replan.log" >&2
  exit "$ROS_STATUS"
fi

cleanup
SERVER_PID=""
if [[ ! -f "$MANIFEST" ]]; then
  printf 'No replay manifest was produced; inspect %s and %s\n' \
    "${OUTPUT_DIR}/ros-replan.log" "${OUTPUT_DIR}/mpd-server.log" >&2
  exit 1
fi

printf '[4/5] Validating recorded manifest\n'
cd "$MPD_ROOT"
env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" -c \
  'from pathlib import Path; from scripts.isaaclab.dynamic_replay_timeline import load_dynamic_replay_manifest; import sys; episode = load_dynamic_replay_manifest(Path(sys.argv[1])); print(f"manifest OK: {len(episode.plans)} plans, {len(episode.world_snapshots)} worlds, {episode.duration_s:.3f}s")' \
  "$MANIFEST" | tee "${OUTPUT_DIR}/manifest-validation.log"

printf '[5/5] Rendering deterministic IsaacLab replay\n'
env -u PYTHONPATH -u LD_LIBRARY_PATH CONDA_PREFIX="$ISAAC_PYTHON_PREFIX" \
  "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/isaaclab/replay_mpd_trajectory.py \
  --manifest "$MANIFEST" \
  --output_video "$VIDEO_PATH" \
  --screenshot_path "$SCREENSHOT_PATH" \
  --output_json "$SUMMARY_PATH" \
  --video_fps "$VIDEO_FPS" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --prediction_horizon_s 3.0 \
  --prediction_samples 10 \
  --enable_cameras >"${OUTPUT_DIR}/isaac-replay.log" 2>&1

trap - EXIT INT TERM
printf '%s\n' \
  "Done." \
  "  manifest: $MANIFEST" \
  "  video:    $VIDEO_PATH" \
  "  frame:    $SCREENSHOT_PATH" \
  "  summary:  $SUMMARY_PATH" \
  "  logs:     $OUTPUT_DIR"
