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
PHASE="phase5"
TIMING_MODE="phase5_joint"
TIMING_MODE_EXPLICIT=false
OUTPUT_DIR=""
RUN_DURATION_S=35
PLAN_RATE_HZ=1.0
WORLD_SCENARIO_OVERRIDE=""
VIDEO_FPS=24
WIDTH=1280
HEIGHT=720
SKIP_BUILD=false
REQUIRE_NO_BRAKE=true

usage() {
  printf '%s\n' \
    "Usage: $0 [options]" \
    "  --profile NAME          Environment profile (currently: to_drawer)" \
    "  --phase NAME            Planner phase: phase4 or phase5 (default: phase5)" \
    "  --timing-mode MODE      Phase-5 mode (default: phase5_joint)" \
    "  --output-dir PATH       Artifact directory (default: timestamped log)" \
    "  --duration-sec N        ROS recording duration (default: 35)" \
    "  --plan-rate-hz HZ       Replan rate (default: 1.0)" \
    "  --world-scenario NAME   Override the profile's dynamic-world scenario" \
    "  --video-fps FPS         Replay frame rate (default: 24)" \
    "  --width PX              Replay width (default: 1280)" \
    "  --height PX             Replay height (default: 720)" \
    "  --allow-brake           Accept recorded safety braking and still render replay" \
    "  --skip-build            Reuse the existing ROS install tree" \
    "  -h, --help              Show this help"
}

while (($#)); do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --timing-mode) TIMING_MODE="$2"; TIMING_MODE_EXPLICIT=true; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --duration-sec) RUN_DURATION_S="$2"; shift 2 ;;
    --plan-rate-hz) PLAN_RATE_HZ="$2"; shift 2 ;;
    --world-scenario) WORLD_SCENARIO_OVERRIDE="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --allow-brake) REQUIRE_NO_BRAKE=false; shift ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PHASE" in
  4|phase4) PHASE="phase4" ;;
  5|phase5) PHASE="phase5" ;;
  *)
    printf 'Unsupported phase: %s (supported: phase4, phase5)\n' "$PHASE" >&2
    exit 2
    ;;
esac
case "$TIMING_MODE" in
  phase5_scalar_duration|phase5_timing_only|phase5_joint) ;;
  *)
    printf 'Unsupported Phase-5 timing mode: %s\n' "$TIMING_MODE" >&2
    exit 2
    ;;
esac
if [[ "$PHASE" == "phase4" && "$TIMING_MODE_EXPLICIT" == true ]]; then
  printf '%s\n' '--timing-mode is only valid with --phase phase5' >&2
  exit 2
fi

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
if [[ -n "$WORLD_SCENARIO_OVERRIDE" ]]; then
  WORLD_SCENARIO="$WORLD_SCENARIO_OVERRIDE"
fi

SERVER_EXTRA_ARGS=()
ROS_EXTRA_ARGS=()
case "$PHASE" in
  phase4)
    SERVER_SCRIPT="${MPD_ROOT}/scripts/runtime/infer_dynamic_server.py"
    ROS_LAUNCH="replan_dynamic_fake_hardware.launch.py"
    SOCKET_BASENAME="mpd-dynamic-runtime.sock"
    TIMING_LABEL="fixed"
    HEALTH_TIMEOUT_S=2
    ;;
  phase5)
    SERVER_SCRIPT="${MPD_ROOT}/scripts/runtime/infer_space_time_server.py"
    ROS_LAUNCH="replan_space_time_fake_hardware.launch.py"
    SOCKET_BASENAME="mpd-space-time-runtime.sock"
    TIMING_LABEL="$TIMING_MODE"
    HEALTH_TIMEOUT_S=10
    SERVER_EXTRA_ARGS+=(--timing-mode "$TIMING_MODE")
    ROS_EXTRA_ARGS+=("timing_mode:=${TIMING_MODE}")
    ;;
esac

if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ "$PHASE" == "phase4" ]]; then
    LOG_GROUP="dynamic-replay-${PROFILE}"
  else
    LOG_GROUP="dynamic-replay-${PROFILE}-phase5"
  fi
  OUTPUT_DIR="${MPD_ROOT}/scripts/inference/logs/${LOG_GROUP}/$(date +%Y%m%d-%H%M%S)"
fi
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
STATIC_SCENE="${OUTPUT_DIR}/static-scene.json"
RECORD_DIR="${OUTPUT_DIR}/episode"
MANIFEST="${RECORD_DIR}/replay-manifest.json"
SOCKET_PATH="${OUTPUT_DIR}/${SOCKET_BASENAME}"
PLANNER_RESULTS="${OUTPUT_DIR}/planner-results"
VIDEO_PATH="${OUTPUT_DIR}/${PROFILE}-dynamic-replay.mp4"
SCREENSHOT_PATH="${OUTPUT_DIR}/${PROFILE}-dynamic-replay-final.png"
SUMMARY_PATH="${OUTPUT_DIR}/${PROFILE}-dynamic-replay-summary.json"
TIMING_SUMMARY_PATH="${OUTPUT_DIR}/${PROFILE}-replan-timing.json"

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

printf '[1/6] Exporting static scene for %s\n' "$ENV_NAME"
cd "$MPD_ROOT"
env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" \
  scripts/isaaclab/export_replay_static_scene.py \
  --profile "$PROFILE" --output "$STATIC_SCENE"

printf '[2/6] Starting resident MPD %s worker (cold model load occurs once)\n' "$PHASE"
env -u PYTHONPATH -u LD_LIBRARY_PATH "$CONDA_EXECUTABLE" run --no-capture-output \
  -n mpd-splines-public python "$SERVER_SCRIPT" \
  --socket "$SOCKET_PATH" \
  --output-root "$PLANNER_RESULTS" \
  --config "$MPD_CONFIG" \
  --device cuda:0 \
  "${SERVER_EXTRA_ARGS[@]}" >"${OUTPUT_DIR}/mpd-server.log" 2>&1 &
SERVER_PID=$!

READY=false
for _ in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf 'MPD worker exited during startup; see %s\n' "${OUTPUT_DIR}/mpd-server.log" >&2
    exit 1
  fi
  if env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" \
    scripts/runtime/infer_dynamic_client.py \
    --socket "$SOCKET_PATH" --timeout-sec "$HEALTH_TIMEOUT_S" health >/dev/null 2>&1; then
    READY=true
    break
  fi
  sleep 1
done
if [[ "$READY" != true ]]; then
  printf 'MPD worker did not become ready in 180 seconds; see %s\n' "${OUTPUT_DIR}/mpd-server.log" >&2
  exit 1
fi

printf '[3/6] Running %s fake hardware, moving obstacle, replanning, and passive trace recording\n' "$PHASE"
cd "$AIRUNTIME_ROOT"
if [[ "$SKIP_BUILD" != true ]]; then
  pixi run build --packages-up-to mpd_dynamic_planner_adapter \
    >"${OUTPUT_DIR}/ros-build.log" 2>&1
fi
set +e
timeout --signal=INT --kill-after=20s "${RUN_DURATION_S}s" \
  pixi run bash -lc 'source install/setup.bash && exec "$@"' bash \
  ros2 launch mpd_dynamic_planner_adapter "$ROS_LAUNCH" \
  plan_only:=false \
  "plan_rate_hz:=${PLAN_RATE_HZ}" \
  "world_scenario:=${WORLD_SCENARIO}" \
  "scene_id:=${ENV_NAME}" \
  "socket_path:=${SOCKET_PATH}" \
  "replay_record_dir:=${RECORD_DIR}" \
  "replay_env_name:=${ENV_NAME}" \
  "replay_static_scene_json:=${STATIC_SCENE}" \
  "target_pose_xyzw:=${TARGET_POSE}" \
  "${ROS_EXTRA_ARGS[@]}" \
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

printf '[4/6] Validating recorded manifest\n'
cd "$MPD_ROOT"
env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" -c \
  'from pathlib import Path; from scripts.isaaclab.dynamic_replay_timeline import load_dynamic_replay_manifest; import sys; episode = load_dynamic_replay_manifest(Path(sys.argv[1])); print(f"manifest OK: {len(episode.plans)} plans, {len(episode.world_snapshots)} worlds, {episode.duration_s:.3f}s")' \
  "$MANIFEST" | tee "${OUTPUT_DIR}/manifest-validation.log"

printf '[5/6] Checking handoff timing, command continuity, and brake events\n'
TIMING_ARGS=(
  --output "$TIMING_SUMMARY_PATH"
  --maximum-gap-s 0.05
)
if [[ "$REQUIRE_NO_BRAKE" == true ]]; then
  TIMING_ARGS+=(--require-no-brake)
fi
if ! env -u PYTHONPATH -u LD_LIBRARY_PATH "$MPD_PYTHON" \
  scripts/isaaclab/summarize_replan_timing.py "$MANIFEST" \
  "${TIMING_ARGS[@]}" | tee "${OUTPUT_DIR}/replan-timing.log"; then
  if [[ "$REQUIRE_NO_BRAKE" == true ]]; then
    printf '%s\n' \
      'Timing validation rejected the episode. If a safety brake is expected,' \
      'rerun with --allow-brake to retain continuity checks and render it.' >&2
  else
    printf 'Timing validation failed; see %s\n' \
      "${OUTPUT_DIR}/replan-timing.log" >&2
  fi
  exit 1
fi

printf '[6/6] Rendering deterministic IsaacLab replay\n'
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
  "  phase:    $PHASE" \
  "  timing:   $TIMING_LABEL" \
  "  manifest: $MANIFEST" \
  "  video:    $VIDEO_PATH" \
  "  frame:    $SCREENSHOT_PATH" \
  "  summary:  $SUMMARY_PATH" \
  "  metrics:  $TIMING_SUMMARY_PATH" \
  "  logs:     $OUTPUT_DIR"
