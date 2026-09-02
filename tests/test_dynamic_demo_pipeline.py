from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "isaaclab" / "run_dynamic_demo_pipeline.sh"


def _run(*arguments):
    return subprocess.run(
        ["bash", SCRIPT.as_posix(), *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_pipeline_help_advertises_phase5_default_and_modes():
    result = _run("--help")

    assert result.returncode == 0
    assert "--phase NAME" in result.stdout
    assert "default: phase5" in result.stdout
    assert "--timing-mode MODE" in result.stdout
    assert "default: phase5_joint" in result.stdout
    assert "--ros-domain-id ID" in result.stdout
    assert "--planner-seed N" in result.stdout
    assert "--world-scenario-file P" in result.stdout
    assert "--skip-render" in result.stdout
    assert "default: auto" in result.stdout
    assert "--phase5-hold-weight W" in result.stdout
    assert "--phase5-tail-weight W" in result.stdout
    assert "--phase5-dynamic-grad-cap N" in result.stdout


def test_pipeline_rejects_unknown_phase_before_starting_any_process():
    result = _run("--phase", "phase6")

    assert result.returncode == 2
    assert "Unsupported phase: phase6" in result.stderr


def test_pipeline_rejects_phase5_timing_mode_for_phase4():
    result = _run("--phase", "phase4", "--timing-mode", "phase5_joint")

    assert result.returncode == 2
    assert "--timing-mode is only valid with --phase phase5" in result.stderr


def test_pipeline_rejects_aligned_switches_for_plain_phase4():
    result = _run("--phase", "phase4", "--aligned-deviation", "off")

    assert result.returncode == 2
    assert "--aligned-* switches are only valid" in result.stderr


def test_pipeline_rejects_invalid_aligned_switch_value():
    result = _run(
        "--phase", "phase4_aligned", "--aligned-mpd-guidance", "maybe"
    )

    assert result.returncode == 2
    assert "expected on or off" in result.stderr


def test_pipeline_rejects_phase5_switches_for_phase4():
    result = _run("--phase", "phase4", "--phase5-hold-weight", "0.2")

    assert result.returncode == 2
    assert "--phase5-* switches are only valid" in result.stderr


def test_pipeline_rejects_invalid_phase5_switch_and_numeric_values():
    switch = _run("--phase", "phase5", "--phase5-mpd-guidance", "maybe")
    numeric = _run("--phase", "phase5", "--phase5-dynamic-grad-cap", "-1")

    assert switch.returncode == 2
    assert "expected on or off" in switch.stderr
    assert numeric.returncode == 2
    assert "non-negative numeric value" in numeric.stderr


def test_pipeline_accepts_phase4_aligned_alias():
    result = _run(
        "--phase",
        "phase4aligned",
        "--timing-mode",
        "phase5_joint",
    )

    assert result.returncode == 2
    assert "Unsupported phase" not in result.stderr
    assert "--timing-mode is only valid with --phase phase5" in result.stderr


def test_pipeline_rejects_invalid_ros_domain_before_starting_any_process():
    result = _run("--ros-domain-id", "233")

    assert result.returncode == 2
    assert "Invalid ROS domain ID: 233" in result.stderr


def test_pipeline_rejects_invalid_planner_seed_before_starting_any_process():
    result = _run("--planner-seed", "not-an-integer")

    assert result.returncode == 2
    assert "Invalid planner seed" in result.stderr


def test_pipeline_rejects_missing_world_scenario_file():
    result = _run("--world-scenario-file", "/does/not/exist.json")

    assert result.returncode == 2
    assert "scenario file is not a regular file" in result.stderr


def test_pipeline_contains_separate_phase4_and_phase5_entrypoints():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'PHASE="phase5"' in source
    assert 'SERVER_SCRIPT="${MPD_ROOT}/scripts/runtime/infer_dynamic_server.py"' in source
    assert 'SERVER_SCRIPT="${MPD_ROOT}/scripts/runtime/infer_space_time_server.py"' in source
    assert 'ROS_LAUNCH="replan_dynamic_fake_hardware.launch.py"' in source
    assert "replan_dynamic_aligned.yaml" in source
    assert 'ROS_LAUNCH="replan_space_time_fake_hardware.launch.py"' in source
    assert 'SERVER_EXTRA_ARGS+=(--timing-mode "$TIMING_MODE")' in source
    assert 'SERVER_EXTRA_ARGS+=(--aligned)' in source
    assert 'ROS_EXTRA_ARGS+=("timing_mode:=${TIMING_MODE}")' in source
    assert "materialize_phase5_ablation_config.py" in source
    assert "--no-dynamic-guidance" in source
    assert "--no-dynamic-selection" in source
    assert '"planner_seed:=${PLANNER_SEED}"' in source
    assert 'if [[ -n "$WORLD_SCENARIO_FILE" ]]; then' in source
    assert 'ROS_EXTRA_ARGS+=("world_scenario_file:=${WORLD_SCENARIO_FILE}")' in source
    assert '\n  "world_scenario_file:=${WORLD_SCENARIO_FILE}" \\\n' not in source
    assert 'if [[ "$SKIP_RENDER" == true ]]' in source


def test_pipeline_normalizes_documented_world_scenario_aliases():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'to_drawer-crossing) WORLD_SCENARIO="to_drawer_crossing"' in source
    assert (
        'to_drawer_bridge-crossing) '
        'WORLD_SCENARIO="to_drawer_bridge_crossing"' in source
    )


def test_pipeline_keeps_runtime_socket_out_of_artifact_directory():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'mktemp -d "${XDG_RUNTIME_DIR:-/tmp}/mpd-${PHASE}.XXXXXX"' in source
    assert 'SOCKET_PATH="${SOCKET_RUNTIME_DIR}/${SOCKET_BASENAME}"' in source
    assert 'SOCKET_PATH="${OUTPUT_DIR}/${SOCKET_BASENAME}"' not in source
    assert 'unlink "$SOCKET_PATH"' in source
    assert 'rmdir "$SOCKET_RUNTIME_DIR"' in source


def test_pipeline_isolates_fake_hardware_in_a_ros_domain():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'PIPELINE_ROS_DOMAIN_ID=""' in source
    assert 'PIPELINE_ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-}"' not in source
    assert 'PIPELINE_ROS_DOMAIN_ID=$((100 + ($$ % 100)))' in source
    assert 'pixi run env ROS_DOMAIN_ID="$PIPELINE_ROS_DOMAIN_ID"' in source
    assert (
        'timeout --signal=INT --kill-after=20s "${RUN_DURATION_S}s" \\\n'
        '  pixi run env ROS_DOMAIN_ID="$PIPELINE_ROS_DOMAIN_ID"' in source
    )
