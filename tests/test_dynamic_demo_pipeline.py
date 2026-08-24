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


def test_pipeline_rejects_unknown_phase_before_starting_any_process():
    result = _run("--phase", "phase6")

    assert result.returncode == 2
    assert "Unsupported phase: phase6" in result.stderr


def test_pipeline_rejects_phase5_timing_mode_for_phase4():
    result = _run("--phase", "phase4", "--timing-mode", "phase5_joint")

    assert result.returncode == 2
    assert "--timing-mode is only valid with --phase phase5" in result.stderr


def test_pipeline_contains_separate_phase4_and_phase5_entrypoints():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'PHASE="phase5"' in source
    assert 'SERVER_SCRIPT="${MPD_ROOT}/scripts/runtime/infer_dynamic_server.py"' in source
    assert 'SERVER_SCRIPT="${MPD_ROOT}/scripts/runtime/infer_space_time_server.py"' in source
    assert 'ROS_LAUNCH="replan_dynamic_fake_hardware.launch.py"' in source
    assert 'ROS_LAUNCH="replan_space_time_fake_hardware.launch.py"' in source
    assert 'SERVER_EXTRA_ARGS+=(--timing-mode "$TIMING_MODE")' in source
    assert 'ROS_EXTRA_ARGS+=("timing_mode:=${TIMING_MODE}")' in source


def test_pipeline_keeps_runtime_socket_out_of_artifact_directory():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'mktemp -d "${XDG_RUNTIME_DIR:-/tmp}/mpd-${PHASE}.XXXXXX"' in source
    assert 'SOCKET_PATH="${SOCKET_RUNTIME_DIR}/${SOCKET_BASENAME}"' in source
    assert 'SOCKET_PATH="${OUTPUT_DIR}/${SOCKET_BASENAME}"' not in source
    assert 'unlink "$SOCKET_PATH"' in source
    assert 'rmdir "$SOCKET_RUNTIME_DIR"' in source
