"""Subprocess helpers for invoking the IsaacLab trajectory evaluator.

This module must remain importable from the legacy MPD environment. Do not
import IsaacLab or Omniverse modules here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean_isaaclab_subprocess_env():
    env = os.environ.copy()
    for key in [
        "AR",
        "CC",
        "CFLAGS",
        "CMAKE_PREFIX_PATH",
        "CPATH",
        "CPP",
        "CPPFLAGS",
        "CUDA_HOME",
        "CUDA_PATH",
        "CXX",
        "CXXFLAGS",
        "GCC",
        "GXX",
        "LD",
        "LDFLAGS",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "NM",
        "PYTHONHOME",
        "PYTHONPATH",
        "RANLIB",
        "STRIP",
    ]:
        env.pop(key, None)
    env["TERM"] = env.get("TERM") if env.get("TERM") not in {None, "", "dumb"} else "xterm-256color"
    return env


def _find_conda_exe():
    candidates = [
        os.environ.get("CONDA_EXE"),
        Path.home() / "anaconda3" / "bin" / "conda",
        Path.home() / "miniconda3" / "bin" / "conda",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return "conda"


def _read_statistics_if_ready(statistics_path):
    if not statistics_path.exists():
        return None
    try:
        return json.loads(statistics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_log_tail(log_path, max_chars=4000):
    try:
        return Path(log_path).read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""


def _process_error_message(process_name, returncode, log_path):
    tail = _read_log_tail(log_path)
    message = f"{process_name} failed with return code {returncode}. Log: {log_path}"
    if tail:
        message = f"{message}\nLast log output:\n{tail}"
    return message


def _process_group_exists(process_group_id):
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(process_group_id, timeout_s):
    deadline = time.monotonic() + timeout_s
    while _process_group_exists(process_group_id):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _terminate_process(process, terminate_timeout_s=10, kill_timeout_s=10):
    """Stop the launcher and every IsaacLab/Kit process that it spawned."""

    process_group_id = process.pid
    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass

    if process.poll() is None:
        try:
            process.wait(timeout=terminate_timeout_s)
        except subprocess.TimeoutExpired:
            pass

    if not _wait_for_process_group_exit(process_group_id, terminate_timeout_s):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _wait_for_process_group_exit(process_group_id, kill_timeout_s)

    if process.poll() is None:
        process.wait(timeout=kill_timeout_s)
    return process.returncode


def _read_required_json(output_path, log_path, process_name, returncode):
    output_path = Path(output_path)
    log_path = Path(log_path)
    if not output_path.exists():
        tail = _read_log_tail(log_path)
        message = (
            f"{process_name} exited with return code {returncode} but did not write expected JSON: "
            f"{output_path}. Log: {log_path}"
        )
        if tail:
            message = f"{message}\nLast log output:\n{tail}"
        raise RuntimeError(message)

    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        tail = _read_log_tail(log_path)
        message = f"{process_name} wrote invalid JSON: {output_path}. Log: {log_path}"
        if tail:
            message = f"{message}\nLast log output:\n{tail}"
        raise RuntimeError(message) from exc


def run_isaaclab_evaluator_subprocess(
    trajectories_path,
    statistics_path,
    log_path,
    isaaclab_root="/home/eric/IsaacLab",
    isaaclab_conda_env="env_isaaclab",
    isaaclab_device="cuda:0",
    isaaclab_headless=True,
    isaaclab_action_repeat=4,
    isaaclab_timeout_s=900,
    make_video=False,
    video_path=None,
):
    """Run the standalone IsaacLab evaluator and return its statistics JSON."""

    trajectories_path = Path(trajectories_path)
    statistics_path = Path(statistics_path)
    log_path = Path(log_path)
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    isaaclab_sh = Path(os.path.expandvars(isaaclab_root)).expanduser() / "isaaclab.sh"
    if not isaaclab_sh.exists():
        raise FileNotFoundError(f"IsaacLab launcher not found: {isaaclab_sh}")

    evaluator_script = REPO_ROOT / "scripts" / "isaaclab" / "evaluate_mpd_trajectories.py"
    evaluator_cmd = [
        str(isaaclab_sh),
        "-p",
        str(evaluator_script),
        "--input",
        str(trajectories_path),
        "--output",
        str(statistics_path),
        "--device",
        isaaclab_device,
        "--action_repeat",
        str(isaaclab_action_repeat),
    ]
    if isaaclab_headless:
        evaluator_cmd.append("--headless")
    if make_video:
        evaluator_cmd.append("--make_video")
        if video_path is not None:
            evaluator_cmd.extend(["--video_path", str(video_path)])

    evaluator_cmd_str = " ".join(shlex.quote(part) for part in evaluator_cmd)
    conda_exe = shlex.quote(_find_conda_exe())
    shell_cmd = (
        f'eval "$({conda_exe} shell.bash hook)" && '
        f"conda activate {shlex.quote(isaaclab_conda_env)} && "
        f"exec {evaluator_cmd_str}"
    )
    cmd = ["bash", "-lc", shell_cmd]

    start_time = time.monotonic()
    statistics_ready_time = None
    completed_returncode = None
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_isaaclab_subprocess_env(),
            start_new_session=True,
        )
        try:
            while True:
                completed_returncode = process.poll()
                if completed_returncode is not None:
                    break

                statistics = _read_statistics_if_ready(statistics_path)
                if statistics is not None:
                    if statistics_ready_time is None:
                        statistics_ready_time = time.monotonic()
                    elif time.monotonic() - statistics_ready_time > 15:
                        completed_returncode = _terminate_process(process)
                        statistics["subprocess_returncode"] = int(completed_returncode)
                        statistics["subprocess_log_path"] = str(log_path)
                        statistics["subprocess_warning"] = (
                            "IsaacLab subprocess was terminated after writing statistics."
                        )
                        return statistics

                if time.monotonic() - start_time > isaaclab_timeout_s:
                    completed_returncode = _terminate_process(process)
                    statistics = _read_statistics_if_ready(statistics_path)
                    if statistics is not None:
                        statistics["subprocess_timeout_s"] = int(isaaclab_timeout_s)
                        statistics["subprocess_returncode"] = int(completed_returncode)
                        statistics["subprocess_log_path"] = str(log_path)
                        statistics["subprocess_warning"] = "IsaacLab subprocess timed out after writing statistics."
                        return statistics
                    raise RuntimeError(f"IsaacLab evaluator timed out after {isaaclab_timeout_s}s. Log: {log_path}")

                time.sleep(1)
        finally:
            _terminate_process(process)

    if completed_returncode != 0:
        statistics = _read_statistics_if_ready(statistics_path)
        if statistics is not None:
            statistics["subprocess_returncode"] = int(completed_returncode)
            statistics["subprocess_log_path"] = str(log_path)
            statistics["subprocess_warning"] = "IsaacLab subprocess exited non-zero after writing statistics."
            return statistics
        raise RuntimeError(_process_error_message("IsaacLab evaluator", completed_returncode, log_path))

    statistics = _read_required_json(
        output_path=statistics_path,
        log_path=log_path,
        process_name="IsaacLab evaluator",
        returncode=completed_returncode,
    )
    statistics["subprocess_returncode"] = int(completed_returncode)
    statistics["subprocess_log_path"] = str(log_path)
    return statistics


def run_isaaclab_replay_subprocess(
    trajectories_path,
    log_path,
    video_path=None,
    screenshot_path=None,
    output_json_path=None,
    trajectory_source="best",
    trajectory_index=0,
    isaaclab_root="/home/eric/IsaacLab",
    isaaclab_conda_env="env_isaaclab",
    isaaclab_device="cuda:0",
    isaaclab_headless=True,
    isaaclab_action_repeat=4,
    isaaclab_timeout_s=900,
    video_fps=24.0,
    width=960,
    height=540,
):
    """Run the standalone IsaacLab replay script and return its metadata JSON."""

    trajectories_path = Path(trajectories_path)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if video_path is None and screenshot_path is None:
        raise ValueError("At least one of video_path or screenshot_path must be provided.")

    if output_json_path is None:
        output_json_path = log_path.with_suffix(".json")

    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    isaaclab_sh = Path(os.path.expandvars(isaaclab_root)).expanduser() / "isaaclab.sh"
    if not isaaclab_sh.exists():
        raise FileNotFoundError(f"IsaacLab launcher not found: {isaaclab_sh}")

    replay_script = REPO_ROOT / "scripts" / "isaaclab" / "replay_mpd_trajectory.py"
    replay_cmd = [
        str(isaaclab_sh),
        "-p",
        str(replay_script),
        "--input",
        str(trajectories_path),
        "--trajectory_source",
        str(trajectory_source),
        "--trajectory_index",
        str(trajectory_index),
        "--output_json",
        str(output_json_path),
        "--device",
        isaaclab_device,
        "--action_repeat",
        str(isaaclab_action_repeat),
        "--video_fps",
        str(video_fps),
        "--width",
        str(width),
        "--height",
        str(height),
        "--enable_cameras",
    ]
    if video_path is not None:
        replay_cmd.extend(["--output_video", str(video_path)])
    if screenshot_path is not None:
        replay_cmd.extend(["--screenshot_path", str(screenshot_path)])
    if isaaclab_headless:
        replay_cmd.append("--headless")

    replay_cmd_str = " ".join(shlex.quote(part) for part in replay_cmd)
    conda_exe = shlex.quote(_find_conda_exe())
    shell_cmd = (
        f'eval "$({conda_exe} shell.bash hook)" && '
        f"conda activate {shlex.quote(isaaclab_conda_env)} && "
        f"exec {replay_cmd_str}"
    )
    cmd = ["bash", "-lc", shell_cmd]

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_isaaclab_subprocess_env(),
            start_new_session=True,
        )
        try:
            completed_returncode = process.wait(timeout=isaaclab_timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(_process_error_message("IsaacLab replay timed out", "timeout", log_path)) from exc
        finally:
            _terminate_process(process)

    if completed_returncode != 0:
        raise RuntimeError(_process_error_message("IsaacLab replay", completed_returncode, log_path))

    replay_metadata = _read_required_json(
        output_path=output_json_path,
        log_path=log_path,
        process_name="IsaacLab replay",
        returncode=completed_returncode,
    )
    replay_metadata["subprocess_returncode"] = int(completed_returncode)
    replay_metadata["subprocess_log_path"] = str(log_path)
    return replay_metadata
