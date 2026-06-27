"""Subprocess helpers for invoking the IsaacLab trajectory evaluator.

This module must remain importable from the legacy MPD environment. Do not
import IsaacLab or Omniverse modules here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_isaaclab_evaluator_subprocess(
    trajectories_path,
    statistics_path,
    log_path,
    isaaclab_root="/home/eric/IsaacLab_ori",
    isaaclab_conda_env="env_isaaclab_ori",
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
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        isaaclab_conda_env,
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
        cmd.append("--headless")
    if make_video:
        cmd.append("--make_video")
        if video_path is not None:
            cmd.extend(["--video_path", str(video_path)])

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=isaaclab_timeout_s,
        )
        log_path.write_text(completed.stdout or "", encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        log_output = exc.stdout or ""
        if isinstance(log_output, bytes):
            log_output = log_output.decode(errors="replace")
        log_path.write_text(log_output, encoding="utf-8")
        if statistics_path.exists():
            statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
            statistics["subprocess_timeout_s"] = int(isaaclab_timeout_s)
            statistics["subprocess_log_path"] = str(log_path)
            statistics["subprocess_warning"] = "IsaacLab subprocess timed out after writing statistics."
            return statistics
        raise RuntimeError(f"IsaacLab evaluator timed out after {isaaclab_timeout_s}s. Log: {log_path}") from exc

    if completed.returncode != 0:
        if statistics_path.exists():
            statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
            statistics["subprocess_returncode"] = int(completed.returncode)
            statistics["subprocess_log_path"] = str(log_path)
            statistics["subprocess_warning"] = "IsaacLab subprocess exited non-zero after writing statistics."
            return statistics
        raise RuntimeError(f"IsaacLab evaluator failed with return code {completed.returncode}. Log: {log_path}")

    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics["subprocess_returncode"] = int(completed.returncode)
    statistics["subprocess_log_path"] = str(log_path)
    return statistics
