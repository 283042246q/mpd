"""Launch Marvin-only Isaac Lab tools from the legacy MPD environment.

This module intentionally imports neither Torch nor Isaac Lab.  The MPD process
finishes writing its portable artifact before a separate Conda process owns the
Isaac/Kit runtime and GPU resources.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]


def _conda_executable() -> str:
    candidates = (
        os.environ.get("CONDA_EXE"),
        Path.home() / "anaconda3/bin/conda",
        Path.home() / "miniconda3/bin/conda",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return "conda"


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in (
        "AR", "CC", "CFLAGS", "CMAKE_PREFIX_PATH", "CPATH", "CPP", "CPPFLAGS",
        "CUDA_HOME", "CUDA_PATH", "CXX", "CXXFLAGS", "GCC", "GXX", "LD",
        "LDFLAGS", "LD_LIBRARY_PATH", "LIBRARY_PATH", "NM", "PYTHONHOME",
        "PYTHONPATH", "RANLIB", "STRIP",
    ):
        environment.pop(key, None)
    environment["TERM"] = environment.get("TERM") or "xterm-256color"
    return environment


def _terminate_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _log_tail(path: Path, max_chars: int = 5000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""


def _run(
    tool: str,
    arguments: list[str],
    *,
    conda_env: str,
    log_path: Path,
    output_json: Path,
    timeout_s: int,
    accepted_returncodes: tuple[int, ...] = (0,),
) -> dict:
    script = REPO_ROOT / "scripts/isaaclab" / tool
    if not script.is_file():
        raise FileNotFoundError(f"Marvin Isaac Lab tool not found: {script}")
    log_path = Path(log_path).expanduser().resolve()
    output_json = Path(output_json).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _conda_executable(),
        "run",
        "--no-capture-output",
        "-n",
        str(conda_env),
        "python",
        str(script),
        *arguments,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_environment(),
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=int(timeout_s))
        except subprocess.TimeoutExpired as error:
            _terminate_group(process)
            raise RuntimeError(
                f"Marvin Isaac Lab {tool} timed out after {timeout_s}s; log={log_path}\n"
                f"{_log_tail(log_path)}"
            ) from error
        finally:
            if process.poll() is None:
                _terminate_group(process)
    if returncode not in accepted_returncodes:
        raise RuntimeError(
            f"Marvin Isaac Lab {tool} exited {returncode}; log={log_path}\n"
            f"{_log_tail(log_path)}"
        )
    if not output_json.is_file():
        raise RuntimeError(
            f"Marvin Isaac Lab {tool} did not create {output_json}; log={log_path}\n"
            f"{_log_tail(log_path)}"
        )
    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON from Marvin Isaac Lab {tool}: {output_json}") from error
    payload["subprocess_returncode"] = int(returncode)
    payload["subprocess_log_path"] = log_path.as_posix()
    return payload


def run_marvin_isaaclab_evaluator(
    artifact: Path,
    output_json: Path,
    log_path: Path,
    *,
    conda_env: str = "env_isaaclab",
    device: str = "cuda:0",
    headless: bool = True,
    action_repeat: int = 4,
    timeout_s: int = 900,
    asset_cache: Path | None = None,
) -> dict:
    cache = asset_cache or (REPO_ROOT / ".cache/isaaclab/marvin_bimanual")
    arguments = [
        "--artifact", str(Path(artifact).expanduser().resolve()),
        "--output", str(Path(output_json).expanduser().resolve()),
        "--asset-cache", str(Path(cache).expanduser().resolve()),
        "--device", str(device),
        "--action-repeat", str(int(action_repeat)),
        "--graceful-shutdown",
    ]
    arguments.extend(("--viz", "none" if headless else "kit"))
    # Exit 2 is a completed evaluation with an MPD-vs-Isaac false negative.
    return _run(
        "evaluate_marvin_bimanual_trajectories.py",
        arguments,
        conda_env=conda_env,
        log_path=log_path,
        output_json=output_json,
        timeout_s=timeout_s,
        accepted_returncodes=(0, 2),
    )


def run_marvin_isaaclab_replay(
    artifact: Path,
    output_json: Path,
    log_path: Path,
    *,
    evaluation: Path | None = None,
    video_path: Path | None = None,
    screenshot_path: Path | None = None,
    trajectory_index: int = 0,
    conda_env: str = "env_isaaclab",
    device: str = "cuda:0",
    headless: bool = True,
    action_repeat: int = 4,
    timeout_s: int = 900,
    video_fps: float = 24.0,
    width: int = 960,
    height: int = 540,
    asset_cache: Path | None = None,
) -> dict:
    cache = asset_cache or (REPO_ROOT / ".cache/isaaclab/marvin_bimanual")
    arguments = [
        "--artifact", str(Path(artifact).expanduser().resolve()),
        "--trajectory-index", str(int(trajectory_index)),
        "--output-json", str(Path(output_json).expanduser().resolve()),
        "--asset-cache", str(Path(cache).expanduser().resolve()),
        "--device", str(device),
        "--action-repeat", str(int(action_repeat)),
        "--video-fps", str(float(video_fps)),
        "--width", str(int(width)),
        "--height", str(int(height)),
        "--graceful-shutdown",
    ]
    if evaluation is not None:
        arguments.extend(("--evaluation", str(Path(evaluation).expanduser().resolve())))
    if video_path is not None:
        arguments.extend(("--output-video", str(Path(video_path).expanduser().resolve())))
    if screenshot_path is not None:
        arguments.extend(("--screenshot", str(Path(screenshot_path).expanduser().resolve())))
    arguments.extend(("--viz", "none" if headless else "kit"))
    return _run(
        "replay_marvin_bimanual_trajectory.py",
        arguments,
        conda_env=conda_env,
        log_path=log_path,
        output_json=output_json,
        timeout_s=timeout_s,
    )
