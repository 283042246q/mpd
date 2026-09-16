#!/usr/bin/env python3
"""Run one F3-C ToDrawer scenario per category until each succeeds.

A scenario succeeds only when the ROS log records reaching the goal and no
controlled-braking request occurred before that goal timestamp.  Failed
attempts are retained and retried with a new deterministic planner seed.  The
successful manifest is rendered in IsaacLab before advancing to the next
scenario.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if SCRIPT_REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_REPO_ROOT.as_posix())

from scripts.isaaclab.benchmark_todrawer_random import (
    CATEGORIES,
    DEFAULT_FACTORIZED_C_CHECKPOINT,
    GENERATION_REVISION,
    REPO_ROOT,
    generate_suite,
)


PIPELINE = REPO_ROOT / "scripts/isaaclab/run_dynamic_demo_pipeline.sh"
REPLAY = REPO_ROOT / "scripts/isaaclab/replay_mpd_trajectory.py"
DEFAULT_AIRUNTIME_ROOT = Path("/home/eric/Projects/physical_ai_runtime")
DEFAULT_ISAACLAB_ROOT = Path("/home/eric/IsaacLab")
DEFAULT_ISAAC_PREFIX = Path("/home/eric/anaconda3/envs/env_isaaclab")
GOAL_PATTERN = re.compile(
    r"\[(?P<time>\d+(?:\.\d+)?)\].*goal reached; holding position"
)
BRAKE_PATTERN = re.compile(
    r"\[(?P<time>\d+(?:\.\d+)?)\].*controlled braking requested"
)
MAX_PLANNER_SEED = 2_147_483_647


@dataclass(frozen=True)
class AttemptAssessment:
    success: bool
    reason: str
    pipeline_returncode: int
    manifest_available: bool
    goal_reached: bool
    goal_timestamp_s: float | None
    brake_timestamps_s: list[float]
    brakes_before_goal: list[float]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _run_streaming(command: list[str], *, cwd: Path, log_path: Path, env=None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        return int(process.wait())


def assess_attempt(attempt_dir: Path, pipeline_returncode: int) -> AttemptAssessment:
    manifest_available = (attempt_dir / "episode/replay-manifest.json").is_file()
    ros_log = attempt_dir / "ros-replan.log"
    text = ros_log.read_text(encoding="utf-8", errors="replace") if ros_log.is_file() else ""
    goal_matches = list(GOAL_PATTERN.finditer(text))
    brake_timestamps = [float(match.group("time")) for match in BRAKE_PATTERN.finditer(text)]
    goal_timestamp = float(goal_matches[0].group("time")) if goal_matches else None
    brakes_before_goal = (
        []
        if goal_timestamp is None
        else [timestamp for timestamp in brake_timestamps if timestamp <= goal_timestamp]
    )

    if pipeline_returncode != 0:
        reason = f"pipeline exited with status {pipeline_returncode}"
    elif not manifest_available:
        reason = "replay manifest missing"
    elif goal_timestamp is None:
        reason = "goal was not reached"
    elif brakes_before_goal:
        reason = "controlled braking occurred before reaching the goal"
    else:
        reason = "goal reached without an earlier brake"

    return AttemptAssessment(
        success=(
            pipeline_returncode == 0
            and manifest_available
            and goal_timestamp is not None
            and not brakes_before_goal
        ),
        reason=reason,
        pipeline_returncode=int(pipeline_returncode),
        manifest_available=manifest_available,
        goal_reached=goal_timestamp is not None,
        goal_timestamp_s=goal_timestamp,
        brake_timestamps_s=brake_timestamps,
        brakes_before_goal=brakes_before_goal,
    )


def planner_seed(suite_seed: int, scenario_index: int, attempt_index: int) -> int:
    """Return a stable seed sequence; attempt_index is zero-based."""

    return int((suite_seed + 1009 * scenario_index + 104729 * attempt_index) % MAX_PLANNER_SEED)


def build_pipeline_command(
    *,
    scenario_path: Path,
    attempt_dir: Path,
    seed: int,
    duration_sec: float,
    plan_rate_hz: float,
    checkpoint: Path,
) -> list[str]:
    return [
        PIPELINE.as_posix(),
        "--profile",
        "to_drawer",
        "--phase",
        "factorized",
        "--factorized-method",
        "f3",
        "--factorized-timing-checkpoint",
        checkpoint.as_posix(),
        "--factorized-adapt-spatial-basis",
        "--world-scenario-file",
        scenario_path.as_posix(),
        "--planner-seed",
        str(seed),
        "--duration-sec",
        str(duration_sec),
        "--plan-rate-hz",
        str(plan_rate_hz),
        "--output-dir",
        attempt_dir.as_posix(),
        "--allow-brake",
        "--skip-build",
        "--skip-render",
    ]


def build_render_command(
    *,
    isaaclab_root: Path,
    manifest: Path,
    video: Path,
    screenshot: Path,
    summary: Path,
    video_fps: float,
    width: int,
    height: int,
) -> list[str]:
    return [
        (isaaclab_root / "isaaclab.sh").as_posix(),
        "-p",
        REPLAY.as_posix(),
        "--manifest",
        manifest.as_posix(),
        "--output_video",
        video.as_posix(),
        "--screenshot_path",
        screenshot.as_posix(),
        "--output_json",
        summary.as_posix(),
        "--video_fps",
        str(video_fps),
        "--width",
        str(width),
        "--height",
        str(height),
        "--prediction_horizon_s",
        "3.0",
        "--prediction_samples",
        "10",
        "--enable_cameras",
    ]


def _next_attempt_index(run_dir: Path) -> int:
    indices = []
    for path in run_dir.glob("attempt-*"):
        try:
            indices.append(int(path.name.removeprefix("attempt-")))
        except ValueError:
            continue
    return max(indices, default=0) + 1


def _materialize_suite(output_dir: Path, suite_seed: int) -> dict[str, Any]:
    suite_path = output_dir / "suite.json"
    if suite_path.is_file():
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        if (
            suite.get("suite_seed") != suite_seed
            or suite.get("scenario_count") != len(CATEGORIES)
            or suite.get("generation_policy", {}).get("revision")
            != GENERATION_REVISION
        ):
            raise ValueError(
                "existing suite.json does not match the requested seed/category count/"
                "generation revision; "
                "choose a new output directory"
            )
        return suite

    suite = generate_suite(len(CATEGORIES), suite_seed)
    for scenario in suite["scenarios"]:
        _write_json(output_dir / "scenarios" / f"{scenario['id']}.json", scenario)
    _write_json(suite_path, suite)
    return suite


def _build_once(airuntime_root: Path, output_dir: Path) -> int:
    return _run_streaming(
        ["pixi", "run", "build", "--packages-up-to", "mpd_dynamic_planner_adapter"],
        cwd=airuntime_root,
        log_path=output_dir / "ros-build.log",
    )


def _render_until_saved(
    *,
    command: list[str],
    attempt_dir: Path,
    video: Path,
    screenshot: Path,
    isaac_prefix: Path,
    retry_delay_sec: float,
    max_render_attempts: int,
) -> None:
    render_env = os.environ.copy()
    render_env.pop("PYTHONPATH", None)
    render_env.pop("LD_LIBRARY_PATH", None)
    render_env["CONDA_PREFIX"] = isaac_prefix.as_posix()
    render_attempt = 1
    while True:
        print(f"[render] attempt={render_attempt} video={video}", flush=True)
        status = _run_streaming(
            command,
            cwd=REPO_ROOT,
            log_path=attempt_dir / f"isaac-replay-attempt-{render_attempt:03d}.log",
            env=render_env,
        )
        if status == 0 and video.is_file() and video.stat().st_size > 0 and screenshot.is_file():
            return
        if max_render_attempts and render_attempt >= max_render_attempts:
            raise RuntimeError(
                f"IsaacLab rendering did not succeed after {max_render_attempts} attempts"
            )
        render_attempt += 1
        time.sleep(retry_delay_sec)


def _parser() -> argparse.ArgumentParser:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "scripts/isaaclab/logs/todrawer-f3c-until-success" / timestamp,
    )
    parser.add_argument("--suite-seed", type=int, default=20260829)
    parser.add_argument("--duration-sec", type=float, default=35.0)
    parser.add_argument("--plan-rate-hz", type=float, default=1.0)
    parser.add_argument(
        "--factorized-c-checkpoint", type=Path, default=DEFAULT_FACTORIZED_C_CHECKPOINT
    )
    parser.add_argument("--retry-delay-sec", type=float, default=5.0)
    parser.add_argument(
        "--max-attempts-per-scenario",
        type=int,
        default=0,
        help="0 retries indefinitely (default); a positive value stops after that many attempts",
    )
    parser.add_argument(
        "--max-render-attempts",
        type=int,
        default=0,
        help="0 retries rendering indefinitely (default)",
    )
    parser.add_argument("--video-fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--airuntime-root", type=Path, default=DEFAULT_AIRUNTIME_ROOT)
    parser.add_argument("--isaaclab-root", type=Path, default=DEFAULT_ISAACLAB_ROOT)
    parser.add_argument("--isaac-prefix", type=Path, default=DEFAULT_ISAAC_PREFIX)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the frozen suite and print first-attempt commands without running them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.duration_sec <= 0 or args.plan_rate_hz <= 0 or args.retry_delay_sec < 0:
        raise SystemExit("duration, plan rate, and retry delay must be valid positive values")
    if args.max_attempts_per_scenario < 0 or args.max_render_attempts < 0:
        raise SystemExit("maximum attempt counts must be non-negative")

    output_dir = args.output_dir.expanduser().resolve()
    checkpoint = args.factorized_c_checkpoint.expanduser().resolve()
    airuntime_root = args.airuntime_root.expanduser().resolve()
    isaaclab_root = args.isaaclab_root.expanduser().resolve()
    isaac_prefix = args.isaac_prefix.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"timing checkpoint is not a file: {checkpoint}")
    if not args.dry_run and not (isaaclab_root / "isaaclab.sh").is_file():
        raise SystemExit(f"IsaacLab launcher is missing: {isaaclab_root / 'isaaclab.sh'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    suite = _materialize_suite(output_dir, args.suite_seed)
    if not args.skip_build and not args.dry_run:
        if _build_once(airuntime_root, output_dir) != 0:
            return 1

    successful = []
    for scenario_index, scenario in enumerate(suite["scenarios"]):
        category = scenario["category"]
        scenario_id = scenario["id"]
        scenario_path = output_dir / "scenarios" / f"{scenario_id}.json"
        video = output_dir / "videos" / f"{scenario_id}-{category}-f3-c.mp4"
        screenshot = output_dir / "videos" / f"{scenario_id}-{category}-f3-c.png"
        replay_summary = output_dir / "videos" / f"{scenario_id}-{category}-f3-c.json"
        success_path = output_dir / "successes" / f"{scenario_id}.json"
        if success_path.is_file() and video.is_file() and video.stat().st_size > 0:
            record = json.loads(success_path.read_text(encoding="utf-8"))
            successful.append(record)
            print(f"[skip] {scenario_id} category={category} already has a successful video")
            continue

        run_dir = output_dir / "runs" / scenario_id / "f3_c"
        attempt_number = _next_attempt_index(run_dir)
        attempts_this_invocation = 0
        while True:
            attempt_index = attempt_number - 1
            seed = planner_seed(args.suite_seed, scenario_index, attempt_index)
            attempt_dir = run_dir / f"attempt-{attempt_number:03d}"
            command = build_pipeline_command(
                scenario_path=scenario_path,
                attempt_dir=attempt_dir,
                seed=seed,
                duration_sec=args.duration_sec,
                plan_rate_hz=args.plan_rate_hz,
                checkpoint=checkpoint,
            )
            run_spec = {
                "scenario_id": scenario_id,
                "category": category,
                "attempt": attempt_number,
                "planner_seed": seed,
                "mode": "f3_c",
                "command": command,
            }
            _write_json(attempt_dir / "run-spec.json", run_spec)
            print(
                f"[run] {scenario_id} category={category} "
                f"attempt={attempt_number} seed={seed}",
                flush=True,
            )
            if args.dry_run:
                print("  " + " ".join(command))
                break

            status = _run_streaming(
                command,
                cwd=REPO_ROOT,
                log_path=attempt_dir / "pipeline.log",
            )
            assessment = assess_attempt(attempt_dir, status)
            _write_json(attempt_dir / "attempt-result.json", asdict(assessment))
            print(f"[result] success={assessment.success} reason={assessment.reason}")
            if assessment.success:
                manifest = attempt_dir / "episode/replay-manifest.json"
                render_command = build_render_command(
                    isaaclab_root=isaaclab_root,
                    manifest=manifest,
                    video=video,
                    screenshot=screenshot,
                    summary=replay_summary,
                    video_fps=args.video_fps,
                    width=args.width,
                    height=args.height,
                )
                _render_until_saved(
                    command=render_command,
                    attempt_dir=attempt_dir,
                    video=video,
                    screenshot=screenshot,
                    isaac_prefix=isaac_prefix,
                    retry_delay_sec=args.retry_delay_sec,
                    max_render_attempts=args.max_render_attempts,
                )
                record = {
                    **run_spec,
                    "attempt_dir": attempt_dir.as_posix(),
                    "manifest": manifest.as_posix(),
                    "video": video.as_posix(),
                    "screenshot": screenshot.as_posix(),
                    "replay_summary": replay_summary.as_posix(),
                    "assessment": asdict(assessment),
                }
                _write_json(success_path, record)
                successful.append(record)
                _write_json(output_dir / "summary.json", {"successful": successful})
                print(f"[success] {scenario_id} category={category} video={video}")
                break

            attempts_this_invocation += 1
            if (
                args.max_attempts_per_scenario
                and attempts_this_invocation >= args.max_attempts_per_scenario
            ):
                print(
                    f"[stop] {scenario_id} did not succeed after "
                    f"{attempts_this_invocation} attempts",
                    file=sys.stderr,
                )
                return 2
            attempt_number += 1
            time.sleep(args.retry_delay_sec)

    if args.dry_run:
        return 0
    _write_json(
        output_dir / "summary.json",
        {
            "schema": "mpd_todrawer_f3c_until_success",
            "schema_version": 1,
            "success_definition": "goal reached and no controlled brake at or before goal",
            "successful": successful,
        },
    )
    print(f"[done] {len(successful)}/{len(CATEGORIES)} categories saved under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
