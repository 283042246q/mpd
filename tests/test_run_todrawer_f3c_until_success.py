from pathlib import Path

from scripts.isaaclab.run_todrawer_f3c_until_success import (
    assess_attempt,
    build_pipeline_command,
    planner_seed,
)


def _attempt(tmp_path: Path, log: str, *, manifest: bool = True) -> Path:
    attempt = tmp_path / "attempt"
    attempt.mkdir(parents=True)
    (attempt / "ros-replan.log").write_text(log, encoding="utf-8")
    if manifest:
        episode = attempt / "episode"
        episode.mkdir()
        (episode / "replay-manifest.json").write_text("{}\n", encoding="utf-8")
    return attempt


def test_goal_before_brake_is_success(tmp_path: Path) -> None:
    attempt = _attempt(
        tmp_path,
        "[node] [100.0] dynamic MPD replanner started\n"
        "[node] [108.0] goal reached; holding position and retaining the target\n"
        "[node] [113.0] controlled braking requested: dynamic_collision\n",
    )

    result = assess_attempt(attempt, 0)

    assert result.success
    assert result.goal_timestamp_s == 108.0
    assert result.brake_timestamps_s == [113.0]
    assert result.brakes_before_goal == []


def test_brake_before_goal_is_failure(tmp_path: Path) -> None:
    attempt = _attempt(
        tmp_path,
        "[node] [104.0] controlled braking requested: stale_world\n"
        "[node] [108.0] goal reached; holding position\n",
    )

    result = assess_attempt(attempt, 0)

    assert not result.success
    assert result.brakes_before_goal == [104.0]


def test_goal_and_manifest_are_both_required(tmp_path: Path) -> None:
    no_goal = _attempt(tmp_path / "no-goal", "", manifest=True)
    no_manifest = _attempt(
        tmp_path / "no-manifest",
        "[node] [108.0] goal reached; holding position\n",
        manifest=False,
    )

    assert not assess_attempt(no_goal, 0).success
    assert not assess_attempt(no_manifest, 0).success
    assert not assess_attempt(no_goal, 1).success


def test_retry_seed_changes_deterministically() -> None:
    seeds = [planner_seed(20260829, 3, attempt) for attempt in range(3)]

    assert seeds == [20263856, 20368585, 20473314]
    assert len(set(seeds)) == 3


def test_pipeline_command_runs_f3_c_without_rendering_failed_attempts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "timing.pt"
    command = build_pipeline_command(
        scenario_path=tmp_path / "scenario.json",
        attempt_dir=tmp_path / "attempt",
        seed=123,
        duration_sec=35.0,
        plan_rate_hz=1.0,
        checkpoint=checkpoint,
    )

    assert command[command.index("--phase") + 1] == "factorized"
    assert command[command.index("--factorized-method") + 1] == "f3"
    assert command[command.index("--planner-seed") + 1] == "123"
    assert "--factorized-adapt-spatial-basis" in command
    assert "--allow-brake" in command
    assert "--skip-render" in command
