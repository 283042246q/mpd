from __future__ import annotations

from pathlib import Path

import pytest

import scripts.isaaclab.run_todrawer_f3c_until_success as runner
from scripts.isaaclab.run_todrawer_f3c_until_success import (
    SUPPORTED_MODES,
    _next_attempt_index,
    anchor_seed,
    assess_attempt,
    build_pipeline_command,
    mode_contract,
    planner_seed,
    resample_attempt_anchors,
)
from scripts.isaaclab.benchmark_todrawer_random import BASE_CROSSINGS, generate_suite
from scripts.isaaclab.todrawer_scenario_validation import trajectory_clearances


def test_next_attempt_index_is_python38_compatible_and_ignores_bad_names(tmp_path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    (run_dir / "attempt-001").mkdir()
    (run_dir / "attempt-007").mkdir()
    (run_dir / "attempt-not-a-number").mkdir()

    assert _next_attempt_index(run_dir) == 8


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


def test_retry_anchor_seed_changes_deterministically() -> None:
    seeds = [anchor_seed(20260829, 3, attempt) for attempt in range(3)]

    assert seeds == [114657124, 147109967, 179562810]
    assert len(set(seeds)) == 3


def test_attempts_resample_only_anchors_within_configured_range() -> None:
    original = generate_suite(1, 117)["scenarios"][0]
    first = resample_attempt_anchors(original, seed=anchor_seed(117, 0, 0))
    second = resample_attempt_anchors(original, seed=anchor_seed(117, 0, 1))
    nominal = {item["id"]: item["anchor"] for item in BASE_CROSSINGS}

    assert first["objects"][0]["anchor_position"] != second["objects"][0]["anchor_position"]
    assert original["objects"][0]["anchor_position"] not in (
        first["objects"][0]["anchor_position"],
        second["objects"][0]["anchor_position"],
    )
    for sampled in (first, second):
        item = sampled["objects"][0]
        assert all(
            abs(actual - expected) <= 0.015
            for actual, expected in zip(
                item["anchor_position"], nominal[item["anchor_id"]]
            )
        )
        assert trajectory_clearances(item).robot_base_m > 0.0
        assert sampled["attempt_anchor_sampling"]["half_range_m"] == 0.015
        ignored = {
            "anchor_position",
            "minimum_robot_base_clearance_m",
            "minimum_static_environment_clearance_m",
            "attempt_anchor_sample_index",
        }
        assert {key: value for key, value in item.items() if key not in ignored} == {
            key: value
            for key, value in original["objects"][0].items()
            if key not in ignored
        }


def test_dense_attempt_uses_separate_anchor_range() -> None:
    dense = generate_suite(5, 811)["scenarios"][4]
    sampled = resample_attempt_anchors(
        dense,
        seed=9,
        anchor_jitter_m=0.0,
        dense_anchor_jitter_m=0.027,
    )

    assert sampled["attempt_anchor_sampling"]["half_range_m"] == 0.027
    assert any(
        sampled_item["anchor_position"] != original_item["anchor_position"]
        for sampled_item, original_item in zip(sampled["objects"], dense["objects"])
    )


def _pipeline_command(tmp_path: Path, mode: str) -> list[str]:
    c_checkpoint = tmp_path / "c.pt"
    tau_r_checkpoint = tmp_path / "tau-r.pt"
    return build_pipeline_command(
        mode=mode,
        scenario_path=tmp_path / "scenario.json",
        attempt_dir=tmp_path / "attempt",
        seed=123,
        duration_sec=35.0,
        plan_rate_hz=1.0,
        factorized_c_checkpoint=c_checkpoint,
        factorized_tau_r_checkpoint=tau_r_checkpoint,
    )


def test_supported_modes_are_the_requested_three_baselines_and_six_factorized() -> None:
    assert SUPPORTED_MODES == (
        "phase4",
        "phase4_aligned",
        "joint",
        "f1_c",
        "f2_c",
        "f3_c",
        "f1_tau_r",
        "f2_tau_r",
        "f3_tau_r",
    )


@pytest.mark.parametrize(
    ("mode", "phase", "timing_mode"),
    (
        ("phase4", "phase4", None),
        ("phase4_aligned", "phase4_aligned", None),
        ("joint", "phase5", "phase5_joint"),
    ),
)
def test_pipeline_command_supports_baseline_modes(
    tmp_path: Path, mode: str, phase: str, timing_mode: str | None
) -> None:
    command = _pipeline_command(tmp_path, mode)

    assert command[command.index("--phase") + 1] == phase
    assert command[command.index("--planner-seed") + 1] == "123"
    assert "--factorized-method" not in command
    assert "--factorized-timing-checkpoint" not in command
    if timing_mode is None:
        assert "--timing-mode" not in command
    else:
        assert command[command.index("--timing-mode") + 1] == timing_mode


@pytest.mark.parametrize("mode", SUPPORTED_MODES[3:])
def test_pipeline_command_supports_six_learned_factorized_modes(
    tmp_path: Path, mode: str
) -> None:
    command = _pipeline_command(tmp_path, mode)
    contract = mode_contract(mode)

    assert command[command.index("--phase") + 1] == "factorized"
    assert command[command.index("--factorized-method") + 1] == contract.factorized_method
    expected_checkpoint = tmp_path / (
        "c.pt" if contract.factorized_representation == "c" else "tau-r.pt"
    )
    assert (
        command[command.index("--factorized-timing-checkpoint") + 1]
        == expected_checkpoint.as_posix()
    )
    assert "--factorized-adapt-spatial-basis" in command
    assert "--allow-brake" in command
    assert "--skip-render" in command


def test_failed_attempt_is_rendered_and_does_not_advance(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "output"
    isaaclab_root = tmp_path / "IsaacLab"
    isaaclab_root.mkdir()
    (isaaclab_root / "isaaclab.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    rendered = []

    def fake_run(command, *, cwd, log_path, env=None):
        del cwd, log_path, env
        attempt_dir = Path(command[command.index("--output-dir") + 1])
        episode = attempt_dir / "episode"
        episode.mkdir(parents=True)
        (episode / "replay-manifest.json").write_text("{}\n", encoding="utf-8")
        (attempt_dir / "ros-replan.log").write_text(
            "[node] [104.0] controlled braking requested: dynamic_collision\n"
            "[node] [108.0] goal reached; holding position\n",
            encoding="utf-8",
        )
        return 0

    def fake_render(**kwargs):
        kwargs["video"].write_bytes(b"video")
        kwargs["screenshot"].write_bytes(b"image")
        Path(kwargs["command"][kwargs["command"].index("--output_json") + 1]).write_text(
            "{}\n", encoding="utf-8"
        )
        rendered.append(kwargs["attempt_dir"])

    monkeypatch.setattr(runner, "_run_streaming", fake_run)
    monkeypatch.setattr(runner, "_render_until_saved", fake_render)

    result = runner.main(
        [
            "--mode",
            "phase4",
            "--output-dir",
            output_dir.as_posix(),
            "--isaaclab-root",
            isaaclab_root.as_posix(),
            "--skip-build",
            "--retry-delay-sec",
            "0",
            "--max-attempts-per-scenario",
            "1",
        ]
    )

    attempt = output_dir / "runs/scenario-000/phase4/attempt-001"
    assert result == 2
    assert rendered == [attempt]
    assert (attempt / "replay.mp4").read_bytes() == b"video"
    assert (attempt / "attempt-result.json").is_file()
    assert not (output_dir / "runs/scenario-001").exists()


def test_pipeline_command_runs_f3_c_with_external_attempt_rendering(tmp_path: Path) -> None:
    command = _pipeline_command(tmp_path, "f3_c")

    assert command[command.index("--phase") + 1] == "factorized"
    assert command[command.index("--factorized-method") + 1] == "f3"
    assert command[command.index("--planner-seed") + 1] == "123"
    assert "--factorized-adapt-spatial-basis" in command
    assert "--allow-brake" in command
    assert "--skip-render" in command
