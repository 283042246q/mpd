import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts.isaaclab.benchmark_todrawer_random import (
    CATEGORIES,
    DIFFICULTIES,
    MODE_SPECS,
    PREDICTION_HORIZON_S,
    _existing_rows,
    _normalize_metrics,
    _paired_summary,
    _parser,
    _trajectory_segment_metrics,
    extract_run_metrics,
    generate_suite,
    write_reports,
)


def test_random_suite_is_deterministic_and_covers_categories():
    first = generate_suite(len(CATEGORIES), 42)
    second = generate_suite(len(CATEGORIES), 42)

    assert first == second
    assert [item["category"] for item in first["scenarios"]] == list(CATEGORIES)
    assert first["schema_version"] == 2
    assert all(1 <= len(item["objects"]) <= 3 for item in first["scenarios"])
    assert all(item["schema"] == "mpd_todrawer_dynamic_scenario" for item in first["scenarios"])


def test_random_suite_uses_stratified_motion_and_structural_feasibility():
    suite = generate_suite(5 * len(CATEGORIES), 1234)

    assert suite["generation_policy"]["maximum_simultaneous_crossings"] == 2
    assert suite["generation_policy"]["vertical_crossings_retained"] is True
    motion_types = set()
    vertical_crossings = 0
    for scenario in suite["scenarios"]:
        assert scenario["difficulty"] in DIFFICULTIES
        objects = scenario["objects"]
        corridor_ids = [item["corridor_id"] for item in objects]
        assert len(corridor_ids) == len(set(corridor_ids))
        for item in objects:
            motion_types.add(item["motion_model"])
            vertical_crossings += abs(item["direction"][2]) > 0.9
            assert 0.08 <= item["speed_m_s"] <= 0.32
            if item["motion_model"] == "constant_acceleration":
                acceleration = item["motion"]["longitudinal_acceleration_m_s2"]
                for elapsed in (0.0, 35.0):
                    velocity = item["speed_m_s"] + acceleration * (
                        elapsed - item["crossing_time_s"]
                    )
                    assert 0.04 - 1.0e-12 <= velocity <= 0.38 + 1.0e-12
            inflation = item["inflation"]
            horizon_inflation = (
                inflation["base_m"] + PREDICTION_HORIZON_S * inflation["horizon_rate_m_s"]
            )
            assert horizon_inflation <= 0.23 + 1.0e-12

        crossing_times = sorted(item["crossing_time_s"] for item in objects)
        if scenario["category"] in {"simultaneous_multi", "curved_crossing"}:
            assert len(objects) == 2
            assert len(scenario["reserved_corridors"]) == 1
        if scenario["category"] in {
            "staggered_multi",
            "inflated_dense",
            "accelerating_crossing",
        }:
            assert all(
                second - first >= 4.5 for first, second in zip(crossing_times, crossing_times[1:])
            )
        if scenario["category"] == "safe_control":
            assert min(crossing_times) >= 55.0
    assert vertical_crossings > 0
    assert motion_types == {
        "constant_velocity",
        "constant_acceleration",
        "sinusoidal_curve",
        "smooth_speed_variation",
        "curved_speed_variation",
    }


def test_benchmark_defaults_to_large_five_mode_matrix():
    args = _parser().parse_args([])

    assert args.scenario_count == 50
    assert args.repeats == 5
    assert args.modes == list(MODE_SPECS)
    assert args.categories is None
    assert MODE_SPECS["phase4"] == ("phase4", None)
    assert MODE_SPECS["phase4_aligned"] == ("phase4_aligned", None)


def test_realized_joint_path_uses_only_active_interval(tmp_path):
    episode = tmp_path / "episode"
    archive = episode / "plans" / "plan-0000" / "trajectory.npz"
    archive.parent.mkdir(parents=True)
    np.savez_compressed(
        archive,
        positions=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        velocities=np.zeros((3, 2)),
        accelerations=np.zeros((3, 2)),
        time_from_start=np.asarray([0.0, 1.0, 2.0]),
    )
    plans = [
        {
            "trajectory": "plans/plan-0000/trajectory.npz",
            "start_s": 5.0,
            "active_from_s": 5.5,
            "active_until_s": 6.5,
        }
    ]

    l2, l1 = _trajectory_segment_metrics(episode / "replay-manifest.json", plans)

    assert l2 == pytest.approx(1.0)
    assert l1 == pytest.approx(1.0)


def test_extract_metrics_and_report_from_synthetic_completed_run(tmp_path):
    attempt = tmp_path / "runs" / "scenario-000" / "repeat-00" / "joint" / "attempt-001"
    episode = attempt / "episode"
    archive = episode / "plans" / "plan-0000" / "trajectory.npz"
    archive.parent.mkdir(parents=True)
    np.savez_compressed(
        archive,
        positions=np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
        velocities=np.zeros((3, 2)),
        accelerations=np.zeros((3, 2)),
        time_from_start=np.asarray([0.0, 1.0, 2.0]),
    )
    manifest = {
        "duration_s": 4.0,
        "plans": [
            {
                "id": "plan-0000",
                "status": "accepted",
                "trajectory": "plans/plan-0000/trajectory.npz",
                "start_s": 1.0,
                "active_from_s": 1.0,
                "active_until_s": 3.0,
                "phase_timing": {"mpd_suffix_s": 8.5},
                "candidate_clearance_diagnostics": [
                    {
                        "candidate_index": 0,
                        "composite_cost": 0.2,
                        "hard_minimum_clearance_m": 0.04,
                        "common_window_minimum_clearance_m": 0.03,
                        "clearance_mean_cost": 0.1,
                        "clearance_cvar_cost": 0.3,
                    }
                ],
            }
        ],
        "events": [{"type": "handoff", "time_s": 1.2}],
    }
    (episode / "replay-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (attempt / "to_drawer-replan-timing.json").write_text(
        json.dumps(
            {
                "maximum_uncovered_command_gap_s": 0.0,
                "guarded_terminal_hold_s": 0.4,
                "maximum_controller_reference_jump_rad": 0.01,
            }
        ),
        encoding="utf-8",
    )
    (attempt / "ros-replan.log").write_text(
        "[1.000] dynamic MPD replanner started\n"
        '[2.000] top-K candidate rejections: [{"reason": "dynamic_collision"}]\n'
        "[3.000] goal reached; holding position\n",
        encoding="utf-8",
    )
    result_dir = attempt / "planner-results" / "request-1"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "success",
                "timing": {"inference_total_sec": 0.4},
                "trajectory": {
                    "minimum_environment_clearance_m": 0.02,
                    "minimum_self_clearance_m": 0.08,
                },
            }
        ),
        encoding="utf-8",
    )
    run_spec = {
        "scenario_id": "scenario-000",
        "category": "single_crossing",
        "repeat": 0,
        "mode": "joint",
        "planner_seed": 42,
    }

    metrics = extract_run_metrics(attempt, run_spec, 0)

    assert metrics["pipeline_completed"]
    assert metrics["manifest_available"]
    assert metrics["failure_class"] is None
    assert metrics["goal_reached"]
    assert metrics["goal_time_s"] == pytest.approx(2.0)
    assert metrics["joint_l2_path_rad"] == pytest.approx(1.0)
    assert metrics["hard_minimum_clearance_m"] == pytest.approx(0.04)
    assert metrics["guard_dynamic_collision_rejections"] == 1

    suite = generate_suite(1, 42)
    write_reports(tmp_path, [metrics], suite)
    report = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "ToDrawer 随机动态重规划基准报告" in report
    assert "场景类型与难度" in report
    assert "joint" in report
    assert "时长、路径与推理耗时" in report
    assert "仅到达目标运行的路径与执行时长" in report
    assert "完整配对结果" in report
    assert "Clearance 汇总" in report
    assert "难度分层结果" in report
    assert "规划轨迹时长 mean s" in report
    assert (tmp_path / "report" / "runs.csv").is_file()


def test_missing_manifest_dds_startup_is_infrastructure_failure(tmp_path):
    attempt = tmp_path / "attempt-001"
    attempt.mkdir()
    (attempt / "ros-replan.log").write_text(
        "rmw_create_node: failed to create domain, error Error\n",
        encoding="utf-8",
    )

    metrics = extract_run_metrics(
        attempt,
        {
            "scenario_id": "scenario-001",
            "category": "staggered_multi",
            "repeat": 0,
            "mode": "joint",
        },
        1,
    )

    assert not metrics["manifest_available"]
    assert not metrics["pipeline_completed"]
    assert metrics["failure_class"] == "dds_startup"


def test_old_terminal_clip_failure_is_revalidated(tmp_path):
    attempt = tmp_path / "attempt-001"
    episode = attempt / "episode"
    episode.mkdir(parents=True)
    (episode / "replay-manifest.json").write_text(
        json.dumps(
            {
                "duration_s": 10.000001,
                "plans": [
                    {
                        "id": "terminal",
                        "status": "accepted",
                        "active_from_s": 10.0,
                        "active_until_s": 10.000001,
                        "phase_timing": {
                            "planning_submitted_s": 8.0,
                            "bridge_start_s": 10.0,
                            "handoff_s": 10.2,
                            "mpd_suffix_s": 8.75,
                        },
                    }
                ],
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    (attempt / "pipeline.log").write_text(
        "ValueError: executed plan 0 has inconsistent phase timing\n",
        encoding="utf-8",
    )

    normalized = _normalize_metrics(
        {
            "attempt_dir": attempt.as_posix(),
            "pipeline_completed": False,
            "pipeline_returncode": 1,
            "error": None,
        }
    )

    assert normalized["pipeline_completed"]
    assert normalized["pipeline_revalidated"]
    assert normalized["terminal_clipped_plan_count"] == 1
    assert normalized["maximum_command_gap_s"] == pytest.approx(0.0)
    assert normalized["maximum_uncovered_command_gap_s"] == pytest.approx(0.0)
    assert normalized["failure_class"] is None


def test_reversed_active_interval_is_a_real_continuity_failure(tmp_path):
    attempt = tmp_path / "attempt-001"
    episode = attempt / "episode"
    episode.mkdir(parents=True)
    (episode / "replay-manifest.json").write_text(
        json.dumps({"duration_s": 10.0, "plans": [], "events": []}),
        encoding="utf-8",
    )
    (attempt / "pipeline.log").write_text(
        "plans[4]: active interval is outside the trajectory duration\n",
        encoding="utf-8",
    )

    normalized = _normalize_metrics(
        {
            "attempt_dir": attempt.as_posix(),
            "pipeline_completed": False,
            "pipeline_returncode": 1,
        }
    )

    assert normalized["manifest_available"]
    assert normalized["failure_class"] == "command_continuity"


def test_paired_summary_requires_manifest_from_all_five_modes():
    rows = [
        {
            "scenario_id": "scenario-000",
            "repeat": 0,
            "mode": mode,
            "manifest_available": True,
            "pipeline_completed": True,
            "goal_reached": mode != "joint",
            "brake_count": int(mode == "phase4"),
            "execution_duration_s": 2.0,
            "joint_l2_path_rad": 1.0,
            "joint_l1_travel_rad": 2.0,
        }
        for mode in MODE_SPECS
    ]

    paired = _paired_summary(rows)

    assert paired["cell_count"] == 1
    assert paired["by_mode"]["phase4"]["goal_reached"] == 1
    assert paired["by_mode"]["phase4"]["goal_and_brake_runs"] == 1
    assert paired["by_mode"]["joint"]["goal_reached"] == 0


def test_existing_rows_keep_historical_infrastructure_attempt_count(tmp_path):
    root = tmp_path / "runs" / "scenario-000" / "repeat-00" / "joint"
    first = root / "attempt-001"
    second = root / "attempt-002"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "ros-replan.log").write_text(
        "rmw_create_node: failed to create domain\n", encoding="utf-8"
    )
    common = {"scenario_id": "scenario-000", "repeat": 0, "mode": "joint"}
    (first / "run-metrics.json").write_text(
        json.dumps(
            {
                **common,
                "attempt_dir": first.as_posix(),
                "pipeline_completed": False,
            }
        ),
        encoding="utf-8",
    )
    (second / "run-metrics.json").write_text(
        json.dumps(
            {
                **common,
                "attempt_dir": second.as_posix(),
                "pipeline_completed": True,
            }
        ),
        encoding="utf-8",
    )

    rows = _existing_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["attempt_count"] == 2
    assert rows[0]["infrastructure_failure_attempts"] == 1
    assert rows[0]["pipeline_completed"]


def test_benchmark_rejects_invalid_explicit_ros_domain():
    script = Path("scripts/isaaclab/benchmark_todrawer_random.py")

    result = subprocess.run(
        [
            "/home/eric/anaconda3/envs/mpd-splines-public/bin/python",
            script.as_posix(),
            "--ros-domain-id",
            "233",
            "--report-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "ros-domain-id must lie in [0, 232]" in result.stderr
