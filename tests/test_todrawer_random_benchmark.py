import json
from pathlib import Path

import numpy as np
import pytest

from scripts.isaaclab.benchmark_todrawer_random import (
    CATEGORIES,
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
    assert all(1 <= len(item["objects"]) <= 5 for item in first["scenarios"])
    assert all(
        item["schema"] == "mpd_todrawer_dynamic_scenario"
        for item in first["scenarios"]
    )


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
        json.dumps({"maximum_command_gap_s": 0.0}), encoding="utf-8"
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
    assert metrics["goal_reached"]
    assert metrics["goal_time_s"] == pytest.approx(2.0)
    assert metrics["joint_l2_path_rad"] == pytest.approx(1.0)
    assert metrics["hard_minimum_clearance_m"] == pytest.approx(0.04)
    assert metrics["guard_dynamic_collision_rejections"] == 1

    suite = generate_suite(1, 42)
    write_reports(tmp_path, [metrics], suite)
    report = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "ToDrawer 随机动态重规划基准报告" in report
    assert "joint" in report
    assert "时长、路径与推理耗时" in report
    assert "Clearance 汇总" in report
    assert "规划轨迹时长 mean s" in report
    assert (tmp_path / "report" / "runs.csv").is_file()
