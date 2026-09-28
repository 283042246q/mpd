from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.isaaclab.analyze_todrawer_mode_motion_start import (
    analyze_logs,
    measure_first_plan_completion,
    measure_manifest_motion_start,
)


def _write_replay(tmp_path):
    episode = tmp_path / "logs" / "f3_c" / "attempt-001" / "episode"
    trajectory = episode / "plans" / "plan-0000" / "trajectory.npz"
    trajectory.parent.mkdir(parents=True)
    positions = np.zeros((5, 7), dtype=np.float64)
    positions[2, 3] = 0.011
    positions[3, 3] = 0.021
    positions[4, 3] = 0.1
    np.savez_compressed(
        trajectory,
        positions=positions,
        velocities=np.zeros_like(positions),
        accelerations=np.zeros_like(positions),
        time_from_start=np.asarray([0.0, 0.1, 0.2, 0.3, 0.4]),
    )
    manifest = {
        "initial_q": [0.0] * 7,
        "episode_start_unix_ns": 10_200_000_000,
        "world_start_unix_ns": 10_000_000_000,
        "plans": [
            {
                "id": "plan-0000",
                "status": "accepted",
                "trajectory": "plans/plan-0000/trajectory.npz",
                "start_s": 3.0,
                "phase_timing": {
                    "planning_submitted_s": 1.0,
                    "bridge_start_s": 2.8,
                    "handoff_s": 3.0,
                },
            }
        ],
    }
    path = episode / "replay-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result_dir = episode.parent / "planner-results" / "request-1"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps({"status": "success", "created_unix_time": 12.5}), encoding="utf-8"
    )
    return path


def test_measure_manifest_motion_start_uses_joint_displacement_threshold(tmp_path):
    manifest = _write_replay(tmp_path)

    low = measure_manifest_motion_start(manifest, threshold_rad=0.01)
    high = measure_manifest_motion_start(manifest, threshold_rad=0.02)

    assert low is not None and high is not None
    assert low["motion_start_from_world_s"] == pytest.approx(3.4)
    assert high["motion_start_from_world_s"] == pytest.approx(3.5)
    assert low["handoff_from_world_s"] == pytest.approx(3.2)
    assert measure_first_plan_completion(manifest) == pytest.approx(2.5)


def test_analyze_logs_groups_mode_and_threshold(tmp_path):
    _write_replay(tmp_path)

    report = analyze_logs(
        tmp_path / "logs", modes=("f3_c",), thresholds_rad=(0.01, 0.02)
    )

    assert report["modes"]["f3_c"]["0.01"]["count"] == 1
    assert report["modes"]["f3_c"]["0.01"]["median"] == pytest.approx(3.4)
    assert report["modes"]["f3_c"]["0.02"]["median"] == pytest.approx(3.5)
    assert report["modes"]["f3_c"]["first_plan_completed_from_world_s"]["median"] == pytest.approx(2.5)
