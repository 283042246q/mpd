import json

import pytest

from scripts.isaaclab.replay_marvin_bimanual_dynamic_log import (
    build_timing_report,
    candidate_visual_states,
    predicted_world_objects,
    selected_plan,
    world_start_unix_ns,
)


def _record():
    return {
        "schema": "marvin_bimanual_dynamic_replay/v1",
        "request_id": "demo",
        "events": [
            {
                "unix_ns": 10,
                "sequence": 1,
                "type": "world",
                "payload": {
                    "world_version": 1,
                    "stamp_unix_ns": 1_000_000_000,
                    "objects": [
                        {
                            "id": "box",
                            "local_sdf": {"type": "box", "size_xyz": [1, 2, 3]},
                            "pose": {
                                "position": [1.0, 2.0, 3.0],
                                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                            },
                            "linear_velocity": [0.5, 0.0, -0.25],
                        }
                    ],
                },
            },
            {
                "unix_ns": 20,
                "sequence": 2,
                "type": "plan_selected",
                "payload": {
                    "result_path": "/tmp/request/result.json",
                    "trajectory_path": "/tmp/request/trajectory.npz",
                    "top_k_index": 0,
                    "trajectory_start_unix_ns": 2_000_000_000,
                },
            },
        ],
    }


def test_dynamic_record_resolves_plan_and_predicts_objects():
    record = _record()
    assert selected_plan(record)["top_k_index"] == 0
    objects = predicted_world_objects(record, 3_000_000_000)
    assert objects[0]["pose"]["position"] == pytest.approx([2.0, 2.0, 2.5])
    assert world_start_unix_ns(record) == 1_000_000_000


def test_candidate_colors_follow_franka_state_semantics():
    record = _record()
    record["events"][1]["payload"]["generation"] = 3
    record["events"].insert(
        1,
        {
            "unix_ns": 15,
            "sequence": 2,
            "type": "candidate_revalidation",
            "payload": {"generation": 3, "candidate": 0, "safe": True},
        },
    )
    record["events"].append(
        {
            "unix_ns": 16,
            "sequence": 3,
            "type": "candidate_revalidation",
            "payload": {"generation": 3, "candidate": 1, "safe": False},
        }
    )
    assert candidate_visual_states(record, 14, 2) == ["hidden", "hidden"]
    assert candidate_visual_states(record, 19, 2) == ["gray", "red"]
    assert candidate_visual_states(record, 21, 2) == ["green", "red"]
    assert candidate_visual_states(record, 2_000_000_000, 2) == ["blue", "red"]


def test_explicit_world_start_precedes_first_snapshot():
    record = _record()
    record["events"].append(
        {
            "unix_ns": 500_000_000,
            "sequence": 0,
            "type": "world_start",
            "payload": {"scenario_start_unix_ns": 500_000_000},
        }
    )
    assert world_start_unix_ns(record) == 500_000_000


def test_pipeline_script_exposes_safe_modes():
    source = open("scripts/isaaclab/run_marvin_bimanual_dynamic_demo_pipeline.sh").read()
    assert "--task-mode" in source
    assert "--execute" in source
    assert "env -u CYCLONEDDS_URI" in source
    assert "replay_marvin_bimanual_trajectory.py" in source
    assert 'WORLD_SCENARIO="warehouse_core_crossing"' in source
    assert "crossing_three|warehouse_core_crossing" in source
    assert "PLANNING_BUDGET_S=15" in source
    assert "PLANNING_BUDGET_S=35" in source
    assert "timing-report.json" in source


@pytest.mark.parametrize("task_mode", ["independent", "cooperative"])
def test_timing_report_covers_discard_and_selected_attempts(task_mode):
    base = 1_000_000_000
    events = [
        (0, "world_start", {"scenario_start_unix_ns": base}),
        (1, "plan_start", {"attempt": 1, "generation": 1}),
        (2, "worker_return_timing", {"backend_elapsed_s": 0.7, "worker_timing_s": {"engine_plan": 0.5}}),
        (3, "plan_discard", {"generation": 1}),
        (4, "plan_start", {"attempt": 2, "generation": 2}),
        (5, "worker_return_timing", {"backend_elapsed_s": 0.8}),
        (6, "candidate_selection_timing", {"timing_s": {"hard_revalidation": 0.1}}),
        (
            7,
            "plan_selected",
            {
                "generation": 2,
                "trajectory_start_unix_ns": base + 10_000_000_000,
                "trajectory_duration_s": 2.5,
                "timing_s": {"trajectory_conversion": 0.01},
            },
        ),
        (8, "execution_timing", {"elapsed_s": 1.0, "goal_send_elapsed_s": 0.1}),
        (9, "action_terminal_timing", {"elapsed_s": 8.0, "success": True}),
    ]
    record = {
        "schema": "marvin_bimanual_dynamic_replay/v1",
        "request_id": "timed",
        "events": [
            {"unix_ns": base + offset * 1_000_000_000, "sequence": index, "type": kind, "payload": payload}
            for index, (offset, kind, payload) in enumerate(events)
        ],
    }
    report = build_timing_report(
        record, task_mode=task_mode, pipeline_stamps_ns={"pipeline_start": base, "worker_ready": base + 2_000_000_000}
    )
    assert report["task_mode"] == task_mode
    assert report["world_start_to_first_plan_s"] == 1.0
    assert report["pipeline_timing_s"]["worker_startup"] == 2.0
    assert [item["outcome"] for item in report["attempts"]] == ["discarded", "selected"]
    assert report["attempts"][0]["plan_wall_elapsed_s"] == 2.0
    assert report["attempts"][1]["plan_wall_elapsed_s"] == 3.0
    assert report["attempts"][1]["scheduled_wait_s"] == 3.0
    assert report["attempts"][1]["trajectory_duration_s"] == 2.5
    assert report["attempts"][1]["execution_timings"][0]["elapsed_s"] == 1.0
    assert report["action_terminal_timing"]["elapsed_s"] == 8.0


def test_timing_report_preserves_failed_worker_attempt():
    record = {
        "schema": "marvin_bimanual_dynamic_replay/v1",
        "request_id": "failed",
        "events": [
            {"unix_ns": 1_000_000_000, "sequence": 1, "type": "plan_start", "payload": {"attempt": 1, "generation": 1}},
            {
                "unix_ns": 3_000_000_000,
                "sequence": 2,
                "type": "worker_failure_timing",
                "payload": {"backend_elapsed_s": 1.9, "error": "no valid trajectory"},
            },
            {
                "unix_ns": 3_100_000_000,
                "sequence": 3,
                "type": "action_terminal_timing",
                "payload": {"elapsed_s": 2.1, "success": False},
            },
        ],
    }
    report = build_timing_report(record, task_mode="independent")
    assert report["attempts"][0]["outcome"] == "worker_failed"
    assert report["attempts"][0]["plan_wall_elapsed_s"] == 2.0
    assert report["attempts"][0]["worker_failure_timing"]["error"] == "no valid trajectory"
    assert report["action_terminal_timing"]["success"] is False
