import json

import pytest

from scripts.isaaclab.replay_marvin_bimanual_dynamic_log import (
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
