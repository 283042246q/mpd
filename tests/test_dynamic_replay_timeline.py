import json

import numpy as np
import pytest

from scripts.isaaclab.dynamic_replay_timeline import (
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_RED,
    DynamicReplayError,
    active_plan_at,
    brake_event_at,
    latest_pending_plan_at,
    load_dynamic_replay_manifest,
    plan_base_color,
    predict_object,
    robot_position_at,
    segment_color,
    world_snapshot_at,
)


def _trajectory(path, offset):
    np.savez_compressed(
        path,
        positions=np.asarray([[offset] * 7, [offset + 1.0] * 7, [offset + 2.0] * 7]),
        time_from_start=np.asarray([0.0, 1.0, 2.0]),
    )


def _manifest(tmp_path):
    _trajectory(tmp_path / "old.npz", 0.0)
    _trajectory(tmp_path / "new.npz", 10.0)
    _trajectory(tmp_path / "rejected.npz", 20.0)
    payload = {
        "schema": "mpd_dynamic_replay",
        "schema_version": 1,
        "duration_s": 5.0,
        "plans": [
            {
                "id": "old",
                "trajectory": "old.npz",
                "created_s": 0.0,
                "start_s": 0.5,
                "status": "superseded",
                "active_from_s": 0.5,
                "active_until_s": 2.0,
                "collision_segments_s": [[1.0, 1.5]],
            },
            {
                "id": "new",
                "trajectory": "new.npz",
                "created_s": 1.0,
                "start_s": 2.0,
                "status": "accepted",
                "active_from_s": 2.0,
                "active_until_s": 4.0,
            },
            {
                "id": "rejected",
                "trajectory": "rejected.npz",
                "created_s": 2.5,
                "start_s": 3.0,
                "status": "rejected",
            },
        ],
        "world_snapshots": [
            {
                "time_s": 0.0,
                "world_version": 1,
                "valid_until_s": 10.0,
                "objects": [
                    {
                        "id": "box",
                        "local_sdf": {"type": "box", "size_xyz": [0.2, 0.3, 0.4]},
                        "pose": {
                            "position": [0.0, 0.0, 0.0],
                            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                        "linear_velocity": [0.0, 0.5, 0.0],
                        "inflation": {"mode": "linear", "base_m": 0.01, "horizon_rate_m_s": 0.02},
                    }
                ],
            },
            {"time_s": 2.0, "world_version": 2, "valid_until_s": 12.0, "objects": []},
        ],
        "events": [
            {"type": "handoff", "time_s": 2.0, "plan_id": "new"},
            {"type": "brake", "time_s": 4.0, "duration_s": 0.5, "reason": "guard_collision"},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_drives_active_pending_and_obsolete_colors(tmp_path):
    manifest = load_dynamic_replay_manifest(_manifest(tmp_path))
    assert active_plan_at(manifest, 1.5).plan_id == "old"
    assert latest_pending_plan_at(manifest, 1.5).plan_id == "new"
    old, new, rejected = manifest.plans
    assert plan_base_color(old, manifest, 1.5) == COLOR_BLUE
    assert plan_base_color(new, manifest, 1.5) == COLOR_GREEN
    assert plan_base_color(old, manifest, 2.5) == COLOR_GRAY
    assert plan_base_color(rejected, manifest, 3.0) == COLOR_RED
    assert segment_color(old, 1.2, COLOR_GRAY) == COLOR_RED


def test_manifest_reads_deduplicated_trajectory_schema_v2(tmp_path):
    path = _manifest(tmp_path)
    np.savez(
        tmp_path / "new.npz",
        artifact_schema_version=np.asarray(2, dtype=np.int64),
        best_trajectory_topk_index=np.asarray(1, dtype=np.int64),
        topk_positions=np.asarray(
            [
                [[30.0] * 7, [31.0] * 7, [32.0] * 7],
                [[10.0] * 7, [11.0] * 7, [12.0] * 7],
            ]
        ),
        time_from_start=np.asarray([0.0, 1.0, 2.0]),
    )
    manifest = load_dynamic_replay_manifest(path)
    assert manifest.plans[1].trajectory.positions[0].tolist() == [10.0] * 7


def test_robot_command_is_interpolated_across_handoff(tmp_path):
    manifest = load_dynamic_replay_manifest(_manifest(tmp_path))
    assert robot_position_at(manifest, 1.0) == pytest.approx([0.5] * 7)
    assert robot_position_at(manifest, 2.5) == pytest.approx([10.5] * 7)
    assert robot_position_at(manifest, 4.5) == pytest.approx([12.0] * 7)


def test_world_prediction_uses_constant_velocity_and_linear_inflation(tmp_path):
    manifest = load_dynamic_replay_manifest(_manifest(tmp_path))
    snapshot = world_snapshot_at(manifest, 1.0)
    predicted = predict_object(snapshot.objects[0], snapshot.time_s, 1.0)
    assert predicted.position == pytest.approx([0.0, 0.5, 0.0])
    assert predicted.inflation_m == pytest.approx(0.03)
    assert world_snapshot_at(manifest, 2.5).world_version == 2


def test_covariance_inflation_and_brake_window(tmp_path):
    path = _manifest(tmp_path)
    manifest = load_dynamic_replay_manifest(path)
    item = dict(manifest.world_snapshots[0].objects[0])
    item["inflation_mode"] = "covariance"
    item["covariance_6x6"] = np.eye(6) * 0.01
    predicted = predict_object(item, 0.0, 1.0)
    assert predicted.inflation_m > 0.3
    assert brake_event_at(manifest, 4.25).reason == "guard_collision"
    assert brake_event_at(manifest, 4.75) is None


def test_invalid_active_interval_is_rejected(tmp_path):
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["plans"][0]["active_until_s"] = 9.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DynamicReplayError, match="active interval"):
        load_dynamic_replay_manifest(path)


def test_overlapping_active_intervals_are_rejected(tmp_path):
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["plans"][1]["active_from_s"] = 1.5
    payload["plans"][1]["start_s"] = 1.5
    payload["plans"][1]["active_until_s"] = 3.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DynamicReplayError, match="overlap"):
        load_dynamic_replay_manifest(path)


def test_world_timeline_must_be_ordered_and_cover_replay(tmp_path):
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["world_snapshots"][1]["time_s"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DynamicReplayError, match="time_s must increase"):
        load_dynamic_replay_manifest(path)

    path = _manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["world_snapshots"][-1]["valid_until_s"] = 4.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DynamicReplayError, match="does not cover"):
        load_dynamic_replay_manifest(path)
