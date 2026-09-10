import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mpd.bimanual.runtime_contract import JOINT_NAMES, SCHEMA
from scripts.inference import inference_marvin_bimanual as inference
from scripts.isaaclab import marvin_bimanual_subprocess as subprocess_bridge
from scripts.isaaclab.marvin_bimanual_asset import (
    CANONICAL_JOINT_NAMES,
    MARVIN_CONTACT_BODY_PATHS,
    TCP_BODY_NAMES,
    TCP_FRAME_NAMES,
    classify_contact_forces,
    load_inference_artifact,
    resolve_canonical_joint_ids,
    resolve_tcp_body_ids,
    sha256_file,
    sha256_tree,
    tcp_poses_from_body_state,
    trajectory_physics_step_schedule,
    validate_marvin_urdf,
    validate_marvin_usd,
    validate_scene_payload,
)


def _request():
    pose = {
        "frame_id": "world",
        "pose_xyzw": [0.5, 0.1, 0.7, 0.0, 0.0, 0.0, 1.0],
    }
    return {
        "schema": SCHEMA,
        "request_id": "phase2-replay",
        "task_mode": "dual_independent",
        "runtime_mode": "snapshot_no_time",
        "robot_model": "marvin_bimanual",
        "planning_frame": "world",
        "scene_id": "EnvWarehouseMarvinBimanual",
        "scene_version": "marvin_warehouse_v2",
        "joint_names": list(JOINT_NAMES),
        "q_start": [0.0] * 14,
        "q_goal": [0.1] * 14,
        "left_goal_pose": pose,
        "right_goal_pose": pose,
    }


def _scene():
    return {
        "schema": "mpd_isaaclab_scene",
        "schema_version": 1,
        "env_name": "EnvWarehouseMarvinBimanual",
        "frame_id": "world",
        "obstacles": [
            {
                "type": "box",
                "name": "warehouse_table",
                "group": "fixed",
                "position": [0.4, 0.0, 0.2],
                "orientation": [1.0, 0.0, 0.0, 0.0],
                "size": [0.8, 1.0, 0.4],
            }
        ],
        "unsupported_obstacles": [],
    }


def test_locked_local_asset_and_tcp_rigid_body_mapping():
    gate = validate_marvin_urdf()
    assert gate["urdf_sha256"] == "41e057811aef172f568a3353b84f1185780f9c65921275c128bf13fad96ab180"
    assert gate["asset_manifest_sha256"] == "0086fa3d8ee69cb54f5d4159926f919fbd551ff3c96635d385a464f54539b06e"
    assert gate["tcp_frame_names"] == TCP_FRAME_NAMES
    assert gate["tcp_body_names"] == TCP_BODY_NAMES
    assert gate["tcp_offsets_xyz"] == ((0.0, 0.0, 0.21),) * 2
    assert len(gate["mesh_hashes"]) == 20


def test_isaac_native_interleaving_maps_to_left_then_right_without_gripper_writes():
    native_joints = tuple(
        name
        for index in range(1, 8)
        for name in (f"Joint{index}_L", f"Joint{index}_R")
    )
    robot = SimpleNamespace(
        joint_names=native_joints,
        body_names=("base_link", *TCP_BODY_NAMES),
    )
    ids = resolve_canonical_joint_ids(robot)
    assert [native_joints[index] for index in ids] == list(CANONICAL_JOINT_NAMES)
    assert ids == [0, 2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 13]
    assert resolve_tcp_body_ids(robot) == (1, 2)


def test_preconverted_usd_requires_matching_local_bundle_record(tmp_path):
    gate = validate_marvin_urdf()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    usd = bundle / "marvin.usda"
    usd.write_text("#usda 1.0\n")
    metadata = {
        "schema": "marvin_bimanual_isaaclab_asset/v1",
        "urdf_sha256": gate["urdf_sha256"],
        "asset_manifest_sha256": gate["asset_manifest_sha256"],
        "usd_sha256": sha256_file(usd),
        "usd_bundle_sha256": sha256_tree(bundle),
        "converter": "test",
    }
    (bundle / "marvin_bimanual_asset.json").write_text(json.dumps(metadata))
    assert validate_marvin_usd(usd)["usd_bundle_sha256"] == metadata["usd_bundle_sha256"]
    usd.write_text("#usda 1.0\n# drift\n")
    with pytest.raises(ValueError, match="usd_sha256"):
        validate_marvin_usd(usd)


def test_tcp_pose_is_composed_from_gripper_base_fixed_offset():
    positions = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    # Identity for left, +90 degrees around Y for right (Isaac body-state xyzw).
    quaternions = np.asarray(
        [[0.0, 0.0, 0.0, 1.0], [0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)]]
    )
    tcp_positions, rotations = tcp_poses_from_body_state(positions, quaternions)
    np.testing.assert_allclose(tcp_positions[0], [1.0, 2.0, 3.21], atol=1e-12)
    np.testing.assert_allclose(tcp_positions[1], [-0.79, 0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        rotations @ np.swapaxes(rotations, -1, -2),
        np.broadcast_to(np.eye(3), (2, 3, 3)),
        atol=1e-12,
    )


def test_contact_categories_do_not_hide_total_contact():
    names = ("Link7_L", "Link7_R", "base_link")
    forces = np.asarray(
        [
            [[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    categories = classify_contact_forces(names, forces, threshold=1.0)
    assert categories["interarm_contact"].tolist() == [True, False, False]
    assert categories["left_world_contact"].tolist() == [False, True, True]
    assert categories["right_world_contact"].tolist() == [False, True, False]
    external = classify_contact_forces(
        names, forces, threshold=1.0, infer_interarm=False
    )
    assert external["interarm_contact"].tolist() == [False, False, False]
    assert external["left_world_contact"].tolist() == [True, True, True]
    assert external["right_world_contact"].tolist() == [True, True, False]


def test_nested_marvin_contact_sensor_paths_cover_unique_rigid_bodies():
    names = [name for name, _ in MARVIN_CONTACT_BODY_PATHS]
    paths = [path for _, path in MARVIN_CONTACT_BODY_PATHS]
    assert len(names) == len(paths) == 26
    assert len(set(names)) == len(names)
    assert len(set(paths)) == len(paths)
    assert names[:2] == ["base_link", "column_link"]
    assert "Link7_L" in names and "Link7_R" in names
    assert "left_gripper_base_link" in names
    assert "right_gripper_base_link" in names


def test_timestamp_schedule_preserves_trajectory_duration_and_legacy_override():
    times = np.linspace(0.0, 10.0, 128)
    schedule = trajectory_physics_step_schedule(times, physics_dt=0.005)
    assert schedule[0] == 0
    assert schedule[1:].min() == 15
    assert schedule[1:].max() == 16
    assert schedule.sum() == 2000
    np.testing.assert_array_equal(
        trajectory_physics_step_schedule(times, 0.005, action_repeat=4),
        np.full(128, 4),
    )


def test_warehouse_scene_gate_rejects_frame_and_primitive_drift():
    assert validate_scene_payload(_scene())["n_boxes"] == 1
    bad_frame = _scene()
    bad_frame["frame_id"] = "base_link"
    with pytest.raises(ValueError, match="frame_id"):
        validate_scene_payload(bad_frame)
    unsupported = _scene()
    unsupported["unsupported_obstacles"] = [{"field_type": "MeshField"}]
    with pytest.raises(ValueError, match="unsupported"):
        validate_scene_payload(unsupported)


def test_phase1_stub_artifact_is_directly_replay_loadable(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request()))
    artifact_path = tmp_path / "artifact"
    assert inference.main(
        [
            "--request", str(request_path),
            "--output-dir", str(artifact_path),
            "--backend", "contract_stub",
            "--stub-points", "6",
        ]
    ) == 0
    artifact = load_inference_artifact(artifact_path)
    assert artifact.positions.shape == (6, 14)
    assert artifact.ee_goal_pose.shape == (2, 3, 4)
    assert artifact.active_ee_mask.tolist() == [1.0, 1.0]
    assert artifact.scene["frame_id"] == "world"


def test_mpd_isaaclab_backend_sequences_evaluation_then_replay(tmp_path, monkeypatch):
    calls = []

    def fake_evaluate(artifact, output_json, log_path, **kwargs):
        calls.append(("evaluate", Path(artifact), Path(output_json), kwargs))
        return {"schema": "marvin_bimanual_isaaclab_evaluation/v1"}

    def fake_replay(artifact, output_json, log_path, **kwargs):
        calls.append(("replay", Path(artifact), Path(output_json), kwargs))
        return {"schema": "marvin_bimanual_isaaclab_replay/v1"}

    monkeypatch.setattr(subprocess_bridge, "run_marvin_isaaclab_evaluator", fake_evaluate)
    monkeypatch.setattr(subprocess_bridge, "run_marvin_isaaclab_replay", fake_replay)
    args = SimpleNamespace(
        isaaclab_conda_env="env_isaaclab",
        isaaclab_device="cpu",
        isaaclab_headless=True,
        isaaclab_action_repeat=3,
        isaaclab_timeout_s=60,
        isaaclab_asset_cache=tmp_path / "cache",
        isaaclab_replay=True,
        isaaclab_capture=True,
        isaaclab_video=None,
        isaaclab_screenshot=None,
        isaaclab_trajectory_index=0,
        isaaclab_video_fps=24.0,
        isaaclab_width=640,
        isaaclab_height=360,
    )
    summary = inference._run_isaaclab_backend(args, tmp_path / "artifact")
    assert [call[0] for call in calls] == ["evaluate", "replay"]
    assert calls[1][3]["evaluation"] == tmp_path / "artifact/isaaclab-evaluation.json"
    assert calls[1][3]["video_path"] == tmp_path / "artifact/isaaclab-replay.mp4"
    assert summary["status"] == "completed"


def test_isaaclab_failure_does_not_overwrite_successful_mpd_artifact(tmp_path, monkeypatch):
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request()))
    artifact_path = tmp_path / "artifact"

    def fail_backend(args, output_dir):
        raise RuntimeError("Isaac unavailable")

    monkeypatch.setattr(inference, "_run_isaaclab_backend", fail_backend)
    returncode = inference.main(
        [
            "--request", str(request_path),
            "--output-dir", str(artifact_path),
            "--backend", "contract_stub",
            "--sim-backend", "isaaclab",
            "--no-isaaclab-capture",
        ]
    )
    assert returncode == 6
    assert json.loads((artifact_path / "result.json").read_text())["status"] == "success"
    isaac_run = json.loads((artifact_path / "isaaclab-run.json").read_text())
    assert isaac_run["status"] == "fault"
    assert isaac_run["error"]["message"] == "Isaac unavailable"
