import time
from types import SimpleNamespace

import pytest

from mpd.inference.dynamic_collision import DynamicWorldError
from scripts.isaaclab.replay_marvin_bimanual_dynamic_log import build_timeline
from scripts.runtime.dynamic_runtime_engine_marvin_bimanual import _snapshot_world
from scripts.runtime.infer_dynamic_server_marvin_bimanual import MarvinDynamicPlannerService
from scripts.inference.inference_marvin_bimanual import _resolve_model_dir


def _world(version=1):
    return {
        "world_version": version,
        "frame_id": "world",
        "stamp_unix_ns": 100,
        "valid_until_unix_ns": 10_000_000_000,
        "objects": [
            {
                "id": "forklift",
                "local_sdf": {"type": "box", "size_xyz": [1.0, 0.5, 1.5]},
                "pose": {"position": [1.0, 2.0, 0.75], "orientation_xyzw": [0, 0, 0, 1]},
                "linear_velocity": [0.5, 0.0, 0.0],
                "covariance_6x6": [0.04 if i in (0, 7, 14) else 0.0 for i in range(36)],
                "inflation": {"mode": "covariance", "base_m": 0.1},
            }
        ],
    }


def test_snapshot_freezes_motion_and_folds_covariance_into_margin():
    frozen = _snapshot_world(_world(), covariance_sigma=3.0)
    obstacle = frozen["objects"][0]
    assert obstacle["linear_velocity"] == [0.0, 0.0, 0.0]
    assert obstacle["inflation"]["mode"] == "linear"
    assert obstacle["inflation"]["horizon_rate_m_s"] == 0.0
    assert obstacle["inflation"]["base_m"] == pytest.approx(0.7)
    assert obstacle["covariance_6x6"] == [0.0] * 36


def test_snapshot_rejects_wrong_frame_and_bad_covariance():
    world = _world()
    world["frame_id"] = "map"
    with pytest.raises(DynamicWorldError, match="frame"):
        _snapshot_world(world, covariance_sigma=3.0)
    world = _world()
    world["objects"][0]["covariance_6x6"] = [0.0]
    with pytest.raises(DynamicWorldError, match="36"):
        _snapshot_world(world, covariance_sigma=3.0)


class _Engine:
    def __init__(self):
        self.dynamic_world = SimpleNamespace(world_version=0)

    def update_world(self, world):
        version = int(world["world_version"])
        if version <= self.dynamic_world.world_version:
            raise DynamicWorldError("world_version must increase monotonically")
        self.dynamic_world.world_version = version
        return version


def test_dynamic_service_rejects_repeated_and_stale_versions(tmp_path):
    service = MarvinDynamicPlannerService(tmp_path / "x.sock", tmp_path, lambda _: None)
    service._engine = _Engine()
    service.set_state("READY")
    first = service.dispatch({"schema_version": 1, "op": "update_world", "world": _world(1)})
    assert first["status"] == "OK"
    repeated = service.dispatch({"schema_version": 1, "op": "update_world", "world": _world(1)})
    assert repeated["status"] == "INVALID_REQUEST"
    stale = service.dispatch(
        {"schema_version": 1, "op": "plan", "request_seq": 1, "world_version": 0}
    )
    assert stale["status"] == "STALE"


def test_dynamic_replay_has_deterministic_latest_only_order():
    record = {
        "schema": "marvin_bimanual_dynamic_replay/v1",
        "request_id": "demo",
        "events": [
            {"unix_ns": 20, "sequence": 2, "type": "joint_state", "payload": {}},
            {"unix_ns": 10, "sequence": 1, "type": "world", "payload": {"world_version": 1}},
            {"unix_ns": 20, "sequence": 1, "type": "world", "payload": {"world_version": 2}},
        ],
    }
    timeline = build_timeline(record)
    assert timeline["world_versions"] == [1, 2]
    assert [item["sequence"] for item in timeline["events"]] == [1, 1, 2]


def test_model_parent_resolution_is_deterministic(tmp_path):
    run = tmp_path / "1234"
    run.mkdir()
    (run / "args.yaml").write_text("robot_model: marvin_bimanual\n")
    config = {"model_selection": "bspline", "model_dir_ddpm_bspline": str(tmp_path)}
    assert _resolve_model_dir(config) == run
    other = tmp_path / "5678"
    other.mkdir()
    (other / "args.yaml").write_text("robot_model: marvin_bimanual\n")
    with pytest.raises(Exception, match="multiple runs"):
        _resolve_model_dir(config)
