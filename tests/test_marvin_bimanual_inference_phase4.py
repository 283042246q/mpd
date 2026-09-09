from pathlib import Path
import threading
import time

import numpy as np

from mpd.bimanual.runtime_contract import JOINT_NAMES, SCHEMA
from scripts.runtime.infer_server import ResidentPlannerService
from scripts.runtime.runtime_engine_marvin_bimanual import (
    MarvinBimanualRuntimeEngine,
    PlanArtifacts,
)


def _request(request_id="resident", world_version=0):
    return {
        "schema": SCHEMA,
        "request_id": request_id,
        "task_mode": "dual_independent",
        "runtime_mode": "snapshot_no_time",
        "robot_model": "marvin_bimanual",
        "planning_frame": "world",
        "scene_id": "EnvWarehouseMarvinBimanual",
        "scene_version": "marvin_warehouse_v2",
        "joint_names": list(JOINT_NAMES),
        "q_start": [0.0] * 14,
        "q_goal": [0.1] * 14,
        "world_version": world_version,
    }


class _Session:
    loads = 0

    def __init__(self, _config, _output, _device, callback):
        type(self).loads += 1
        self.instance_id = "resident-test"
        callback("WARMING")

    def health(self):
        return {
            "loads": 1,
            "checkpoint_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "scene_sha256": "c" * 64,
            "robot_sha256": "d" * 64,
            "device": "cpu",
            "warmup_elapsed_sec": 0.01,
            "dense_validation_enabled": True,
            "b3_enabled": True,
        }

    def plan(self, request):
        positions = np.asarray([request.q_start, request.q_goal], dtype=float)
        zero = np.zeros_like(positions)
        result = {
            "schema": "marvin_bimanual_result/v2",
            "status": "success",
            "request_id": request.request_id,
            "joint_names": list(JOINT_NAMES),
            "positions": positions.tolist(),
            "velocities": zero.tolist(),
            "accelerations": zero.tolist(),
            "time_from_start": [0.0, 1.0],
            "world_version": request.world_version,
            "validation": {"valid": True},
        }
        return PlanArtifacts(
            result,
            {
                "positions": positions,
                "velocities": zero,
                "accelerations": zero,
                "time_from_start": np.asarray([0.0, 1.0]),
                "joint_names": np.asarray(JOINT_NAMES),
            },
        )


def _message(seq=1, deadline=None):
    return {
        "schema_version": 1,
        "op": "plan",
        "request_seq": seq,
        "world_version": 0,
        "deadline_unix_ns": deadline,
        "request": _request(f"resident-{seq}"),
    }


def test_resident_engine_constructs_session_once():
    _Session.loads = 0
    engine = MarvinBimanualRuntimeEngine(
        Path("unused"), Path("/tmp/unused"), "cpu", lambda _: None,
        session_factory=_Session,
    )
    first = engine.plan(_request("one"))
    second = engine.plan(_request("two"))
    assert _Session.loads == 1
    assert engine.health()["loads"] == 1
    np.testing.assert_array_equal(
        first.trajectory_arrays["positions"], second.trajectory_arrays["positions"]
    )


def test_service_has_deadline_stale_and_atomic_artifacts(tmp_path):
    engine = MarvinBimanualRuntimeEngine(
        Path("unused"), tmp_path, "cpu", lambda _: None, session_factory=_Session
    )
    service = ResidentPlannerService(
        tmp_path / "worker.sock", tmp_path, lambda _: engine
    )
    service._engine = engine
    service.set_state("READY")
    stale = service.dispatch(_message(1, time.time_ns() - 1))
    assert stale["status"] == "STALE"
    response = service.dispatch(_message(2, time.time_ns() + 1_000_000_000))
    assert response["status"] == "OK"
    assert Path(response["result_path"]).is_file()
    assert Path(response["trajectory_path"]).is_file()
    repeated = service.dispatch(_message(2, None))
    assert repeated["status"] == "STALE"


def test_non_reentrant_planning_returns_busy(tmp_path):
    service = ResidentPlannerService(
        tmp_path / "worker.sock", tmp_path, lambda _: None
    )
    service.set_state("READY")
    service._plan_lock.acquire()
    try:
        assert service.dispatch(_message(3))["status"] == "BUSY"
    finally:
        service._plan_lock.release()
