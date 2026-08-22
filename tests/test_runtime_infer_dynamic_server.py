from pathlib import Path

import numpy as np

from scripts.runtime.infer_dynamic_server import DynamicResidentPlannerService
from scripts.runtime.runtime_engine import PlanArtifacts


class _World:
    world_version = 0


class FakeDynamicEngine:
    instance_id = "dynamic-engine"

    def __init__(self, state_callback):
        self.dynamic_world = _World()
        state_callback("WARMING")

    def health(self):
        return {
            "dynamic_world": {"world_version": self.dynamic_world.world_version},
            "dense_validation": {
                "fully_warmed": True,
                "full_batch": True,
                "pruning_used": False,
            },
        }

    def update_world(self, snapshot):
        self.dynamic_world.world_version = snapshot["world_version"]
        return self.dynamic_world.world_version

    def plan(self, request):
        assert request["_dynamic_world_version"] == self.dynamic_world.world_version
        assert request["_trajectory_start_unix_ns"] == 2_000_000_000
        return PlanArtifacts(
            result_payload={"status": "success"},
            trajectory_arrays={
                "positions": np.zeros((2, 7)),
                "velocities": np.zeros((2, 7)),
                "time_from_start": np.asarray([0.0, 1.0]),
                "joint_names": np.asarray([f"fr3_joint{i}" for i in range(1, 8)]),
            },
        )


def _service(tmp_path: Path):
    service = DynamicResidentPlannerService(
        tmp_path / "dynamic.sock",
        tmp_path / "output",
        engine_factory=FakeDynamicEngine,
    )
    service._engine = FakeDynamicEngine(service.set_state)
    service.set_state("READY")
    return service


def test_world_update_and_matching_plan_contract(tmp_path):
    service = _service(tmp_path)
    update = service.dispatch(
        {
            "schema_version": 1,
            "op": "update_world",
            "world": {"world_version": 7},
        }
    )
    assert update == {"schema_version": 1, "status": "OK", "world_version": 7}
    response = service.dispatch(
        {
            "schema_version": 1,
            "op": "plan",
            "request_seq": 8,
            "world_version": 7,
            "trajectory_start_unix_ns": 2_000_000_000,
            "deadline_unix_ns": None,
            "request": {"request_id": "dynamic"},
        }
    )
    assert response["status"] == "OK"
    assert response["world_version"] == 7
    assert response["trajectory_artifact"] == {
        "schema_version": 1,
        "compression": "zlib",
    }


def test_plan_rejects_world_version_that_is_not_loaded(tmp_path):
    service = _service(tmp_path)
    service._engine.dynamic_world.world_version = 4
    response = service.dispatch(
        {
            "schema_version": 1,
            "op": "plan",
            "request_seq": 9,
            "world_version": 5,
            "trajectory_start_unix_ns": 2_000_000_000,
            "request": {},
        }
    )
    assert response["status"] == "STALE"
    assert response["reason"] == "world_version_not_loaded"
    assert response["loaded_world_version"] == 4
