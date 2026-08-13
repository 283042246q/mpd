from pathlib import Path
import socket
import threading
import time

import numpy as np

from scripts.runtime.infer_server import ResidentPlannerService
from scripts.runtime.ipc_protocol import receive_message, send_message
from scripts.runtime.runtime_engine import PlanArtifacts


class FakeEngine:
    instance_id = "fake-engine"

    def __init__(self, state_callback):
        state_callback("LOADING")
        state_callback("WARMING")

    def health(self):
        return {"instance_id": self.instance_id, "fully_warmed": True}

    def plan(self, request):
        positions = np.zeros((4, 7), dtype=np.float64)
        return PlanArtifacts(
            result_payload={
                "schema_version": 1,
                "status": "success",
                "request_id": request["request_id"],
                "trajectory_file": "trajectory.npz",
            },
            trajectory_arrays={
                "positions": positions,
                "velocities": positions.copy(),
                "accelerations": positions.copy(),
                "time_from_start": np.linspace(0.0, 1.0, 4),
                "joint_names": np.asarray([f"fr3_joint{i}" for i in range(1, 8)]),
            },
        )


def _request(socket_path: Path, message):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(2.0)
        stream.connect(socket_path.as_posix())
        send_message(stream, message)
        return receive_message(stream)


def _wait_until_ready(socket_path: Path):
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if socket_path.exists():
            try:
                response = _request(
                    socket_path,
                    {"schema_version": 1, "op": "health"},
                )
                if response["state"] == "READY":
                    return response
            except (ConnectionError, OSError):
                pass
        threading.Event().wait(0.01)
    raise AssertionError("Resident planner did not become READY.")


def test_service_plans_once_and_rejects_replayed_sequence(tmp_path):
    socket_path = tmp_path / "mpd.sock"
    output_root = tmp_path / "output"
    service = ResidentPlannerService(
        socket_path,
        output_root,
        engine_factory=FakeEngine,
    )
    thread = threading.Thread(target=service.serve_forever)
    thread.start()
    try:
        health = _wait_until_ready(socket_path)
        assert health["engine"]["instance_id"] == "fake-engine"

        message = {
            "schema_version": 1,
            "op": "plan",
            "request_seq": 7,
            "world_version": 3,
            "deadline_unix_ns": time.time_ns() + 2_000_000_000,
            "request": {"request_id": "test-plan"},
        }
        response = _request(socket_path, message)
        assert response["status"] == "OK"
        assert response["request_seq"] == 7
        assert Path(response["result_path"]).is_file()
        assert Path(response["trajectory_path"]).is_file()

        replay = _request(socket_path, message)
        assert replay["status"] == "STALE"
        assert replay["reason"] == "request_seq_not_newer"
    finally:
        if socket_path.exists():
            _request(socket_path, {"schema_version": 1, "op": "shutdown"})
        thread.join(timeout=3.0)
    assert not thread.is_alive()
    assert not socket_path.exists()


def test_service_rejects_expired_deadline_without_planning(tmp_path):
    socket_path = tmp_path / "mpd.sock"
    service = ResidentPlannerService(
        socket_path,
        tmp_path / "output",
        engine_factory=FakeEngine,
    )
    thread = threading.Thread(target=service.serve_forever)
    thread.start()
    try:
        _wait_until_ready(socket_path)
        response = _request(
            socket_path,
            {
                "schema_version": 1,
                "op": "plan",
                "request_seq": 1,
                "world_version": 0,
                "deadline_unix_ns": time.time_ns() - 1,
                "request": {"request_id": "expired"},
            },
        )
        assert response["status"] == "STALE"
        assert response["reason"] == "deadline_expired_before_planning"
    finally:
        if socket_path.exists():
            _request(socket_path, {"schema_version": 1, "op": "shutdown"})
        thread.join(timeout=3.0)
