import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from mpd.bimanual.runtime_contract import JOINT_NAMES, SCHEMA


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/runtime/infer_once_marvin_bimanual.py"


def _request(request_id="phase3-one-shot"):
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
    }


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_strict_one_shot_publishes_bound_atomic_artifact(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request()), encoding="utf-8")
    artifact = tmp_path / "artifact"
    subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "--request",
            str(request_path),
            "--output-dir",
            str(artifact),
            "--backend",
            "contract_stub",
            "--stub-points",
            "6",
        ],
        cwd=ROOT,
        check=True,
    )
    result = json.loads((artifact / "result.json").read_text())
    assert result["request_id"] == "phase3-one-shot"
    assert result["one_shot"]["entrypoint"] == ENTRY.name
    assert result["artifacts"] == {
        "request_sha256": _sha256(request_path),
        "trajectory_sha256": _sha256(artifact / "trajectory.npz"),
        "scene_file_sha256": _sha256(artifact / "scene.json"),
    }
    with np.load(artifact / "trajectory.npz", allow_pickle=False) as trajectory:
        assert trajectory["positions"].shape == (6, 14)


def test_one_shot_rejects_non_snapshot_runtime(tmp_path):
    request = _request("phase3-bad-mode")
    request["runtime_mode"] = "fixed_time_dynamic"
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    artifact = tmp_path / "artifact"
    completed = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "--request",
            str(request_path),
            "--output-dir",
            str(artifact),
            "--backend",
            "contract_stub",
        ],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 2
    result = json.loads((artifact / "result.json").read_text())
    assert result["status"] == "invalid_request"
    assert not (artifact / "trajectory.npz").exists()
