from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from mpd.inference.space_time_guidance import SpaceTimeGuidanceSettings
from scripts.runtime.dynamic_runtime_engine_marvin_bimanual import (
    MarvinBimanualDynamicRuntimeEngine,
)
from scripts.runtime.infer_dynamic_server_marvin_bimanual import _parser as dynamic_parser
from scripts.runtime.infer_space_time_server_marvin_bimanual import (
    _parser as space_time_parser,
)
from scripts.runtime.runtime_engine_marvin_bimanual import PlanArtifacts
from scripts.runtime.space_time_runtime_engine_marvin_bimanual import (
    MarvinBimanualSpaceTimeRuntimeEngine,
)
from scripts.inference.inference_marvin_bimanual import _resolve_model_dir
from mpd.utils.loaders import load_params_from_yaml


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG = ROOT / "scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml"


def _world():
    covariance = [0.0] * 36
    covariance[0] = 0.04
    return {
        "world_version": 3,
        "frame_id": "world",
        "stamp_unix_ns": 1_000_000_000,
        "valid_until_unix_ns": 20_000_000_000,
        "objects": [
            {
                "id": "moving-box",
                "local_sdf": {"type": "box", "size_xyz": [1.0, 0.5, 1.5]},
                "pose": {
                    "position": [1.0, 2.0, 0.75],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "linear_velocity": [0.5, 0.0, 0.0],
                "covariance_6x6": covariance,
                "inflation": {"mode": "covariance", "base_m": 0.1},
            }
        ],
    }


class _WorldCapture:
    def __init__(self):
        self.payload = None

    def update(self, payload):
        self.payload = payload
        return int(payload["world_version"])


def test_runtime_config_is_locked_to_an_existing_compatible_checkpoint():
    config = load_params_from_yaml(RUNTIME_CONFIG)
    model_dir = _resolve_model_dir(config)
    checkpoint_args = load_params_from_yaml(model_dir / "args.yaml")
    assert checkpoint_args["dataset_subdir"] == config["dataset_subdir"]
    assert checkpoint_args["bimanual_network_variant"] == config["runtime"]["network_variant"]
    assert (model_dir / "checkpoints" / config["checkpoint"]).is_file()


@pytest.mark.parametrize(
    ("mode", "expected_velocity"),
    [("snapshot_no_time", [0.0, 0.0, 0.0]), ("fixed_time_dynamic", [0.5, 0.0, 0.0])],
)
def test_phase6_runtime_modes_freeze_only_snapshot(mode, expected_velocity):
    engine = MarvinBimanualDynamicRuntimeEngine.__new__(MarvinBimanualDynamicRuntimeEngine)
    engine.runtime_mode = mode
    engine.covariance_sigma = 3.0
    engine.dynamic_world = _WorldCapture()
    assert engine.update_world(_world()) == 3
    obstacle = engine.dynamic_world.payload["objects"][0]
    assert obstacle["linear_velocity"] == expected_velocity
    if mode == "fixed_time_dynamic":
        assert obstacle["covariance_6x6"][0] == pytest.approx(0.04)
        assert obstacle["inflation"]["mode"] == "covariance"
    else:
        assert obstacle["covariance_6x6"] == [0.0] * 36


def test_phase6_and_phase7_servers_have_separate_modes_and_sockets():
    fixed = dynamic_parser().parse_args(
        [
            "--socket",
            "/tmp/fixed.sock",
            "--output-root",
            "/tmp/fixed",
            "--runtime-mode",
            "fixed_time_dynamic",
        ]
    )
    timed = space_time_parser().parse_args(["--socket", "/tmp/timed.sock", "--output-root", "/tmp/timed"])
    assert fixed.runtime_mode == "fixed_time_dynamic"
    assert fixed.socket != timed.socket
    assert timed.timing_mode == "phase5_joint"


def test_phase7_upgrades_top_k_to_candidate_specific_schema_v3(monkeypatch):
    candidate_times = torch.tensor([[0.0, 3.0, 7.0], [0.0, 4.0, 9.0]], dtype=torch.float32)
    timing_control_points = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    results = SimpleNamespace(
        candidate_timesteps=candidate_times,
        q_trajs_pos_iter_0=torch.zeros((2, 3, 14)),
        timing_control_points=timing_control_points,
    )
    artifacts = PlanArtifacts(
        result_payload={
            "trajectory_artifact": {"schema_version": 2},
            "dynamic_world": {"fixed_timing": True},
        },
        trajectory_arrays={
            "top_k_candidate_indices": np.asarray([1, 0], dtype=np.int64),
            "time_from_start": np.asarray([0.0, 5.0, 10.0]),
        },
    )
    monkeypatch.setattr(
        MarvinBimanualDynamicRuntimeEngine,
        "plan",
        lambda self, request: artifacts,
    )
    engine = MarvinBimanualSpaceTimeRuntimeEngine.__new__(MarvinBimanualSpaceTimeRuntimeEngine)
    engine._session = SimpleNamespace(
        config=SimpleNamespace(n_trajectory_samples=2),
        last_plan_results=results,
        device=torch.device("cpu"),
    )
    engine.space_time_guide = SimpleNamespace(reset=lambda count: None, statistics=[])
    engine.space_time_settings = SpaceTimeGuidanceSettings()
    engine.runtime_mode = "inference_time_optimized"

    output = engine.plan({"runtime_mode": "inference_time_optimized"})
    arrays = output.trajectory_arrays
    assert int(arrays["artifact_schema_version"]) == 3
    assert int(arrays["timing_schema_version"]) == 1
    np.testing.assert_allclose(arrays["top_k_time_from_start"][0], [0.0, 4.0, 9.0])
    np.testing.assert_allclose(arrays["time_from_start"], [0.0, 4.0, 9.0])
    assert arrays["timing_control_points"].shape == (2, 8)
    assert output.result_payload["dynamic_world"]["candidate_specific_time"] is True
