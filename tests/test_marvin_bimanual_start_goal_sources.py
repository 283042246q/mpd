import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import yaml

from mpd.bimanual import start_goal_sources
from mpd.bimanual.runtime_contract import BimanualRequest
from scripts.generate_data.generate_marvin_warehouse_bimanual import validate_config
from scripts.inference import inference_marvin_bimanual
from scripts.runtime import infer_once_marvin_bimanual


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml"
STATES = ROOT / "scripts/inference/cfgs/start_goal_states/EnvWarehouse-RobotMarvinBimanual-states.yaml"
REGIONS = ROOT / "scripts/inference/cfgs/start_goal_regions/EnvWarehouse-RobotMarvinBimanual-regions.yaml"


def test_checked_in_states_template_builds_strict_bimanual_requests():
    first = start_goal_sources.request_from_states_file(STATES, request_id="states-0", seed=7, sample_index=0)
    second = start_goal_sources.request_from_states_file(STATES, request_id="states-1", seed=7, sample_index=1)
    assert BimanualRequest.from_dict(first).request_id == "states-0"
    assert BimanualRequest.from_dict(second).request_id == "states-1"
    assert first["q_start"] != second["q_start"]
    assert first["scene"]["start_goal_source"]["type"] == "states_file"


def test_dataset_source_filters_dual_independent_and_keeps_dual_ee_pose(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "marvin-data"
    dataset_dir.mkdir()
    with h5py.File(dataset_dir / "dataset_merged.hdf5", "w") as dataset:
        dataset["q_start"] = np.asarray([[0.0] * 14, [0.1] * 14, [0.2] * 14])
        dataset["q_goal"] = np.asarray([[0.3] * 14, [0.4] * 14, [0.5] * 14])
        goals = np.zeros((3, 2, 3, 4))
        goals[:, :, :3, :3] = np.eye(3)
        goals[:, 0, :, 3] = [0.4, 0.3, 0.2]
        goals[:, 1, :, 3] = [0.4, -0.3, 0.2]
        dataset["ee_goal_pose"] = goals
        dataset["task_mode"] = np.asarray([b"left_only", b"dual_independent", b"dual_independent"])

    monkeypatch.setattr(start_goal_sources, "DATASET_BASE_DIR", tmp_path)
    request = start_goal_sources.request_from_dataset(
        {"dataset_subdir": "marvin-data", "dataset_file_merged": "dataset_merged.hdf5"},
        request_id="dataset-0",
        seed=11,
        sample_index=0,
    )
    parsed = BimanualRequest.from_dict(request)
    assert parsed.q_start == (0.1,) * 14
    assert request["scene"]["start_goal_source"]["row"] == 1
    assert parsed.left_goal_pose is not None
    assert parsed.right_goal_pose is not None


def test_direct_phase1_cli_generates_and_persists_request_from_config(tmp_path):
    config = tmp_path / "runtime.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "start_goal_source": "states_file",
                "start_goal_states_path": str(STATES),
            }
        )
    )
    artifact = tmp_path / "artifact"
    returncode = inference_marvin_bimanual.main(
        [
            "--config",
            str(config),
            "--output-dir",
            str(artifact),
            "--backend",
            "contract_stub",
            "--sample-index",
            "1",
            "--request-id",
            "direct-states-1",
            "--stub-points",
            "6",
        ]
    )
    assert returncode == 0
    generated = json.loads((artifact / "request.json").read_text())
    result = json.loads((artifact / "result.json").read_text())
    assert BimanualRequest.from_dict(generated).request_id == "direct-states-1"
    assert generated["scene"]["start_goal_source"]["index"] == 1
    assert result["request_id"] == "direct-states-1"


def test_regions_template_and_runtime_config_paths_are_valid():
    runtime = yaml.safe_load(CONFIG.read_text())
    assert start_goal_sources.resolve_source_path(CONFIG, runtime["start_goal_states_path"]) == STATES
    assert start_goal_sources.resolve_source_path(CONFIG, runtime["start_goal_regions_path"]) == REGIONS
    region_config = yaml.safe_load(REGIONS.read_text())
    assert validate_config(region_config)["task_mode"] == "dual_independent"


def test_regions_sampling_uses_system_entropy_and_ignores_seed_and_index(
    tmp_path, monkeypatch
):
    import scripts.generate_data.generate_marvin_warehouse_bimanual as generation

    constructor_seeds = []

    class FakeGenerator:
        def __init__(self, config, seed, progress_label=None):
            constructor_seeds.append(seed)
            self.rng = np.random.default_rng(123)
            self.deadline = float("inf")

        def _random_valid_state(self):
            return np.zeros(14)

        def _target_state(self, q_start, arm, region_name):
            return np.asarray(q_start)

        def valid(self, q):
            return True

        def _sample_endpoint(self, q_start, mode, regions):
            return np.ones(14)

        def _pose(self, q, arm):
            return SimpleNamespace(
                rotation=np.eye(3),
                translation=np.asarray([0.4, 0.3 if arm == "left" else -0.3, 0.2]),
            )

        def close(self):
            pass

    monkeypatch.setattr(generation, "MarvinWarehouseGenerator", FakeGenerator)
    region_file = tmp_path / "regions.yaml"
    region_file.write_text(
        yaml.safe_dump(
            {
                "schema": "marvin_bimanual_regions/v1",
                "inference_selection": {
                    "start": {"left": "random", "right": "random"},
                    "goal": {"left": "left_table", "right": "right_table"},
                },
            }
        )
    )

    first = start_goal_sources.request_from_regions(
        region_file, request_id="regions-a", seed=7, sample_index=3
    )
    second = start_goal_sources.request_from_regions(
        region_file, request_id="regions-b", seed=99, sample_index=88
    )

    assert constructor_seeds == [None, None]
    assert first["seed"] == 7 and second["seed"] == 99
    assert first["scene"]["start_goal_source"]["sampling"] == "system_entropy"
    assert second["scene"]["start_goal_source"]["sampling"] == "system_entropy"


def test_phase3_one_shot_still_requires_explicit_request():
    request_action = next(
        action for action in infer_once_marvin_bimanual._build_parser()._actions if action.dest == "request"
    )
    assert request_action.required is True
