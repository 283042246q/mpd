from copy import deepcopy
import h5py
import numpy as np
import pytest
import yaml

from scripts.generate_data.generate_marvin_warehouse_cooperative import (
    CONTEXT_DATASETS,
    DEFAULT_CONFIG,
    PHASE_IDS,
    SOLUTION_DATASETS,
    TASK_MODE,
    box_surface_spheres,
    interpolate_object_states,
    obb_signed_distance,
    phase_schedule,
    validate_config,
    write_dataset,
)
from scripts.generate_data.launch_generate_marvin_warehouse_cooperative import build_context_shards


def load_config():
    return yaml.safe_load(DEFAULT_CONFIG.read_text())


def test_production_config_encodes_requested_cooperative_pipeline():
    config = load_config()
    validate_config(config)
    assert config["task_mode"] == TASK_MODE
    assert config["object_planning"] == {
        **config["object_planning"],
        "state_space": "xyz_yaw",
        "proposals_per_task": 4,
        "use_lift_transfer_lower": True,
        "use_rrt_connect": True,
        "simplify_path": False,
    }
    assert config["continuous_ik"]["initial_branches"] == 4
    assert config["continuous_ik"]["beam_width"] == 3
    assert config["joint_optimization"]["degree"] == 5
    assert config["joint_optimization"]["control_points"] == 22
    assert config["validation"]["path_points"] == 128
    assert config["validation"]["spline_points"] == 512
    assert config["validation"]["adaptive_max_joint_step"] == 0.025
    assert config["dataset"]["num_contexts"] == 100
    assert config["dataset"]["solutions_per_task_target"] == 4
    assert config["launcher"]["workers"] == 3
    assert config["launcher"]["worker_lifetime_trajectories"] == 10


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("task_mode",), "dual_independent", "cooperative"),
        (("payload", "fixed_geometry"), False, "payload"),
        (("object_planning", "state_space"), "se3", "xyz_yaw"),
        (("object_planning", "simplify_path"), True, "not be simplified"),
        (("joint_optimization", "hard_closed_chain"), False, "hard constraints"),
        (("validation", "payload_collision"), False, "must be true"),
    ],
)
def test_invalid_cooperative_contract_is_rejected(path, value, message):
    config = deepcopy(load_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_xyz_object_rrt_schema_is_supported_alongside_default_xyz_yaw():
    config = load_config()
    config["object_planning"]["state_space"] = "xyz"
    config["object_planning"]["bounds"] = config["object_planning"]["bounds"][:3]
    config["object_planning"]["fixed_yaw"] = 0.0
    validate_config(config)


def test_object_rotation_and_regions_are_checked_during_dry_run_validation():
    config = load_config()
    config["object_planning"]["base_rotation"][0][0] = 2.0
    with pytest.raises(ValueError, match="base_rotation"):
        validate_config(config)
    config = load_config()
    config["object_regions"]["goal"]["x"] = [0.8, 0.2]
    with pytest.raises(ValueError, match="object_regions.goal.x"):
        validate_config(config)


def test_object_interpolation_preserves_vertices_endpoints_and_unwraps_yaw():
    states = np.array([[0, 0, 0.2, np.deg2rad(170)], [0, 0, 0.4, np.deg2rad(-170)]])
    result = interpolate_object_states(states, 9)
    assert result.shape == (9, 4)
    assert np.allclose(result[[0, -1], :3], states[[0, -1], :3])
    assert np.max(np.abs(np.diff(result[:, 3]))) < np.deg2rad(10)


def test_phase_schedule_allows_support_only_at_task_boundaries():
    phases = phase_schedule(20)
    assert phases[0] == PHASE_IDS["grasp"]
    assert phases[-1] == PHASE_IDS["place"]
    assert PHASE_IDS["lift"] in phases
    assert PHASE_IDS["transfer"] in phases
    assert PHASE_IDS["lower"] in phases
    assert not np.isin(phases[1:-1], [PHASE_IDS["grasp"], PHASE_IDS["place"]]).any()


def test_payload_box_proxy_and_exact_obb_distance():
    points, radius = box_surface_spheres([0.30, 0.24, 0.20], 0.025)
    assert points.shape[1] == 3 and len(points) > 100 and radius == 0.025
    pose = np.eye(4)
    distance = obb_signed_distance(np.array([[0, 0, 0], [0.20, 0, 0]]), pose, [0.15, 0.12, 0.10])
    assert np.allclose(distance, [-0.10, 0.05])


def test_launcher_honors_ten_trajectory_worker_lifetime_for_four_solution_contexts():
    shards = build_context_shards(100, max_contexts_per_shard=2, workers=3)
    assert len(shards) == 50
    assert sum(count for _, count in shards) == 100
    assert all(count <= 2 for _, count in shards)
    assert [start for start, _ in shards] == list(range(0, 100, 2))


def test_empty_solution_dataset_preserves_failed_context(tmp_path):
    config = load_config()
    contexts = [
        {
            "task_id": 7,
            "object_start": np.full(4, np.nan),
            "object_goal": np.full(4, np.nan),
            "solutions": 0,
            "failure_reason": "context_sampling",
        }
    ]
    write_dataset(tmp_path, config, [], [], contexts, seed=1, stats={"context_sampling_failure": 1})
    with h5py.File(tmp_path / "dataset_merged.hdf5", "r") as data:
        assert data["sol_path"].shape == (0, 128, 14)
        assert data["bspline_params_cc"].shape == (0, 14, 22)
        assert data["context_task_id"][:] == [7]
        assert data["context_solutions_found"][:] == [0]
        assert data["context_failure_reason"].asstr()[:] == ["context_sampling"]
        assert all(key in data for key in SOLUTION_DATASETS + CONTEXT_DATASETS)
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert manifest["num_contexts"] == 1
    assert manifest["num_trajectories"] == 0
    assert manifest["failed_context_reasons"] == {"context_sampling": 1}
