from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import BSpline
import yaml

from scripts.generate_data.augment_marvin_warehouse_reverse import (
    augment_dataset,
    main,
    reverse_bspline_parameters,
)
from scripts.generate_data.generate_marvin_warehouse_bimanual import (
    DEFAULT_CONFIG,
    _write_dataset,
    file_sha256,
)


def _knots():
    return np.r_[np.zeros(6), np.linspace(1.0 / 17.0, 16.0 / 17.0, 16), np.ones(6)]


def _dual_fk(q):
    q = np.asarray(q)
    poses = np.zeros((len(q), 2, 3, 4), dtype=np.float32)
    poses[:, :, :3, :3] = np.eye(3, dtype=np.float32)
    poses[:, 0, :, 3] = q[:, :3]
    poses[:, 1, :, 3] = q[:, 7:10]
    return poses


def _write_source(root: Path):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    paths, metadata = [], []
    specifications = (
        (10, "left_only", "random_to_placement", "random", "left_table"),
        (11, "dual_independent", "placement_to_random", "left_table", "random"),
    )
    for row, (task_id, mode, direction, source_left, goal_left) in enumerate(specifications):
        q_start = np.linspace(0.01 + row, 0.14 + row, 14)
        q_goal = q_start + 0.25
        path = np.linspace(q_start, q_goal, 128)
        coefficients = np.linspace(q_start, q_goal, 22).T
        pose = _dual_fk(q_goal[None])[0]
        item = {
            "task_id": task_id,
            "task_mode": mode,
            "direction": direction,
            "q_start": q_start,
            "q_goal": q_goal,
            "planning_time": 2.0 + row,
            "joint_path_length": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            "bspline": (_knots(), coefficients, 5),
            "ee_goal_pose": pose,
            "source_region_left": source_left,
            "goal_region_left": goal_left,
            "source_region_right": "inactive" if mode == "left_only" else "right_table",
            "goal_region_right": "inactive" if mode == "left_only" else "random",
        }
        paths.append(path)
        metadata.append(item)
    _write_dataset(root, config, paths, metadata, config["seed"])


def test_reverse_bspline_parameters_preserve_the_reversed_curve():
    rng = np.random.default_rng(7)
    knots = _knots()[None]
    coefficients = rng.normal(size=(1, 14, 22))
    degree = np.array([5])
    reverse_knots, reverse_coefficients, reverse_degree = reverse_bspline_parameters(knots, coefficients, degree)
    u = np.linspace(0.0, 1.0, 101)
    forward = BSpline(knots[0], coefficients[0].T, int(degree[0]))(1.0 - u)
    reverse = BSpline(reverse_knots[0], reverse_coefficients[0].T, int(reverse_degree[0]))(u)
    np.testing.assert_allclose(reverse, forward, atol=1e-12)


def test_marvin_v3_reverse_augmentation_updates_all_contract_fields(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "EnvWarehouse-RobotMarvinBimanual-independent-v3-reversed-test"
    _write_source(source)
    source_digest = file_sha256(source / "dataset_merged.hdf5")

    report = augment_dataset(source, output, chunk_size=1, dual_fk_pose_fn=_dual_fk)
    assert report["source_num_trajectories"] == 2
    assert report["output_num_trajectories"] == 4
    assert file_sha256(source / "dataset_merged.hdf5") == source_digest

    with h5py.File(source / "dataset_merged.hdf5", "r") as original, h5py.File(
        output / "dataset_merged.hdf5", "r"
    ) as augmented:
        np.testing.assert_array_equal(augmented["task_id"][:], [10, 10, 11, 11])
        np.testing.assert_array_equal(augmented["augmentation_pair_id"][:], [10, 10, 11, 11])
        np.testing.assert_array_equal(augmented["augmentation_is_reversed"][:], [False, True, False, True])
        np.testing.assert_array_equal(augmented["augmentation_source_row"][:], [0, 0, 1, 1])
        np.testing.assert_allclose(augmented["sol_path"][1], original["sol_path"][0][::-1])
        np.testing.assert_allclose(augmented["q_start"][1], original["q_goal"][0])
        np.testing.assert_allclose(augmented["q_goal"][1], original["q_start"][0])
        np.testing.assert_allclose(augmented["bspline_params_cc"][1], original["bspline_params_cc"][0][:, ::-1])
        expected_knots = 1.0 - original["bspline_params_tt"][0][::-1]
        np.testing.assert_allclose(augmented["bspline_params_tt"][1], expected_knots)
        np.testing.assert_allclose(augmented["ee_goal_pose"][0], _dual_fk(original["q_goal"][:1])[0])
        np.testing.assert_allclose(augmented["ee_goal_pose"][1], _dual_fk(original["q_start"][:1])[0])
        assert augmented["direction"].asstr()[:].tolist() == [
            "random_to_placement",
            "placement_to_random",
            "placement_to_random",
            "random_to_placement",
        ]
        assert augmented["source_region_left"].asstr()[:2].tolist() == ["random", "left_table"]
        assert augmented["goal_region_left"].asstr()[:2].tolist() == ["left_table", "random"]
        np.testing.assert_array_equal(augmented["active_joint_mask"][0], augmented["active_joint_mask"][1])
        np.testing.assert_array_equal(augmented["active_ee_mask"][0], augmented["active_ee_mask"][1])
        assert bool(augmented.attrs["forward_reverse_pairs_share_task_id"])

    manifest = yaml.safe_load((output / "manifest.yaml").read_text())
    assert manifest["num_trajectories"] == 4
    assert manifest["dataset_sha256"] == file_sha256(output / "dataset_merged.hdf5")
    assert manifest["direction_counts"] == {
        "random_to_placement": 2,
        "placement_to_random": 2,
    }
    assert manifest["reverse_augmentation"]["pairs_share_task_id"] is True
    assert (output / "augmentation_report.json").is_file()

    # The production loader maps every row with the same task_id to one group
    # before train/validation splitting. Reusing the task ID is the no-leakage
    # contract, while the explicit pair field makes it auditable.
    with h5py.File(output / "dataset_merged.hdf5", "r") as augmented:
        groups = {}
        for row, task_id in enumerate(augmented["task_id"][:]):
            groups.setdefault(int(task_id), []).append(row)
    assert groups == {10: [0, 1], 11: [2, 3]}

    import mpd.paths
    from scripts.train.train_marvin_warehouse_bimanual import validate_dataset

    monkeypatch.setattr(mpd.paths, "DATASET_BASE_DIR", str(tmp_path))
    training_config = yaml.safe_load(
        (Path(__file__).parents[1] / "scripts/train/cfgs/marvin_bimanual_warehouse_independent.yaml").read_text()
    )
    training_config["dataset_subdir"] = output.name
    assert validate_dataset(training_config)["num_trajectories"] == 4


def test_reverse_augmentation_dry_run_does_not_create_output(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)
    assert main([str(source), "--output-dir", str(output), "--dry-run"]) == 0
    assert not output.exists()
