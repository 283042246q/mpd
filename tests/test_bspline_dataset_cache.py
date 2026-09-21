from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch


def _dataset_shell(root, robot):
    from mpd.datasets.trajectories_dataset_bspline import TrajectoryDatasetBspline

    dataset = object.__new__(TrajectoryDatasetBspline)
    dataset.planning_task = SimpleNamespace(
        robot=robot,
        parametric_trajectory=SimpleNamespace(
            bspline=SimpleNamespace(d=5, n_pts=8),
            zero_vel_at_start_and_goal=True,
            zero_acc_at_start_and_goal=True,
            remove_control_points_fn=lambda control_points: control_points[..., 1:-1, :],
        ),
    )
    dataset.base_dir = str(root)
    dataset.dataset_file_merged = "dataset_merged.hdf5"
    dataset.tensor_args = {"device": "cpu", "dtype": torch.float32}
    dataset.context_qs = True
    dataset.context_ee_goal_pose = True
    dataset.context_ee_goal_pose_bimanual = True
    dataset.field_key_control_points = "control_points"
    dataset.field_key_q_start = "q_start"
    dataset.field_key_q_goal = "q_goal"
    dataset.field_key_context_qs = "qs"
    dataset.field_key_context_ee_goal_pose = "ee_goal_pose"
    dataset.field_key_context_ee_goal_orientation = "ee_goal_orientation"
    dataset.field_key_context_ee_goal_position = "ee_goal_position"
    dataset.field_key_context_active_ee_mask = "active_ee_mask"
    dataset.fields = {}
    dataset.map_task_id_to_control_points_id = {}
    dataset.map_control_points_id_to_task_id = {}
    dataset.reload_data = False
    dataset.preload_data_to_device = False
    dataset.skip_collision_statistics_for_validated_splines = True
    return dataset


def test_serialized_bsplines_are_batch_loaded_and_validated_statistics_are_skipped(
    tmp_path, monkeypatch
):
    from mpd.datasets import trajectories_dataset_bspline as module

    class DummyMarvin:
        pass

    monkeypatch.setattr(module.robots, "RobotMarvinBimanual", DummyMarvin)
    coefficients = np.arange(4 * 14 * 8, dtype=np.float64).reshape(4, 14, 8)
    poses = np.zeros((4, 2, 3, 4), dtype=np.float32)
    poses[..., :3, :3] = np.eye(3, dtype=np.float32)
    with h5py.File(tmp_path / "dataset_merged.hdf5", "w") as data:
        data.attrs["validated_splines"] = True
        data.create_dataset("sol_path", shape=(4, 128, 14), dtype=np.float64)
        data.create_dataset(
            "bspline_params_cc", data=coefficients, chunks=(2, 14, 8), compression="gzip"
        )
        data.create_dataset("task_id", data=[10, 10, 11, 11], chunks=(2,), compression="gzip")
        data.create_dataset("ee_goal_pose", data=poses, chunks=(2, 2, 3, 4), compression="gzip")
        data.create_dataset(
            "active_ee_mask", data=np.ones((4, 2), dtype=bool), chunks=(2, 2), compression="gzip"
        )

    dataset = _dataset_shell(tmp_path, DummyMarvin())

    def collision_statistics_must_not_run():
        raise AssertionError("collision statistics unexpectedly ran")

    dataset.run_collision_statistics = collision_statistics_must_not_run
    dataset.load_data()

    expected = torch.tensor(coefficients, dtype=torch.float32).transpose(1, 2)
    torch.testing.assert_close(dataset.fields["control_points"], expected[:, 1:-1, :])
    torch.testing.assert_close(dataset.fields["q_start"], expected[:, 0, :])
    torch.testing.assert_close(dataset.fields["q_goal"], expected[:, -1, :])
    assert dataset.map_task_id_to_control_points_id == {10: [0, 1], 11: [2, 3]}

    caches = list(Path(tmp_path).glob("*marvin_dual_ee_v2_batched.pickle"))
    assert len(caches) == 1
    with caches[0].open("rb") as stream:
        cached = module.pickle.load(stream)
    assert cached["cache_schema"] == "trajectory_bspline_reload/v2-batched-atomic"
    assert cached["collision_statistics_skipped"] is True


def test_validated_statistics_bypass_rejects_unvalidated_hdf5(tmp_path, monkeypatch):
    from mpd.datasets import trajectories_dataset_bspline as module

    class DummyMarvin:
        pass

    monkeypatch.setattr(module.robots, "RobotMarvinBimanual", DummyMarvin)
    poses = np.zeros((2, 2, 3, 4), dtype=np.float32)
    poses[..., :3, :3] = np.eye(3, dtype=np.float32)
    with h5py.File(tmp_path / "dataset_merged.hdf5", "w") as data:
        data.attrs["validated_splines"] = False
        data.create_dataset("sol_path", shape=(2, 128, 14), dtype=np.float64)
        data.create_dataset("bspline_params_cc", data=np.zeros((2, 14, 8)))
        data.create_dataset("task_id", data=[1, 1])
        data.create_dataset("ee_goal_pose", data=poses)
        data.create_dataset("active_ee_mask", data=np.ones((2, 2), dtype=bool))

    dataset = _dataset_shell(tmp_path, DummyMarvin())
    with pytest.raises(ValueError, match="serialized, validated splines"):
        dataset.load_data()
    assert not list(Path(tmp_path).glob("*.pickle"))


def test_atomic_pickle_failure_preserves_existing_cache(tmp_path, monkeypatch):
    from mpd.datasets import trajectories_dataset_bspline as module

    target = tmp_path / "cache.pickle"
    target.write_bytes(b"existing-cache")

    def fail_dump(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(module.pickle, "dump", fail_dump)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        module._atomic_pickle_dump({"new": True}, target)

    assert target.read_bytes() == b"existing-cache"
    assert not list(tmp_path.glob(".cache.pickle.*.tmp"))


def test_warehouse_entrypoint_enables_bypass_only_after_strict_validation(tmp_path, monkeypatch):
    from scripts.train import train_marvin_warehouse_bimanual as entrypoint

    config = Path(__file__).parents[1] / "scripts/train/cfgs/marvin_bimanual_warehouse_independent.yaml"
    calls = []
    monkeypatch.setattr(entrypoint, "validate_dataset", lambda resolved: calls.append("validated"))

    def fake_run_training(resolved):
        calls.append("training")
        assert resolved["skip_collision_statistics_for_validated_splines"] is True
        return 0

    monkeypatch.setattr(entrypoint, "run_training", fake_run_training)
    assert entrypoint.main(
        [
            "--config",
            str(config),
            "--dataset-subdir",
            "EnvWarehouse-RobotMarvinBimanual-test",
            "--network-variant",
            "D",
            "--results-dir",
            str(tmp_path / "new-run"),
        ]
    ) == 0
    assert calls == ["validated", "training"]
