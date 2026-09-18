from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.generate_data.soak_marvin_pipeline_stage import (
    MINIMUM_SOAK_SECONDS,
    build_parser,
    load_fixed_inputs,
)


def _write_dataset(path):
    modes = np.asarray(
        ["dual_independent", "dual_independent", "left_only", "right_only"],
        dtype=h5py.string_dtype(),
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("task_mode", data=modes)
        handle.create_dataset("q_start", data=np.zeros((4, 14)))
        handle.create_dataset("q_goal", data=np.ones((4, 14)))
        handle.create_dataset("sol_path", data=np.zeros((4, 8, 14)))
        handle.create_dataset("bspline_params_tt", data=np.zeros((4, 28)))
        handle.create_dataset("bspline_params_cc", data=np.zeros((4, 14, 22)))
        handle.create_dataset("bspline_params_k", data=np.full(4, 5))


def test_load_fixed_inputs_selects_exact_homogeneous_batches(tmp_path):
    dataset = tmp_path / "dataset.hdf5"
    _write_dataset(dataset)
    config = {
        "gpu_query_batch_size_dual": 2,
        "gpu_query_batch_size_left": 1,
        "gpu_query_batch_size_right": 1,
    }

    endpoints, rows, trajectories = load_fixed_inputs(
        dataset, config, trajectory_rows=3
    )

    assert endpoints["dual_independent"][0].shape == (2, 14)
    assert rows["left_only"].tolist() == [2]
    assert rows["right_only"].tolist() == [3]
    assert [item["row"] for item in trajectories] == [0, 1, 2]


def test_load_fixed_inputs_rejects_insufficient_mode_rows(tmp_path):
    dataset = tmp_path / "dataset.hdf5"
    _write_dataset(dataset)
    config = {
        "gpu_query_batch_size_dual": 3,
        "gpu_query_batch_size_left": 1,
        "gpu_query_batch_size_right": 1,
    }

    with pytest.raises(ValueError, match="dual_independent"):
        load_fixed_inputs(dataset, config)


def test_default_soak_duration_is_at_least_40_minutes(tmp_path):
    args = build_parser().parse_args(
        ["--mode", "endpoint", "--output-dir", str(tmp_path)]
    )
    assert args.duration_seconds >= MINIMUM_SOAK_SECONDS
