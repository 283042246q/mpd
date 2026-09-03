from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from mpd.datasets.spacetime_schema import SCHEMA_VERSION
from mpd.timing_training.trainer import CHECKPOINT_VERSION, train_timing_diffusion


def _training_dataset(root: Path) -> Path:
    root.mkdir()
    (root / "shards").mkdir()
    (root / "splits").mkdir()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "robot": {
            "name": "TestRobot",
            "dof": 2,
            "urdf_sha256": "u",
            "joint_limits_sha256": "j",
            "collision_spheres_sha256": "s",
            "collision_parent_bounds_sha256": "p",
        },
        "spatial_spline": {"num_control_points": 12, "degree": 5},
        "timing_spline": {"num_control_points": 8, "degree": 3},
    }
    (root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    np.save(root / "splits" / "train_base_path_ids.npy", np.asarray([1, 2]))
    np.save(root / "splits" / "val_base_path_ids.npy", np.asarray([3]))
    np.save(root / "splits" / "test_base_path_ids.npy", np.asarray([4]))
    rows = 8
    rng = np.random.default_rng(7)
    paths = rng.normal(size=(rows, 12, 2)).astype(np.float32)
    unique_c = rng.normal(size=(rows, 6)).astype(np.float32)
    full_c = unique_c[:, [0, 0, 1, 2, 3, 4, 5, 5]]
    with h5py.File(root / "shards" / "part-0000000.hdf5", "w") as shard:
        shard.create_dataset("spatial/control_points", data=paths)
        shard.create_dataset("timing/control_points", data=full_c)
        shard.create_dataset("timing/tau", data=np.linspace(-1.0, 1.0, rows))
        shard.create_dataset(
            "timing/shape_control_points",
            data=rng.normal(size=(rows, 5)).astype(np.float32),
        )
        shard.create_dataset("quality/accepted", data=np.ones(rows, dtype=np.bool_))
        shard.create_dataset(
            "quality/duration_bounds_valid", data=np.ones(rows, dtype=np.bool_)
        )
        shard.create_dataset(
            "index/base_path_id", data=np.repeat(np.arange(1, 5), 2)
        )
        shard.create_dataset("index/variant_id", data=np.tile([0, 1], 4))
        shard.create_dataset("index/task_id", data=np.arange(rows))
    return root


@pytest.mark.parametrize("representation", ["c", "tau_r"])
def test_standalone_training_writes_self_describing_checkpoint(
    tmp_path, representation
):
    dataset_root = _training_dataset(tmp_path / f"dataset-{representation}")
    output_dir = tmp_path / f"output-{representation}"

    result = train_timing_diffusion(
        environment="TestEnv",
        representation=representation,
        dataset_roots=[dataset_root],
        output_dir=output_dir,
        data_config={"batch_size": 2, "num_workers": 0},
        model_config={
            "num_phase_points": 16,
            "path_width": 8,
            "path_embedding_dim": 12,
            "hidden_dim": 16,
            "time_embedding_dim": 8,
            "num_residual_blocks": 1,
        },
        diffusion_config={"num_diffusion_steps": 4, "clip_clean": 5.0},
        training_config={
            "seed": 3,
            "device": "cpu",
            "max_steps": 2,
            "learning_rate": 1e-3,
            "use_amp": False,
            "log_every": 1,
            "validate_every": 1,
            "validation_batches": 1,
            "checkpoint_every": 1,
        },
    )

    assert result.final_step == 2
    assert np.isfinite(result.train_loss)
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_version"] == CHECKPOINT_VERSION
    assert checkpoint["representation"] == representation
    assert checkpoint["environment"] == "TestEnv"
    assert checkpoint["step"] == 2
    assert checkpoint["normalization"]["representation"] == representation
    assert checkpoint["model_config"]["dof"] == 2
    assert (output_dir / "resolved_config.yaml").is_file()
    assert (output_dir / "normalization.json").is_file()
    assert (output_dir / "metrics.jsonl").is_file()

    if representation == "c":
        resumed_training = {
            "seed": 3,
            "device": "cpu",
            "max_steps": 3,
            "learning_rate": 1e-3,
            "use_amp": False,
            "log_every": 1,
            "validate_every": 1,
            "validation_batches": 1,
            "checkpoint_every": 1,
        }
        resumed = train_timing_diffusion(
            environment="TestEnv",
            representation=representation,
            dataset_roots=[dataset_root],
            output_dir=output_dir,
            data_config={"batch_size": 2, "num_workers": 0},
            model_config={
                "num_phase_points": 16,
                "path_width": 8,
                "path_embedding_dim": 12,
                "hidden_dim": 16,
                "time_embedding_dim": 8,
                "num_residual_blocks": 1,
            },
            diffusion_config={"num_diffusion_steps": 4, "clip_clean": 5.0},
            training_config=resumed_training,
            resume_checkpoint=checkpoint_path,
        )
        assert resumed.final_step == 3
