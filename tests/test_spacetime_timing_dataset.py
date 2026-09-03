from pathlib import Path

import h5py
import numpy as np
import yaml

from mpd.datasets.spacetime_schema import SCHEMA_VERSION
from mpd.datasets.spacetime_timing_dataset import (
    SpaceTimeTimingDataset,
    TimingNormalization,
    split_mask_from_base_path_ids,
)


def _write_dataset(root: Path) -> Path:
    root.mkdir()
    (root / "shards").mkdir()
    (root / "splits").mkdir()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "robot": {
            "name": "RobotPanda",
            "dof": 2,
            "urdf_sha256": "u",
            "joint_limits_sha256": "j",
            "collision_spheres_sha256": "s",
            "collision_parent_bounds_sha256": "p",
        },
        "spatial_spline": {"num_control_points": 4, "degree": 3},
        "timing_spline": {"num_control_points": 8, "degree": 3},
    }
    (root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    np.save(root / "splits" / "train_base_path_ids.npy", np.asarray([1, 2]))
    np.save(root / "splits" / "val_base_path_ids.npy", np.asarray([3]))
    np.save(root / "splits" / "test_base_path_ids.npy", np.asarray([4]))
    rows = 6
    base_ids = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int64)
    paths = np.arange(rows * 4 * 2, dtype=np.float32).reshape(rows, 4, 2) / 10.0
    unique_c = np.arange(rows * 6, dtype=np.float32).reshape(rows, 6) / 10.0
    full_c = unique_c[:, [0, 0, 1, 2, 3, 4, 5, 5]]
    shape = np.arange(rows * 5, dtype=np.float32).reshape(rows, 5) / 20.0
    with h5py.File(root / "shards" / "part-0000000.hdf5", "w") as shard:
        shard.create_dataset("spatial/control_points", data=paths)
        shard.create_dataset("timing/control_points", data=full_c)
        shard.create_dataset("timing/tau", data=np.linspace(-1.0, 1.0, rows))
        tau_modes = np.tile(np.asarray([-4.0, -2.0, 0.0], dtype=np.float32), (rows, 1))
        shard.create_dataset("timing/tau_modes", data=tau_modes)
        shard.create_dataset("timing/shape_control_points", data=shape)
        shard.create_dataset("quality/accepted", data=np.ones(rows, dtype=np.bool_))
        bounds_valid = np.ones(rows, dtype=np.bool_)
        bounds_valid[2] = False
        shard.create_dataset("quality/duration_bounds_valid", data=bounds_valid)
        mode_valid = np.ones((rows, 3), dtype=np.bool_)
        mode_valid[1, 2] = False
        shard.create_dataset("quality/tau_r_mode_valid", data=mode_valid)
        shard.create_dataset("index/base_path_id", data=base_ids)
        shard.create_dataset("index/variant_id", data=np.arange(rows, dtype=np.int16))
        shard.create_dataset("index/task_id", data=np.arange(rows, dtype=np.int64) + 10)
    return root


def test_c_view_reads_canonical_splits_and_computes_normalization(tmp_path):
    root = _write_dataset(tmp_path / "dataset")
    dataset = SpaceTimeTimingDataset([root], split="train", representation="c")

    assert len(dataset) == 4
    np.testing.assert_allclose(dataset[0]["timing"].numpy(), np.arange(6) / 10.0)
    normalization = dataset.compute_normalization(chunk_rows=2)
    assert normalization.count == 4
    assert normalization.path_mean.shape == (2,)
    assert normalization.target_mean.shape == (6,)
    dataset.set_normalization(normalization)
    expected = (np.arange(6) / 10.0 - normalization.target_mean) / (
        normalization.target_std
    )
    np.testing.assert_allclose(dataset[0]["timing"].numpy(), expected, rtol=1e-6)


def test_tau_r_view_filters_invalid_duration_bounds(tmp_path):
    root = _write_dataset(tmp_path / "dataset")
    dataset = SpaceTimeTimingDataset([root], split="train", representation="tau_r")

    assert len(dataset) == 8
    first = dataset[0]
    assert first["timing"].shape == (6,)
    assert np.isclose(first["timing"][0].item(), -4.0)
    assert first["timing_mode_index"].item() == 0
    normalization = dataset.compute_normalization()
    restored = TimingNormalization.from_dict(normalization.to_dict())
    np.testing.assert_array_equal(restored.target_mean, normalization.target_mean)
    assert restored.representation == "tau_r"


def test_hash_fallback_keeps_variants_of_each_base_path_in_one_split():
    ids = np.repeat(np.arange(100, dtype=np.int64), 7)
    memberships = {
        split: split_mask_from_base_path_ids(ids, split, seed=42)
        for split in ("train", "val", "test")
    }
    assert np.all(sum(mask.astype(np.int8) for mask in memberships.values()) == 1)
    for base_id in np.unique(ids):
        rows = ids == base_id
        assert sum(bool(np.any(mask[rows])) for mask in memberships.values()) == 1


def test_missing_split_requires_explicit_smoke_fallback(tmp_path):
    root = _write_dataset(tmp_path / "dataset")
    for path in (root / "splits").glob("*.npy"):
        path.unlink()

    try:
        SpaceTimeTimingDataset([root], split="train", representation="c")
    except FileNotFoundError as error:
        assert "allow_hash_split_fallback" in str(error)
    else:
        raise AssertionError("missing formal split must not be silently accepted")

    dataset = SpaceTimeTimingDataset(
        [root],
        split="train",
        representation="c",
        allow_hash_split_fallback=True,
    )
    assert len(dataset) > 0
