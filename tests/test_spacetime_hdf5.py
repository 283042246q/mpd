import h5py
import numpy as np

from mpd.datasets.spacetime_hdf5 import SpaceTimeShardWriter, write_grouped_splits
from mpd.datasets.spacetime_schema import CANONICAL_SAMPLE_FIELDS, SCHEMA_VERSION


def _records(rows):
    dimensions = {"H_P": 29, "D": 7, "K": 8, "H_T": 128}
    result = {}
    for name, (dtype, shape) in CANONICAL_SAMPLE_FIELDS.items():
        concrete = tuple(dimensions[value] for value in shape)
        result[name] = np.zeros((rows,) + concrete, dtype=dtype)
    result["index/base_path_id"] = np.asarray([10, 10, 20][:rows], dtype=np.int64)
    result["quality/accepted"][:] = True
    return result


def test_spacetime_shard_writer_roundtrip(tmp_path):
    path = tmp_path / "part-00000.hdf5"
    dimensions = {
        "H_P": 29,
        "D": 7,
        "K": 8,
        "H_T": 128,
        "spatial_degree": 5,
        "timing_degree": 3,
    }
    with SpaceTimeShardWriter(
        path, dimensions=dimensions, robot_hashes={"urdf_sha256": "abc"}
    ) as writer:
        writer.append(_records(2))
        writer.append(_records(1))
        assert writer.size == 3

    with h5py.File(path, "r") as result:
        assert result.attrs["schema_version"] == SCHEMA_VERSION
        assert result.attrs["num_samples"] == 3
        assert result["spatial/control_points"].shape == (3, 29, 7)
        assert result["timing/control_points"].shape == (3, 8)
        assert result["scenes/id"].shape == (0,)


def test_grouped_splits_do_not_leak_retiming_variants(tmp_path):
    ids = np.repeat(np.arange(100, dtype=np.int64), 6)
    splits = write_grouped_splits(tmp_path, ids, seed=7)

    train = set(splits["train"].tolist())
    val = set(splits["val"].tolist())
    test = set(splits["test"].tolist())
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(100))
