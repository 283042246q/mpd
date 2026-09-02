"""Streaming HDF5 writer for canonical Space-Time MPD samples."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import h5py
import numpy as np

from mpd.datasets.spacetime_schema import CANONICAL_SAMPLE_FIELDS, SCHEMA_VERSION


def _concrete_shape(symbolic: Sequence[str], dimensions: Mapping[str, int]) -> Tuple[int, ...]:
    return tuple(int(dimensions[name]) for name in symbolic)


class SpaceTimeShardWriter:
    """Append-only writer that never accumulates the full dataset in RAM."""

    def __init__(
        self,
        path: Path,
        *,
        dimensions: Mapping[str, int],
        robot_hashes: Mapping[str, str],
        compression: str = "lzf",
        chunk_rows: int = 256,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        self._file = h5py.File(str(self.path), "w")
        self._file.attrs["schema_version"] = SCHEMA_VERSION
        for key, value in robot_hashes.items():
            self._file.attrs[key] = value
        self._datasets: Dict[str, h5py.Dataset] = {}
        for name, (dtype, symbolic_shape) in CANONICAL_SAMPLE_FIELDS.items():
            sample_shape = _concrete_shape(symbolic_shape, dimensions)
            chunks = (max(1, int(chunk_rows)),) + sample_shape
            self._datasets[name] = self._file.create_dataset(
                name,
                shape=(0,) + sample_shape,
                maxshape=(None,) + sample_shape,
                chunks=chunks,
                compression=compression,
                dtype=dtype,
            )
        self._file["spatial"].attrs["degree"] = int(dimensions["spatial_degree"])
        self._file["timing"].attrs["degree"] = int(dimensions["timing_degree"])
        self._file["timing"].attrs["representation"] = "dt_ds_softplus_v1"
        scenes = self._file.require_group("scenes")
        scenes.create_dataset("id", shape=(0,), maxshape=(None,), dtype=np.int64)
        scenes.create_dataset("horizon_s", shape=(0,), maxshape=(None,), dtype=np.float32)
        scenes.create_dataset("object_count", shape=(0,), maxshape=(None,), dtype=np.int32)

    def __enter__(self) -> "SpaceTimeShardWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def size(self) -> int:
        return int(next(iter(self._datasets.values())).shape[0])

    def set_attributes(self, attributes: Mapping[str, object]) -> None:
        for name, value in attributes.items():
            self._file.attrs[name] = value

    def append(self, records: Mapping[str, np.ndarray]) -> None:
        missing = sorted(set(self._datasets) - set(records))
        extra = sorted(set(records) - set(self._datasets))
        if missing or extra:
            raise ValueError(f"canonical record fields mismatch; missing={missing}, extra={extra}")
        arrays = {name: np.asarray(records[name]) for name in self._datasets}
        row_counts = {array.shape[0] for array in arrays.values()}
        if len(row_counts) != 1:
            raise ValueError(f"all fields must have the same row count, got {sorted(row_counts)}")
        row_count = row_counts.pop()
        if row_count == 0:
            return
        for name, dataset in self._datasets.items():
            if arrays[name].shape[1:] != dataset.shape[1:]:
                raise ValueError(
                    f"field {name} has sample shape {arrays[name].shape[1:]}, "
                    f"expected {dataset.shape[1:]}"
                )
        old_size = self.size
        new_size = old_size + row_count
        for name, dataset in self._datasets.items():
            array = arrays[name]
            dataset.resize(new_size, axis=0)
            dataset[old_size:new_size] = array.astype(dataset.dtype, copy=False)

    def close(self) -> None:
        if getattr(self, "_file", None) is not None:
            self._file.attrs["num_samples"] = self.size
            self._file.flush()
            self._file.close()
            self._file = None


def write_grouped_splits(
    split_root: Path,
    base_path_ids: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.9,
    validation_fraction: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Split unique base paths once so retiming variants cannot leak."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation_fraction leaves no test split")
    unique_ids = np.unique(np.asarray(base_path_ids, dtype=np.int64))
    # SplitMix64 gives each base path a stable assignment. Adding later shards
    # cannot move an existing path between train/val/test.
    unsigned = unique_ids.astype(np.uint64, copy=False) + np.uint64(seed)
    unsigned = (unsigned ^ (unsigned >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    unsigned = (unsigned ^ (unsigned >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    unsigned = unsigned ^ (unsigned >> np.uint64(31))
    uniform = unsigned.astype(np.float64) / float(np.iinfo(np.uint64).max)
    train_mask = uniform < train_fraction
    validation_mask = (uniform >= train_fraction) & (
        uniform < train_fraction + validation_fraction
    )
    splits = {
        "train": np.sort(unique_ids[train_mask]),
        "val": np.sort(unique_ids[validation_mask]),
        "test": np.sort(unique_ids[~(train_mask | validation_mask)]),
    }
    split_root.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        np.save(str(split_root / f"{name}_base_path_ids.npy"), values, allow_pickle=False)
    return splits
