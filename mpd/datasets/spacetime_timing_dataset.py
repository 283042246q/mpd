"""Independent HDF5 dataset views for conditional TimingDiffusion training.

This loader reads the canonical Space-Time MPD shards directly.  It does not
instantiate the legacy trajectory dataset, a planning environment, or the
spatial MPD training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from mpd.datasets.spacetime_schema import SCHEMA_VERSION, sha256_file
from mpd.parametric_trajectory.timing_fitting import encode_timing_control_points


TIMING_REPRESENTATIONS = ("c", "tau_r")


def _stable_split_values(base_path_ids: np.ndarray, seed: int) -> np.ndarray:
    """Match the SplitMix64 assignment used by canonical split generation."""

    if seed < 0:
        raise ValueError("split seed must be non-negative")
    values = np.asarray(base_path_ids, dtype=np.int64)
    with np.errstate(over="ignore"):
        unsigned = values.astype(np.uint64, copy=False) + np.uint64(seed)
        unsigned = (unsigned ^ (unsigned >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        unsigned = (unsigned ^ (unsigned >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        unsigned = unsigned ^ (unsigned >> np.uint64(31))
    return unsigned.astype(np.float64) / float(np.iinfo(np.uint64).max)


def split_mask_from_base_path_ids(
    base_path_ids: np.ndarray,
    split: str,
    *,
    seed: int,
    train_fraction: float = 0.9,
    validation_fraction: float = 0.05,
) -> np.ndarray:
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split: {split}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation_fraction leaves no test split")
    values = _stable_split_values(base_path_ids, seed)
    if split == "train":
        return values < train_fraction
    if split == "val":
        return (values >= train_fraction) & (
            values < train_fraction + validation_fraction
        )
    return values >= train_fraction + validation_fraction


@dataclass(frozen=True)
class TimingNormalization:
    path_mean: np.ndarray
    path_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    count: int
    representation: str

    def __post_init__(self) -> None:
        if self.representation not in TIMING_REPRESENTATIONS:
            raise ValueError(f"unknown timing representation: {self.representation}")
        if self.path_mean.ndim != 1 or self.path_std.shape != self.path_mean.shape:
            raise ValueError("path normalization must contain equal-length vectors")
        if self.target_mean.shape != (6,) or self.target_std.shape != (6,):
            raise ValueError("timing normalization must contain six-vectors")
        if np.any(self.path_std <= 0.0) or np.any(self.target_std <= 0.0):
            raise ValueError("normalization standard deviations must be positive")

    def to_dict(self) -> Dict[str, object]:
        return {
            "path_mean": self.path_mean.tolist(),
            "path_std": self.path_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
            "count": int(self.count),
            "representation": self.representation,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TimingNormalization":
        return cls(
            path_mean=np.asarray(value["path_mean"], dtype=np.float32),
            path_std=np.asarray(value["path_std"], dtype=np.float32),
            target_mean=np.asarray(value["target_mean"], dtype=np.float32),
            target_std=np.asarray(value["target_std"], dtype=np.float32),
            count=int(value["count"]),
            representation=str(value["representation"]),
        )


@dataclass(frozen=True)
class _ShardSelection:
    dataset_root: Path
    shard_path: Path
    rows: np.ndarray


def _manifest_contract(manifest: Mapping[str, object]) -> Dict[str, object]:
    robot = manifest["robot"]
    spatial = manifest["spatial_spline"]
    timing = manifest["timing_spline"]
    return {
        "robot_name": robot["name"],
        "robot_hashes": tuple(
            robot[name]
            for name in (
                "urdf_sha256",
                "joint_limits_sha256",
                "collision_spheres_sha256",
                "collision_parent_bounds_sha256",
            )
        ),
        "dof": int(robot["dof"]),
        "spatial_num_control_points": int(spatial["num_control_points"]),
        "spatial_degree": int(spatial["degree"]),
        "timing_num_control_points": int(timing["num_control_points"]),
        "timing_degree": int(timing["degree"]),
    }


class SpaceTimeTimingDataset(Dataset):
    """A lazy conditional view yielding full ``P`` and a six-value timing target."""

    def __init__(
        self,
        dataset_roots: Sequence[Path],
        *,
        split: str,
        representation: str,
        normalization: Optional[TimingNormalization] = None,
        split_seed: int = 1726484688,
        train_fraction: float = 0.9,
        validation_fraction: float = 0.05,
        allow_hash_split_fallback: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        if representation not in TIMING_REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {TIMING_REPRESENTATIONS}, got {representation}"
            )
        if not dataset_roots:
            raise ValueError("at least one dataset root is required")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.split = split
        self.representation = representation
        self.normalization = normalization
        self.split_seed = int(split_seed)
        self.train_fraction = float(train_fraction)
        self.validation_fraction = float(validation_fraction)
        self.allow_hash_split_fallback = bool(allow_hash_split_fallback)
        self._handles: Dict[Path, h5py.File] = {}
        self._selections: List[_ShardSelection] = []
        self._manifests: List[Dict[str, object]] = []
        self._manifest_paths: List[Path] = []
        reference_contract: Optional[Dict[str, object]] = None
        remaining = max_samples

        for root_value in dataset_roots:
            root = Path(root_value).resolve()
            manifest_path = root / "manifest.yaml"
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = yaml.safe_load(stream)
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema in {manifest_path}: {manifest.get('schema_version')}"
                )
            contract = _manifest_contract(manifest)
            if reference_contract is None:
                reference_contract = contract
            elif contract != reference_contract:
                raise ValueError(
                    f"dataset contract mismatch between roots: {reference_contract} != {contract}"
                )
            self._manifests.append(manifest)
            self._manifest_paths.append(manifest_path)
            split_ids = self._load_split_ids(root, split)
            for shard_path in sorted((root / "shards").glob("part-*.hdf5")):
                with h5py.File(str(shard_path), "r") as shard:
                    selected = self._select_rows(
                        shard,
                        split=split,
                        split_ids=split_ids,
                        split_seed=self.split_seed,
                    )
                if remaining is not None:
                    selected = selected[:remaining]
                    remaining -= selected.size
                if selected.size:
                    self._selections.append(
                        _ShardSelection(root, shard_path.resolve(), selected)
                    )
                if remaining == 0:
                    break
            if remaining == 0:
                break

        assert reference_contract is not None
        self.contract = reference_contract
        self.dof = int(reference_contract["dof"])
        self.num_spatial_control_points = int(
            reference_contract["spatial_num_control_points"]
        )
        self._cumulative = np.cumsum(
            np.asarray([selection.rows.size for selection in self._selections], dtype=np.int64)
        )
        if len(self) == 0:
            raise ValueError(
                f"no {split} samples found for representation={representation}"
            )
        if normalization is not None:
            self.set_normalization(normalization)

    def _load_split_ids(self, root: Path, split: str) -> Optional[np.ndarray]:
        split_path = root / "splits" / f"{split}_base_path_ids.npy"
        if split_path.is_file():
            return np.asarray(np.load(str(split_path), allow_pickle=False), dtype=np.int64)
        if not self.allow_hash_split_fallback:
            raise FileNotFoundError(
                f"missing canonical split {split_path}; use allow_hash_split_fallback only "
                "for incomplete/smoke datasets"
            )
        return None

    def _select_rows(
        self,
        shard: h5py.File,
        *,
        split: str,
        split_ids: Optional[np.ndarray],
        split_seed: int,
    ) -> np.ndarray:
        required = {
            "spatial/control_points",
            "timing/control_points",
            "quality/accepted",
            "index/base_path_id",
        }
        if self.representation == "tau_r":
            required.update(
                {
                    "timing/tau",
                    "timing/shape_control_points",
                    "quality/duration_bounds_valid",
                }
            )
        missing = sorted(name for name in required if name not in shard)
        if missing:
            suffix = (
                "; run scripts/spacetime_data/augment_normalized_timing.py first"
                if self.representation == "tau_r"
                else ""
            )
            raise ValueError(f"shard is missing timing training fields {missing}{suffix}")
        base_path_ids = np.asarray(shard["index/base_path_id"][:], dtype=np.int64)
        if split_ids is None:
            split_mask = split_mask_from_base_path_ids(
                base_path_ids,
                split,
                seed=split_seed,
                train_fraction=self.train_fraction,
                validation_fraction=self.validation_fraction,
            )
        else:
            split_mask = np.isin(base_path_ids, split_ids)
        mask = split_mask & np.asarray(shard["quality/accepted"][:], dtype=np.bool_)
        if self.representation == "tau_r":
            mask &= np.asarray(
                shard["quality/duration_bounds_valid"][:], dtype=np.bool_
            )
            mask &= np.isfinite(np.asarray(shard["timing/tau"][:]))
            shape = np.asarray(shard["timing/shape_control_points"][:])
            mask &= np.all(np.isfinite(shape), axis=1)
        return np.flatnonzero(mask).astype(np.int64, copy=False)

    @property
    def manifests(self) -> Tuple[Mapping[str, object], ...]:
        return tuple(self._manifests)

    @property
    def identity(self) -> Dict[str, object]:
        return {
            "dataset_roots": sorted(
                {str(selection.dataset_root) for selection in self._selections}
            ),
            "manifest_sha256": [sha256_file(path) for path in self._manifest_paths],
            "contract": self.contract,
            "representation": self.representation,
            "split": self.split,
            "num_samples": len(self),
        }

    def __len__(self) -> int:
        return int(self._cumulative[-1]) if self._cumulative.size else 0

    def _locate(self, index: int) -> Tuple[_ShardSelection, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = int(np.searchsorted(self._cumulative, index, side="right"))
        previous = int(self._cumulative[shard_index - 1]) if shard_index else 0
        selection = self._selections[shard_index]
        return selection, int(selection.rows[index - previous])

    def _open(self, path: Path) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(str(path), "r")
            self._handles[path] = handle
        return handle

    def _read_target(self, shard: h5py.File, row: int) -> np.ndarray:
        if self.representation == "c":
            return encode_timing_control_points(
                np.asarray(shard["timing/control_points"][row], dtype=np.float64)
            ).astype(np.float32)
        tau = float(shard["timing/tau"][row])
        shape = np.asarray(
            shard["timing/shape_control_points"][row], dtype=np.float32
        )
        return np.concatenate((np.asarray([tau], dtype=np.float32), shape))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        selection, row = self._locate(index)
        shard = self._open(selection.shard_path)
        path = np.asarray(shard["spatial/control_points"][row], dtype=np.float32)
        target = self._read_target(shard, row)
        if self.normalization is not None:
            path = (path - self.normalization.path_mean[None, :]) / (
                self.normalization.path_std[None, :]
            )
            target = (target - self.normalization.target_mean) / (
                self.normalization.target_std
            )
        return {
            "path": torch.from_numpy(np.ascontiguousarray(path)),
            "timing": torch.from_numpy(np.ascontiguousarray(target)),
            "base_path_id": torch.tensor(
                int(shard["index/base_path_id"][row]), dtype=torch.int64
            ),
            "variant_id": torch.tensor(
                int(shard["index/variant_id"][row]), dtype=torch.int64
            ),
            "task_id": torch.tensor(int(shard["index/task_id"][row]), dtype=torch.int64),
        }

    def set_normalization(self, normalization: TimingNormalization) -> None:
        if normalization.representation != self.representation:
            raise ValueError(
                "normalization representation does not match dataset: "
                f"{normalization.representation} != {self.representation}"
            )
        if normalization.path_mean.shape != (self.dof,):
            raise ValueError("normalization path dimension does not match dataset")
        self.normalization = normalization

    def compute_normalization(
        self, *, chunk_rows: int = 1024, min_std: float = 1e-6
    ) -> TimingNormalization:
        if chunk_rows <= 0 or min_std <= 0.0:
            raise ValueError("chunk_rows and min_std must be positive")
        path_sum = np.zeros(self.dof, dtype=np.float64)
        path_square_sum = np.zeros(self.dof, dtype=np.float64)
        target_sum = np.zeros(6, dtype=np.float64)
        target_square_sum = np.zeros(6, dtype=np.float64)
        path_count = 0
        target_count = 0
        for selection in self._selections:
            with h5py.File(str(selection.shard_path), "r") as shard:
                for start in range(0, selection.rows.size, chunk_rows):
                    rows = selection.rows[start : start + chunk_rows]
                    paths = np.asarray(
                        shard["spatial/control_points"][rows], dtype=np.float64
                    )
                    if self.representation == "c":
                        full = np.asarray(
                            shard["timing/control_points"][rows], dtype=np.float64
                        )
                        targets = full[:, [0, 2, 3, 4, 5, 7]]
                    else:
                        tau = np.asarray(shard["timing/tau"][rows], dtype=np.float64)
                        shape = np.asarray(
                            shard["timing/shape_control_points"][rows], dtype=np.float64
                        )
                        targets = np.concatenate((tau[:, None], shape), axis=1)
                    path_sum += np.sum(paths, axis=(0, 1))
                    path_square_sum += np.sum(np.square(paths), axis=(0, 1))
                    path_count += paths.shape[0] * paths.shape[1]
                    target_sum += np.sum(targets, axis=0)
                    target_square_sum += np.sum(np.square(targets), axis=0)
                    target_count += targets.shape[0]
        path_mean = path_sum / path_count
        target_mean = target_sum / target_count
        path_variance = np.maximum(path_square_sum / path_count - np.square(path_mean), 0.0)
        target_variance = np.maximum(
            target_square_sum / target_count - np.square(target_mean), 0.0
        )
        return TimingNormalization(
            path_mean=path_mean.astype(np.float32),
            path_std=np.maximum(np.sqrt(path_variance), min_std).astype(np.float32),
            target_mean=target_mean.astype(np.float32),
            target_std=np.maximum(np.sqrt(target_variance), min_std).astype(np.float32),
            count=target_count,
            representation=self.representation,
        )

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        if hasattr(self, "_handles"):
            self.close()


def write_normalization_json(path: Path, normalization: TimingNormalization) -> None:
    Path(path).write_text(
        json.dumps(normalization.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
