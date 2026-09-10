#!/usr/bin/env python3
"""Safely merge complete Marvin Warehouse shards while preserving sources."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from scripts.generate_data.generate_marvin_warehouse_bimanual import EE_GOAL_SCHEMA, file_sha256
from scripts.generate_data.launch_generate_marvin_warehouse_bimanual import merge_shards


REQUIRED_FIELDS = {
    "sol_path",
    "q_start",
    "q_goal",
    "task_id",
    "task_mode",
    "direction",
    "active_joint_mask",
    "ee_goal_pose",
    "active_ee_mask",
    "bspline_params_tt",
    "bspline_params_cc",
    "bspline_params_k",
}

# These change execution/provenance, not the geometry or acceptance contract.
OPERATIONAL_CONFIG_KEYS = {
    "seed",
    "output_dir",
    "num_trajectories",
    "workers",
    "tasks_per_shard",
    "worker_lifetime_trajectories",
    "max_worker_restarts_per_shard",
}


def _planning_contract(config):
    contract = deepcopy(config)
    for key in OPERATIONAL_CONFIG_KEYS:
        contract.pop(key, None)
    return contract


def _gap_ranges(sorted_ids, limit=100):
    ranges = []
    for left, right in zip(sorted_ids[:-1], sorted_ids[1:]):
        if right > left + 1:
            ranges.append([int(left + 1), int(right - 1)])
            if len(ranges) == limit:
                break
    return ranges


def validate_shards(root: Path):
    """Return ID-ordered complete shards, a config, and a merge report."""
    root = root.resolve()
    shard_root = root / "shards"
    if not shard_root.is_dir():
        raise FileNotFoundError(f"missing shard directory: {shard_root}")
    candidates = sorted(path for path in shard_root.iterdir() if path.is_dir() and path.name.isdigit())
    if not candidates:
        raise ValueError(f"no numeric shard directories under {shard_root}")

    baseline_contract = None
    baseline_schema = None
    baseline_fields = None
    baseline_shapes = None
    baseline_dtypes = None
    first_config = None
    seen = {}
    records = []

    for shard in candidates:
        required_files = (
            shard / "dataset_merged.hdf5",
            shard / "manifest.yaml",
            shard / "generation_config.yaml",
            shard / "args.yaml",
        )
        missing_files = [path.name for path in required_files if not path.is_file()]
        if missing_files:
            raise ValueError(f"incomplete shard {shard}: missing {missing_files}")

        manifest = yaml.safe_load((shard / "manifest.yaml").read_text())
        config = yaml.safe_load((shard / "generation_config.yaml").read_text())
        if not isinstance(manifest, dict) or not isinstance(config, dict):
            raise ValueError(f"invalid YAML metadata in {shard}")
        dataset_path = shard / "dataset_merged.hdf5"
        actual_hash = file_sha256(dataset_path)
        if actual_hash != manifest.get("dataset_sha256"):
            raise ValueError(f"dataset hash mismatch in {shard}")

        contract = _planning_contract(config)
        schema = (
            manifest.get("schema"),
            manifest.get("ee_goal_schema"),
            manifest.get("scene_version"),
            manifest.get("model_sha256"),
            manifest.get("asset_sha256"),
            tuple(manifest.get("joint_names", ())),
        )
        if schema[1] != EE_GOAL_SCHEMA:
            raise ValueError(f"unsupported EE goal schema in {shard}: {schema[1]}")
        if baseline_contract is None:
            baseline_contract = contract
            baseline_schema = schema
            first_config = config
        elif contract != baseline_contract:
            raise ValueError(f"planning-critical generation config differs in {shard}")
        elif schema != baseline_schema:
            raise ValueError(f"scene/robot/schema provenance differs in {shard}")

        with h5py.File(dataset_path, "r") as data:
            fields = set(data.keys())
            missing_fields = REQUIRED_FIELDS.difference(fields)
            if missing_fields:
                raise ValueError(f"shard {shard} lacks fields: {sorted(missing_fields)}")
            count = len(data["task_id"])
            if count == 0 or count != int(manifest.get("num_trajectories", -1)):
                raise ValueError(f"trajectory count mismatch in {shard}")
            if any(value.shape[0] != count for value in data.values()):
                raise ValueError(f"field row count mismatch in {shard}")
            ids = np.asarray(data["task_id"][:], dtype=np.int64)
            if data["task_id"].ndim != 1 or not np.all(ids[1:] > ids[:-1]):
                raise ValueError(f"task_id must be strictly increasing inside {shard}")
            shapes = {key: data[key].shape[1:] for key in fields}
            dtypes = {key: data[key].dtype for key in fields}
            if baseline_fields is None:
                baseline_fields, baseline_shapes, baseline_dtypes = fields, shapes, dtypes
            elif fields != baseline_fields or shapes != baseline_shapes or dtypes != baseline_dtypes:
                raise ValueError(f"HDF5 fields/shapes/dtypes differ in {shard}")

        for task_id in ids:
            task_id = int(task_id)
            previous = seen.get(task_id)
            if previous is not None:
                raise ValueError(f"duplicate task_id {task_id} in {previous} and {shard}")
            seen[task_id] = shard
        records.append((int(ids[0]), int(ids[-1]), count, shard, actual_hash))

    records.sort(key=lambda item: item[0])
    sorted_ids = np.asarray(sorted(seen), dtype=np.int64)
    span = int(sorted_ids[-1] - sorted_ids[0] + 1)
    report = {
        "schema": "marvin_bimanual_warehouse_standalone_merge/v1",
        "source_shards": len(records),
        "num_trajectories": len(sorted_ids),
        "task_id_min": int(sorted_ids[0]),
        "task_id_max": int(sorted_ids[-1]),
        "missing_task_ids": span - len(sorted_ids),
        "missing_task_id_ranges_first_100": _gap_ranges(sorted_ids),
        "source_shards_preserved": True,
        "sources": [
            {
                "path": str(shard.relative_to(root)),
                "first_task_id": first,
                "last_task_id": last,
                "num_trajectories": count,
                "dataset_sha256": digest,
            }
            for first, last, count, shard, digest in records
        ],
    }
    return [record[3] for record in records], first_config, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Root containing shards/<numeric-id>/ directories")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing merged files")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    target = root / "dataset_merged.hdf5"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing merged dataset: {target}")
    paths, config, report = validate_shards(root)
    print(json.dumps({key: value for key, value in report.items() if key != "sources"}, indent=2))
    if args.dry_run:
        return 0

    merge_shards(root, paths, config)
    manifest_path = root / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["standalone_merge"] = report
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    (root / "merge_report.json").write_text(json.dumps(report, indent=2))
    print(f"merged {report['num_trajectories']} trajectories -> {target}")
    print(f"source shards preserved under {root / 'shards'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
