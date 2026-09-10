#!/usr/bin/env python3
"""Combine complete Marvin Warehouse datasets and assign fresh task IDs."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import yaml

from scripts.generate_data.generate_marvin_warehouse_bimanual import EE_GOAL_SCHEMA, file_sha256


REQUIRED_FIELDS = {
    "sol_path",
    "q_start",
    "q_goal",
    "task_id",
    "planning_time",
    "joint_path_length",
    "task_mode",
    "direction",
    "source_region_left",
    "source_region_right",
    "goal_region_left",
    "goal_region_right",
    "active_joint_mask",
    "ee_goal_pose",
    "active_ee_mask",
    "bspline_params_tt",
    "bspline_params_cc",
    "bspline_params_k",
}
CATEGORICAL_FIELDS = (
    "task_mode",
    "direction",
    "source_region_left",
    "source_region_right",
    "goal_region_left",
    "goal_region_right",
)
OPERATIONAL_CONFIG_KEYS = {
    "seed",
    "output_dir",
    "num_trajectories",
    "workers",
    "tasks_per_shard",
    "worker_lifetime_trajectories",
    "max_worker_restarts_per_shard",
    "combined_dataset",
}
PROVENANCE_KEYS = (
    "schema",
    "ee_goal_schema",
    "scene_version",
    "model_sha256",
    "asset_sha256",
    "joint_names",
)
HDF5_CONTRACT_ATTRS = (
    "joint_names",
    "scene_version",
    "validated_splines",
    "ee_goal_schema",
    "ee_goal_links",
)


def _load_yaml(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _planning_contract(config):
    contract = deepcopy(config)
    for key in OPERATIONAL_CONFIG_KEYS:
        contract.pop(key, None)
    return contract


def _normalise_value(value):
    if isinstance(value, np.ndarray):
        return [_normalise_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _normalise_value(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _dataset_contract(handle):
    return {
        key: {"tail_shape": tuple(value.shape[1:]), "dtype": str(value.dtype)}
        for key, value in sorted(handle.items())
    }


def _hdf5_attributes(handle):
    missing = [key for key in HDF5_CONTRACT_ATTRS if key not in handle.attrs]
    if missing:
        raise ValueError(f"HDF5 lacks required attributes: {missing}")
    return {key: _normalise_value(handle.attrs[key]) for key in HDF5_CONTRACT_ATTRS}


def _resolve_source(path):
    path = Path(path).expanduser().resolve()
    root = path if path.is_dir() else path.parent
    dataset = root / "dataset_merged.hdf5" if path.is_dir() else path
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if dataset.name != "dataset_merged.hdf5":
        raise ValueError(f"expected a dataset_merged.hdf5 input, got {dataset}")
    return root, dataset


def validate_sources(paths, start_task_id=0):
    if start_task_id < 0:
        raise ValueError("start-task-id must be nonnegative")
    resolved = [_resolve_source(path) for path in paths]
    dataset_paths = [dataset for _, dataset in resolved]
    if len(set(dataset_paths)) != len(dataset_paths):
        raise ValueError("the same source dataset was supplied more than once")

    baseline_provenance = None
    baseline_contract = None
    baseline_hdf5_contract = None
    baseline_hdf5_attrs = None
    first_manifest = first_config = first_args = None
    sources = []
    total = 0

    for source_index, (root, dataset) in enumerate(resolved):
        manifest = _load_yaml(root / "manifest.yaml")
        config = _load_yaml(root / "generation_config.yaml")
        args = _load_yaml(root / "args.yaml")
        digest = file_sha256(dataset)
        if digest != manifest.get("dataset_sha256"):
            raise ValueError(f"dataset hash does not match manifest: {dataset}")
        provenance = {key: manifest.get(key) for key in PROVENANCE_KEYS}
        if provenance["schema"] != "marvin_bimanual_warehouse_dataset/v3":
            raise ValueError(f"unsupported dataset schema in {root}: {provenance['schema']}")
        if provenance["ee_goal_schema"] != EE_GOAL_SCHEMA:
            raise ValueError(f"unsupported EE goal schema in {root}: {provenance['ee_goal_schema']}")

        with h5py.File(dataset, "r") as handle:
            fields = set(handle.keys())
            missing = REQUIRED_FIELDS.difference(fields)
            if missing:
                raise ValueError(f"dataset {dataset} lacks fields: {sorted(missing)}")
            count = len(handle["task_id"])
            if count < 1 or count != int(manifest.get("num_trajectories", -1)):
                raise ValueError(f"trajectory count does not match manifest: {dataset}")
            if handle["task_id"].shape != (count,):
                raise ValueError(f"task_id must have shape ({count},): {dataset}")
            if any(value.shape[0] != count for value in handle.values()):
                raise ValueError(f"HDF5 field row counts differ: {dataset}")
            ids = np.asarray(handle["task_id"][:], dtype=np.int64)
            if len(np.unique(ids)) != count:
                raise ValueError(f"source contains duplicate task IDs: {dataset}")
            hdf5_contract = _dataset_contract(handle)
            hdf5_attrs = _hdf5_attributes(handle)

        planning_contract = _planning_contract(config)
        if baseline_provenance is None:
            baseline_provenance = provenance
            baseline_contract = planning_contract
            baseline_hdf5_contract = hdf5_contract
            baseline_hdf5_attrs = hdf5_attrs
            first_manifest, first_config, first_args = manifest, config, args
        else:
            if provenance != baseline_provenance:
                raise ValueError(f"scene/robot/schema provenance differs in {root}")
            if planning_contract != baseline_contract:
                raise ValueError(f"planning-critical generation config differs in {root}")
            if hdf5_contract != baseline_hdf5_contract:
                raise ValueError(f"HDF5 fields/shapes/dtypes differ in {root}")
            if hdf5_attrs != baseline_hdf5_attrs:
                raise ValueError(f"HDF5 training attributes differ in {root}")

        new_first = start_task_id + total
        new_last = new_first + count - 1
        sources.append(
            {
                "source_index": source_index,
                "root": str(root),
                "dataset": str(dataset),
                "dataset_sha256": digest,
                "num_trajectories": count,
                "old_task_id_min": int(ids.min()),
                "old_task_id_max": int(ids.max()),
                "new_task_id_first": new_first,
                "new_task_id_last": new_last,
                "generation_seed": args.get("seed"),
            }
        )
        total += count

    return {
        "resolved": resolved,
        "sources": sources,
        "total": total,
        "manifest": first_manifest,
        "config": first_config,
        "args": first_args,
        "start_task_id": start_task_id,
    }


def _decode_strings(values):
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]


def _write_combined_dataset(staging_root, validated, chunk_size):
    target = staging_root / "dataset_merged.hdf5"
    total = validated["total"]
    task_counts, direction_counts, stats = Counter(), Counter(), Counter()
    region_counts = {key: Counter() for key in CATEGORICAL_FIELDS[2:]}

    first_dataset = validated["resolved"][0][1]
    with h5py.File(first_dataset, "r") as baseline, h5py.File(target, "x") as destination:
        destination.attrs.update(baseline.attrs)
        for key, value in baseline.items():
            destination.create_dataset(
                key,
                shape=(total,) + value.shape[1:],
                dtype=value.dtype,
                compression="gzip",
            )

        output_offset = 0
        for source_record, (root, dataset) in zip(validated["sources"], validated["resolved"]):
            manifest = _load_yaml(root / "manifest.yaml")
            stats.update(manifest.get("stats", {}))
            with h5py.File(dataset, "r") as source:
                count = source_record["num_trajectories"]
                for input_start in range(0, count, chunk_size):
                    input_stop = min(input_start + chunk_size, count)
                    output_start = output_offset + input_start
                    output_stop = output_offset + input_stop
                    for key, value in source.items():
                        if key == "task_id":
                            destination[key][output_start:output_stop] = np.arange(
                                validated["start_task_id"] + output_start,
                                validated["start_task_id"] + output_stop,
                                dtype=destination[key].dtype,
                            )
                        else:
                            destination[key][output_start:output_stop] = value[input_start:input_stop]
                    modes = _decode_strings(source["task_mode"][input_start:input_stop])
                    directions = _decode_strings(source["direction"][input_start:input_stop])
                    task_counts.update(modes)
                    direction_counts.update(directions)
                    for key in region_counts:
                        region_counts[key].update(
                            _decode_strings(source[key][input_start:input_stop])
                        )
            output_offset += source_record["num_trajectories"]

    if output_offset != total:
        raise RuntimeError(f"internal row-count error: wrote {output_offset}/{total}")
    return target, task_counts, direction_counts, region_counts, stats


def combine_sources(output, validated, chunk_size=256):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{output.name}.incomplete-", dir=output.parent) as temporary:
        staging = Path(temporary)
        dataset, task_counts, direction_counts, region_counts, stats = _write_combined_dataset(
            staging, validated, chunk_size
        )
        dataset_digest = file_sha256(dataset)
        sources = validated["sources"]
        total = validated["total"]

        manifest = deepcopy(validated["manifest"])
        for stale_key in ("shards", "standalone_merge", "combined_dataset"):
            manifest.pop(stale_key, None)
        manifest.update(
            num_trajectories=total,
            task_counts=dict(task_counts),
            direction_counts=dict(direction_counts),
            region_counts={key: dict(value) for key, value in region_counts.items()},
            stats=dict(stats),
            dataset_sha256=dataset_digest,
            combined_dataset={
                "schema": "marvin_bimanual_warehouse_combined_dataset/v1",
                "task_ids_reindexed": True,
                "task_id_first": validated["start_task_id"],
                "task_id_last": validated["start_task_id"] + total - 1,
                "source_datasets_preserved": True,
                "sources": sources,
            },
        )

        config = deepcopy(validated["config"])
        config["output_dir"] = str(output)
        config["num_trajectories"] = total
        config["seed"] = None
        config["combined_dataset"] = {
            "task_ids_reindexed": True,
            "source_seeds": [source["generation_seed"] for source in sources],
            "sources": [source["root"] for source in sources],
        }

        args = deepcopy(validated["args"])
        args["seed"] = None
        args["num_trajectories"] = total
        args["task_ids_reindexed"] = True
        args["source_seeds"] = [source["generation_seed"] for source in sources]

        report = {
            "schema": "marvin_bimanual_warehouse_dataset_fusion/v1",
            "output": str(output),
            "num_trajectories": total,
            "task_id_first": validated["start_task_id"],
            "task_id_last": validated["start_task_id"] + total - 1,
            "dataset_sha256": dataset_digest,
            "source_datasets_preserved": True,
            "sources": sources,
        }
        (staging / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        (staging / "generation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (staging / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False))
        (staging / "generation_summary.json").write_text(json.dumps(manifest, indent=2))
        (staging / "merge_report.json").write_text(json.dumps(report, indent=2))
        os.replace(staging, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="Two or more dataset roots (or dataset_merged.hdf5 files), in output order",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-task-id", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true", help="validate without writing output")
    args = parser.parse_args(argv)
    if len(args.sources) < 2:
        raise ValueError("at least two source datasets are required")
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")

    validated = validate_sources(args.sources, args.start_task_id)
    preview = {
        "output": str(args.output_dir.expanduser().resolve()),
        "num_trajectories": validated["total"],
        "task_id_first": validated["start_task_id"],
        "task_id_last": validated["start_task_id"] + validated["total"] - 1,
        "sources": validated["sources"],
    }
    print(json.dumps(preview, indent=2))
    if args.dry_run:
        return 0

    report = combine_sources(args.output_dir, validated, args.chunk_size)
    print(f"combined {report['num_trajectories']} trajectories -> {report['output']}")
    print("source datasets were preserved; output task IDs are globally unique and contiguous")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
