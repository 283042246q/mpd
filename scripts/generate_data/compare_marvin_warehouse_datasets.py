#!/usr/bin/env python3
"""Compare only same-numbered Marvin Warehouse shards from two computers."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import wasserstein_distance
import yaml


OPERATIONAL_CONFIG_KEYS = {
    "seed",
    "output_dir",
    "num_trajectories",
    "workers",
    "tasks_per_shard",
    "worker_lifetime_trajectories",
    "max_worker_restarts_per_shard",
}
CATEGORICAL_FIELDS = (
    "task_mode",
    "direction",
    "source_region_left",
    "source_region_right",
    "goal_region_left",
    "goal_region_right",
)
PATH_VALIDATION_TIME_FIELDS = (
    "path_resample_seconds",
    "path_densify_seconds",
    "path_torch_audit_seconds",
    "path_pybullet_audit_seconds",
)


def load_yaml_if_present(path: Path):
    if not path.is_file():
        return None
    value = yaml.safe_load(path.read_text())
    return value if isinstance(value, dict) else None


def planning_contract(config):
    if config is None:
        return None
    contract = deepcopy(config)
    for key in OPERATIONAL_CONFIG_KEYS:
        contract.pop(key, None)
    return contract


def flatten_dict(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_dict(value[key], child))
    else:
        result[prefix] = value
    return result


def dict_differences(left, right):
    left_flat, right_flat = flatten_dict(left or {}), flatten_dict(right or {})
    differences = {}
    for key in sorted(set(left_flat) | set(right_flat)):
        if left_flat.get(key) != right_flat.get(key):
            differences[key] = {"a": left_flat.get(key), "b": right_flat.get(key)}
    return differences


def row_keys(values, decimals=None):
    values = np.asarray(values, dtype=np.float64)
    if decimals is not None:
        values = np.round(values, decimals=decimals)
    values = np.ascontiguousarray(values)
    width = values.dtype.itemsize * values.shape[1]
    return values.view(np.dtype((np.void, width))).reshape(-1)


def overlap_metrics(keys_a, keys_b):
    unique_a, counts_a = np.unique(keys_a, return_counts=True)
    unique_b, counts_b = np.unique(keys_b, return_counts=True)
    common, indices_a, indices_b = np.intersect1d(unique_a, unique_b, return_indices=True)
    paired = int(np.minimum(counts_a[indices_a], counts_b[indices_b]).sum())
    denominator = min(len(keys_a), len(keys_b))
    return {
        "rows_a": int(len(keys_a)),
        "rows_b": int(len(keys_b)),
        "unique_a": int(len(unique_a)),
        "unique_b": int(len(unique_b)),
        "within_a_duplicate_rows": int(len(keys_a) - len(unique_a)),
        "within_b_duplicate_rows": int(len(keys_b) - len(unique_b)),
        "common_unique_values": int(len(common)),
        "cross_paired_rows": paired,
        "cross_overlap_fraction_of_smaller_dataset": float(paired / denominator) if denominator else 0.0,
    }


def hash_dataset_rows(dataset, chunk_size):
    digests = np.empty(len(dataset), dtype="S16")
    for start in range(0, len(dataset), chunk_size):
        rows = np.ascontiguousarray(dataset[start : start + chunk_size])
        for offset, row in enumerate(rows):
            digests[start + offset] = hashlib.blake2b(row.tobytes(), digest_size=16).digest()
    return digests


def categorical_distribution(dataset):
    values = dataset.asstr()[:] if h5py.check_string_dtype(dataset.dtype) else dataset[:].astype(str)
    counts = Counter(map(str, values))
    total = len(values)
    return {
        "counts": dict(sorted(counts.items())),
        "fractions": {key: count / total for key, count in sorted(counts.items())},
    }


def categorical_comparison(distribution_a, distribution_b):
    fractions_a = distribution_a["fractions"]
    fractions_b = distribution_b["fractions"]
    categories = sorted(set(fractions_a) | set(fractions_b))
    return {
        "a": distribution_a,
        "b": distribution_b,
        "total_variation_distance": 0.5
        * sum(abs(fractions_a.get(key, 0.0) - fractions_b.get(key, 0.0)) for key in categories),
    }


def vector_distribution_comparison(values_a, values_b):
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if values_a.ndim == 1:
        values_a, values_b = values_a[:, None], values_b[:, None]
    distances = [wasserstein_distance(values_a[:, i], values_b[:, i]) for i in range(values_a.shape[1])]
    return {
        "mean_a": values_a.mean(axis=0).tolist(),
        "mean_b": values_b.mean(axis=0).tolist(),
        "std_a": values_a.std(axis=0).tolist(),
        "std_b": values_b.std(axis=0).tolist(),
        "mean_difference_b_minus_a": (values_b.mean(axis=0) - values_a.mean(axis=0)).tolist(),
        "wasserstein_per_dimension": distances,
        "wasserstein_mean": float(np.mean(distances)),
        "wasserstein_max": float(np.max(distances)),
    }


def scalar_distribution_comparison(values_a, values_b):
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    quantiles = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    return {
        "quantile_levels": quantiles,
        "quantiles_a": np.quantile(values_a, quantiles).tolist(),
        "quantiles_b": np.quantile(values_b, quantiles).tolist(),
        "mean_a": float(values_a.mean()),
        "mean_b": float(values_b.mean()),
        "std_a": float(values_a.std()),
        "std_b": float(values_b.std()),
        "wasserstein": float(wasserstein_distance(values_a, values_b)),
    }


def hdf5_contract(handle):
    return {
        key: {"tail_shape": list(value.shape[1:]), "dtype": str(value.dtype)}
        for key, value in sorted(handle.items())
    }


def training_context(handle):
    required = ("q_start", "ee_goal_pose", "active_ee_mask")
    missing = [key for key in required if key not in handle]
    if missing:
        raise ValueError(f"dataset lacks scheme-3 context fields: {missing}")
    count = len(handle["q_start"])
    return np.concatenate(
        (
            np.asarray(handle["q_start"][:], dtype=np.float64),
            np.asarray(handle["ee_goal_pose"][:], dtype=np.float64).reshape(count, -1),
            np.asarray(handle["active_ee_mask"][:], dtype=np.float64),
        ),
        axis=1,
    )


def compatibility_report(root_a, root_b, handle_a, handle_b):
    manifest_a = load_yaml_if_present(root_a / "manifest.yaml")
    manifest_b = load_yaml_if_present(root_b / "manifest.yaml")
    config_a = load_yaml_if_present(root_a / "generation_config.yaml")
    config_b = load_yaml_if_present(root_b / "generation_config.yaml")
    args_a = load_yaml_if_present(root_a / "args.yaml")
    args_b = load_yaml_if_present(root_b / "args.yaml")
    manifest_keys = ("schema", "ee_goal_schema", "scene_version", "model_sha256", "asset_sha256", "joint_names")
    provenance_a = {key: manifest_a.get(key) for key in manifest_keys} if manifest_a else None
    provenance_b = {key: manifest_b.get(key) for key in manifest_keys} if manifest_b else None
    hdf5_a, hdf5_b = hdf5_contract(handle_a), hdf5_contract(handle_b)
    config_differences = dict_differences(planning_contract(config_a), planning_contract(config_b))
    return {
        "compatible": provenance_a == provenance_b and hdf5_a == hdf5_b and not config_differences,
        "provenance_equal": provenance_a == provenance_b,
        "provenance_a": provenance_a,
        "provenance_b": provenance_b,
        "hdf5_contract_equal": hdf5_a == hdf5_b,
        "hdf5_contract_a": hdf5_a,
        "hdf5_contract_b": hdf5_b,
        "planning_config_differences": config_differences,
        "recorded_dataset_sha256_a": manifest_a.get("dataset_sha256") if manifest_a else None,
        "recorded_dataset_sha256_b": manifest_b.get("dataset_sha256") if manifest_b else None,
        "generation_seed_a": args_a.get("seed") if args_a else None,
        "generation_seed_b": args_b.get("seed") if args_b else None,
    }


def _combine_overlap_metrics(metrics):
    integer_keys = (
        "rows_a",
        "rows_b",
        "unique_a",
        "unique_b",
        "within_a_duplicate_rows",
        "within_b_duplicate_rows",
        "common_unique_values",
        "cross_paired_rows",
    )
    combined = {key: int(sum(item[key] for item in metrics)) for key in integer_keys}
    denominator = sum(min(item["rows_a"], item["rows_b"]) for item in metrics)
    combined["cross_overlap_fraction_of_matched_shard_rows"] = (
        float(combined["cross_paired_rows"] / denominator) if denominator else 0.0
    )
    return combined


def _compact_overlap(metrics):
    return {
        "rows_a": metrics["rows_a"],
        "rows_b": metrics["rows_b"],
        "cross_paired_rows": metrics["cross_paired_rows"],
        "cross_overlap_fraction_of_smaller_dataset": metrics[
            "cross_overlap_fraction_of_smaller_dataset"
        ],
    }


def _numeric_shards(root):
    result = {}
    for path in root.iterdir():
        if not path.is_dir() or not path.name.isdigit():
            continue
        shard_id = int(path.name)
        if shard_id in result:
            raise ValueError(
                f"duplicate numeric shard ID {shard_id} under {root}: "
                f"{result[shard_id].name!r} and {path.name!r}"
            )
        result[shard_id] = path
    return result


def _fully_overlapping_shards(per_shard, metric):
    return [
        item["shard_id"]
        for item in per_shard
        if item[metric]["rows_a"] == item[metric]["rows_b"]
        and item[metric]["cross_paired_rows"] == item[metric]["rows_a"]
    ]


def _distribution_from_counter(counts):
    total = sum(counts.values())
    return {
        "counts": dict(sorted(counts.items())),
        "fractions": {key: value / total for key, value in sorted(counts.items())} if total else {},
    }


def _numeric_or_none(comparison_fn, values_a, values_b):
    if not values_a or not values_b:
        return None
    array_a, array_b = np.concatenate(values_a), np.concatenate(values_b)
    if not np.isfinite(array_a).all() or not np.isfinite(array_b).all():
        finite_a = np.isfinite(array_a).all(axis=-1) if array_a.ndim > 1 else np.isfinite(array_a)
        finite_b = np.isfinite(array_b).all(axis=-1) if array_b.ndim > 1 else np.isfinite(array_b)
        array_a, array_b = array_a[finite_a], array_b[finite_b]
    if len(array_a) == 0 or len(array_b) == 0:
        return None
    return comparison_fn(array_a, array_b)


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if numerator is not None and denominator else None


def _summarize_generation_stats(stats, recorded_shards, trajectories):
    endpoint_seconds = float(stats.get("endpoint_sampling_seconds", 0.0))
    rrt_seconds = float(stats.get("rrt_seconds", 0.0))
    path_seconds = float(sum(stats.get(key, 0.0) for key in PATH_VALIDATION_TIME_FIELDS))
    spline_seconds = float(stats.get("spline_total_seconds", 0.0))
    core_seconds = endpoint_seconds + rrt_seconds + path_seconds + spline_seconds
    task_samples = int(stats.get("task_samples", 0))
    endpoint_failures = int(stats.get("endpoint_failures", 0))
    rrt_attempts = int(stats.get("rrt_attempts", 0))
    rrt_exact = int(stats.get("rrt_exact", 0))
    target_candidates = int(stats.get("target_candidates", 0))
    result = {
        "totals": dict(sorted(stats.items())),
        "recorded_shards_per_stat": dict(sorted(recorded_shards.items())),
        "trajectories": trajectories,
        "endpoint_success_rate": _safe_ratio(task_samples - endpoint_failures, task_samples),
        "rrt_exact_rate": _safe_ratio(rrt_exact, rrt_attempts),
        "post_exact_accept_rate": _safe_ratio(trajectories, rrt_exact),
        "overall_yield_per_task_sample": _safe_ratio(trajectories, task_samples),
        "rrt_attempts_per_final_trajectory": _safe_ratio(rrt_attempts, trajectories),
        "endpoint_seconds_per_task_sample": _safe_ratio(endpoint_seconds, task_samples),
        "target_seconds_per_candidate": _safe_ratio(
            float(stats.get("target_seconds", 0.0)), target_candidates
        ),
        "rrt_seconds_per_attempt": _safe_ratio(rrt_seconds, rrt_attempts),
        "core_worker_seconds": core_seconds,
        "core_worker_seconds_per_final_trajectory": _safe_ratio(core_seconds, trajectories),
    }
    result["core_worker_time_stage_fractions"] = {
        "endpoint_sampling": _safe_ratio(endpoint_seconds, core_seconds),
        "rrt_solve": _safe_ratio(rrt_seconds, core_seconds),
        "raw_path_validation": _safe_ratio(path_seconds, core_seconds),
        "spline_fit_and_validation": _safe_ratio(spline_seconds, core_seconds),
    }
    return result


def compare_matching_shards(
    root_a, root_b, decimals=6, chunk_size=256, include_large_hashes=True
):
    root_a, root_b = Path(root_a).expanduser().resolve(), Path(root_b).expanduser().resolve()
    shard_root_a, shard_root_b = root_a / "shards", root_b / "shards"
    if not shard_root_a.is_dir() or not shard_root_b.is_dir():
        raise FileNotFoundError("both inputs must contain a shards/ directory")
    shards_a, shards_b = _numeric_shards(shard_root_a), _numeric_shards(shard_root_b)
    common_ids = sorted(set(shards_a) & set(shards_b))
    if not common_ids:
        raise ValueError("the two roots have no same-numbered numeric shards")

    overlap_accumulators = {
        "task_id_overlap": [],
        "training_context_overlap_exact": [],
        f"training_context_overlap_rounded_{decimals}_decimals": [],
        f"joint_endpoint_pair_overlap_rounded_{decimals}_decimals": [],
    }
    if include_large_hashes:
        overlap_accumulators["exact_raw_path_overlap"] = []
        overlap_accumulators["exact_bspline_control_point_overlap"] = []

    numeric = {
        "q_start_a": [],
        "q_start_b": [],
        "q_goal_a": [],
        "q_goal_b": [],
        "path_length_a": [],
        "path_length_b": [],
        "planning_time_a": [],
        "planning_time_b": [],
        "ee_left_a": [],
        "ee_left_b": [],
        "ee_right_a": [],
        "ee_right_b": [],
    }
    categories_a = {field: Counter() for field in CATEGORICAL_FIELDS}
    categories_b = {field: Counter() for field in CATEGORICAL_FIELDS}
    generation_stats_a, generation_stats_b = Counter(), Counter()
    recorded_stats_a, recorded_stats_b = Counter(), Counter()
    per_shard = []

    for index, shard_id in enumerate(common_ids, 1):
        path_a = shards_a[shard_id] / "dataset_merged.hdf5"
        path_b = shards_b[shard_id] / "dataset_merged.hdf5"
        if not path_a.is_file() or not path_b.is_file():
            raise ValueError(f"same-numbered shard {shard_id} is incomplete on one computer")
        with h5py.File(path_a, "r") as handle_a, h5py.File(path_b, "r") as handle_b:
            compatibility = compatibility_report(
                shards_a[shard_id], shards_b[shard_id], handle_a, handle_b
            )
            task = overlap_metrics(
                row_keys(np.asarray(handle_a["task_id"][:], dtype=np.int64)[:, None]),
                row_keys(np.asarray(handle_b["task_id"][:], dtype=np.int64)[:, None]),
            )
            context_a, context_b = training_context(handle_a), training_context(handle_b)
            context_exact = overlap_metrics(row_keys(context_a), row_keys(context_b))
            context_rounded = overlap_metrics(
                row_keys(context_a, decimals), row_keys(context_b, decimals)
            )
            endpoint_a = np.concatenate((handle_a["q_start"][:], handle_a["q_goal"][:]), axis=1)
            endpoint_b = np.concatenate((handle_b["q_start"][:], handle_b["q_goal"][:]), axis=1)
            endpoints = overlap_metrics(row_keys(endpoint_a, decimals), row_keys(endpoint_b, decimals))
            shard_metrics = {
                "shard_id": f"{shard_id:09d}",
                "directory_a": shards_a[shard_id].name,
                "directory_b": shards_b[shard_id].name,
                "rows_a": len(handle_a["task_id"]),
                "rows_b": len(handle_b["task_id"]),
                "compatible": compatibility["compatible"],
                "generation_seed_a": compatibility["generation_seed_a"],
                "generation_seed_b": compatibility["generation_seed_b"],
                "planning_config_differences": compatibility["planning_config_differences"],
                "task_id_overlap": _compact_overlap(task),
                "training_context_overlap_exact": _compact_overlap(context_exact),
                f"training_context_overlap_rounded_{decimals}_decimals": _compact_overlap(
                    context_rounded
                ),
                f"joint_endpoint_pair_overlap_rounded_{decimals}_decimals": _compact_overlap(
                    endpoints
                ),
            }
            overlap_accumulators["task_id_overlap"].append(task)
            overlap_accumulators["training_context_overlap_exact"].append(context_exact)
            overlap_accumulators[
                f"training_context_overlap_rounded_{decimals}_decimals"
            ].append(context_rounded)
            overlap_accumulators[
                f"joint_endpoint_pair_overlap_rounded_{decimals}_decimals"
            ].append(endpoints)

            if include_large_hashes:
                raw = overlap_metrics(
                    hash_dataset_rows(handle_a["sol_path"], chunk_size),
                    hash_dataset_rows(handle_b["sol_path"], chunk_size),
                )
                spline = overlap_metrics(
                    hash_dataset_rows(handle_a["bspline_params_cc"], chunk_size),
                    hash_dataset_rows(handle_b["bspline_params_cc"], chunk_size),
                )
                shard_metrics["exact_raw_path_overlap"] = _compact_overlap(raw)
                shard_metrics["exact_bspline_control_point_overlap"] = _compact_overlap(spline)
                overlap_accumulators["exact_raw_path_overlap"].append(raw)
                overlap_accumulators["exact_bspline_control_point_overlap"].append(spline)

            numeric["q_start_a"].append(handle_a["q_start"][:])
            numeric["q_start_b"].append(handle_b["q_start"][:])
            numeric["q_goal_a"].append(handle_a["q_goal"][:])
            numeric["q_goal_b"].append(handle_b["q_goal"][:])
            for field, key in (("joint_path_length", "path_length"), ("planning_time", "planning_time")):
                if field in handle_a and field in handle_b:
                    numeric[f"{key}_a"].append(handle_a[field][:])
                    numeric[f"{key}_b"].append(handle_b[field][:])
            pose_a, pose_b = handle_a["ee_goal_pose"][:], handle_b["ee_goal_pose"][:]
            mask_a = handle_a["active_ee_mask"][:].astype(bool)
            mask_b = handle_b["active_ee_mask"][:].astype(bool)
            for arm_index, arm in enumerate(("left", "right")):
                numeric[f"ee_{arm}_a"].append(pose_a[mask_a[:, arm_index], arm_index, :, 3])
                numeric[f"ee_{arm}_b"].append(pose_b[mask_b[:, arm_index], arm_index, :, 3])
            for field in CATEGORICAL_FIELDS:
                if field in handle_a and field in handle_b:
                    categories_a[field].update(categorical_distribution(handle_a[field])["counts"])
                    categories_b[field].update(categorical_distribution(handle_b[field])["counts"])
            for shard_root, totals, recorded in (
                (shards_a[shard_id], generation_stats_a, recorded_stats_a),
                (shards_b[shard_id], generation_stats_b, recorded_stats_b),
            ):
                manifest = load_yaml_if_present(shard_root / "manifest.yaml") or {}
                for key, value in manifest.get("stats", {}).items():
                    if isinstance(value, (int, float)):
                        totals[key] += value
                        recorded[key] += 1
            per_shard.append(shard_metrics)
        if index % 100 == 0 or index == len(common_ids):
            print(f"compared matching shards {index}/{len(common_ids)}", flush=True)

    combined_overlap = {
        key: _combine_overlap_metrics(values) for key, values in overlap_accumulators.items()
    }
    categorical = {
        field: categorical_comparison(
            _distribution_from_counter(categories_a[field]), _distribution_from_counter(categories_b[field])
        )
        for field in CATEGORICAL_FIELDS
        if categories_a[field] and categories_b[field]
    }
    report = {
        "schema": "marvin_bimanual_warehouse_matching_shard_comparison/v1",
        "root_a": str(root_a),
        "root_b": str(root_b),
        "matching_shard_count": len(common_ids),
        "matching_shard_ids": [f"{shard_id:09d}" for shard_id in common_ids],
        "only_in_a_count": len(set(shards_a) - set(shards_b)),
        "only_in_b_count": len(set(shards_b) - set(shards_a)),
        "only_in_a": [
            f"{shard_id:09d}" for shard_id in sorted(set(shards_a) - set(shards_b))
        ],
        "only_in_b": [
            f"{shard_id:09d}" for shard_id in sorted(set(shards_b) - set(shards_a))
        ],
        "compatible_matching_shards": sum(item["compatible"] for item in per_shard),
        "incompatible_matching_shards": [
            item["shard_id"] for item in per_shard if not item["compatible"]
        ],
        **combined_overlap,
        "categorical_distributions": categorical,
        "joint_start_distribution": _numeric_or_none(
            vector_distribution_comparison, numeric["q_start_a"], numeric["q_start_b"]
        ),
        "joint_goal_distribution": _numeric_or_none(
            vector_distribution_comparison, numeric["q_goal_a"], numeric["q_goal_b"]
        ),
        "joint_path_length_distribution": _numeric_or_none(
            scalar_distribution_comparison, numeric["path_length_a"], numeric["path_length_b"]
        ),
        "planning_time_distribution": _numeric_or_none(
            scalar_distribution_comparison, numeric["planning_time_a"], numeric["planning_time_b"]
        ),
        "active_ee_goal_distributions": {
            arm: _numeric_or_none(
                vector_distribution_comparison, numeric[f"ee_{arm}_a"], numeric[f"ee_{arm}_b"]
            )
            for arm in ("left", "right")
        },
        "generation_stats": {
            "a": _summarize_generation_stats(
                generation_stats_a, recorded_stats_a, combined_overlap["task_id_overlap"]["rows_a"]
            ),
            "b": _summarize_generation_stats(
                generation_stats_b, recorded_stats_b, combined_overlap["task_id_overlap"]["rows_b"]
            ),
        },
        "per_matching_shard": per_shard,
    }
    worker_seconds_a = report["generation_stats"]["a"][
        "core_worker_seconds_per_final_trajectory"
    ]
    worker_seconds_b = report["generation_stats"]["b"][
        "core_worker_seconds_per_final_trajectory"
    ]
    report["generation_stats"]["a_to_b_core_worker_time_ratio"] = _safe_ratio(
        worker_seconds_a, worker_seconds_b
    )
    overlap_key = f"training_context_overlap_rounded_{decimals}_decimals"
    report["fully_overlapping_shards"] = {
        "training_context_exact": _fully_overlapping_shards(
            per_shard, "training_context_overlap_exact"
        ),
        f"training_context_rounded_{decimals}_decimals": _fully_overlapping_shards(
            per_shard, overlap_key
        ),
        f"joint_endpoint_pair_rounded_{decimals}_decimals": _fully_overlapping_shards(
            per_shard, f"joint_endpoint_pair_overlap_rounded_{decimals}_decimals"
        ),
    }
    if include_large_hashes:
        report["fully_overlapping_shards"].update(
            exact_raw_path=_fully_overlapping_shards(per_shard, "exact_raw_path_overlap"),
            exact_bspline_control_points=_fully_overlapping_shards(
                per_shard, "exact_bspline_control_point_overlap"
            ),
        )
    overlap = report[overlap_key]["cross_overlap_fraction_of_matched_shard_rows"]
    report["assessment"] = {
        "training_context_overlap_fraction": overlap,
        "training_context_overlap_level": (
            "very_high"
            if overlap >= 0.5
            else "high"
            if overlap >= 0.1
            else "moderate"
            if overlap >= 0.01
            else "low"
        ),
        "comparison_scope": "same-numbered shards only",
    }
    if not include_large_hashes:
        report["large_field_hashes_skipped"] = True
    return report


def print_matching_shard_summary(report, decimals):
    rounded = report[f"training_context_overlap_rounded_{decimals}_decimals"]
    print(f"Matching shards compared: {report['matching_shard_count']}")
    print(f"Shard IDs only in A/B: {report['only_in_a_count']}/{report['only_in_b_count']}")
    print(
        "Compatible/incompatible matching shards: "
        f"{report['compatible_matching_shards']}/{len(report['incompatible_matching_shards'])}"
    )
    print(
        "Overlapping task IDs within matching shards: "
        f"{report['task_id_overlap']['cross_paired_rows']} "
        f"({report['task_id_overlap']['cross_overlap_fraction_of_matched_shard_rows']:.2%})"
    )
    print(
        f"Overlapping contexts within matching shards ({decimals} decimals): "
        f"{rounded['cross_paired_rows']} "
        f"({rounded['cross_overlap_fraction_of_matched_shard_rows']:.2%})"
    )
    stats_a, stats_b = report["generation_stats"]["a"], report["generation_stats"]["b"]
    if stats_a["rrt_exact_rate"] is not None and stats_b["rrt_exact_rate"] is not None:
        print(f"RRT exact rate A/B: {stats_a['rrt_exact_rate']:.2%}/{stats_b['rrt_exact_rate']:.2%}")
    worker_a = stats_a["core_worker_seconds_per_final_trajectory"]
    worker_b = stats_b["core_worker_seconds_per_final_trajectory"]
    if worker_a is not None and worker_b is not None:
        print(f"Core worker seconds per final trajectory A/B: {worker_a:.2f}/{worker_b:.2f}")
    if "exact_raw_path_overlap" in report:
        raw = report["exact_raw_path_overlap"]
        spline = report["exact_bspline_control_point_overlap"]
        print(
            f"Exact raw paths: {raw['cross_paired_rows']} "
            f"({raw['cross_overlap_fraction_of_matched_shard_rows']:.2%})"
        )
        print(
            "Exact spline control points: "
            f"{spline['cross_paired_rows']} ({spline['cross_overlap_fraction_of_matched_shard_rows']:.2%})"
        )
        print(
            "Fully identical raw-path shards: "
            f"{len(report['fully_overlapping_shards']['exact_raw_path'])}/"
            f"{report['matching_shard_count']}"
        )
    print(f"Context overlap assessment: {report['assessment']['training_context_overlap_level']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_a", type=Path, help="Computer-A dataset root containing shards/")
    parser.add_argument("root_b", type=Path, help="Computer-B dataset root containing shards/")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--round-decimals", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--skip-large-field-hashes",
        action="store_true",
        help="Skip full raw-path and spline hashing for a faster comparison",
    )
    args = parser.parse_args(argv)
    if args.round_decimals < 0 or args.chunk_size < 1:
        raise ValueError("round-decimals must be nonnegative and chunk-size must be positive")
    report = compare_matching_shards(
        args.root_a,
        args.root_b,
        decimals=args.round_decimals,
        chunk_size=args.chunk_size,
        include_large_hashes=not args.skip_large_field_hashes,
    )
    print_matching_shard_summary(report, args.round_decimals)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"JSON report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
