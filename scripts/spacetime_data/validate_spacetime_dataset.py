#!/usr/bin/env python3
"""Audit canonical Space-Time MPD shards independently from generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy import interpolate
import yaml

from mpd.datasets.spacetime_legacy import open_uniform_knots
from mpd.datasets.spacetime_schema import CANONICAL_SAMPLE_FIELDS, SCHEMA_VERSION, sha256_file
from mpd.parametric_trajectory.timing_fitting import TimingSplineNumpy, attach_spatial_derivatives


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ratio-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def validate(dataset_root: Path, *, max_samples: int = None, ratio_tolerance: float = 1e-3) -> Dict[str, object]:
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "manifest.yaml"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {manifest.get('schema_version')}")
    robot = manifest["robot"]
    for filename, hash_key in (
        ("robot.urdf", "urdf_sha256"),
        ("joint_limits.yaml", "joint_limits_sha256"),
        ("collision_spheres.yaml", "collision_spheres_sha256"),
        ("collision_parent_bounds.yaml", "collision_parent_bounds_sha256"),
    ):
        actual = sha256_file(dataset_root / "robot" / filename)
        if actual != robot[hash_key]:
            raise ValueError(f"robot bundle hash mismatch for {filename}")

    spatial_config = manifest["spatial_spline"]
    timing_config = manifest["timing_spline"]
    timing_spline = TimingSplineNumpy(
        num_control_points=int(timing_config["num_control_points"]),
        degree=int(timing_config["degree"]),
        num_phase_points=int(timing_config["num_phase_points"]),
        u_min=float(timing_config["u_min"]),
    )
    with (dataset_root / "robot" / "joint_limits.yaml").open("r", encoding="utf-8") as stream:
        limits = yaml.safe_load(stream)
    joint_names = robot["active_joint_names"]
    dq_max = np.asarray([limits[name]["qdot_max"] for name in joint_names])
    ddq_max = np.asarray([limits[name]["qddot_max"] for name in joint_names])
    urdf_root = ET.parse(str(dataset_root / "robot" / "robot.urdf")).getroot()
    urdf_limits = {
        joint.attrib["name"]: (
            float(joint.find("limit").attrib["lower"]),
            float(joint.find("limit").attrib["upper"]),
        )
        for joint in urdf_root.findall("joint")
        if joint.attrib.get("type") != "fixed"
    }
    q_min = np.asarray([urdf_limits[name][0] for name in joint_names])
    q_max = np.asarray([urdf_limits[name][1] for name in joint_names])
    knots = open_uniform_knots(
        int(spatial_config["num_control_points"]), int(spatial_config["degree"])
    )
    dimensions = {
        "H_P": int(spatial_config["num_control_points"]),
        "D": int(robot["dof"]),
        "K": int(timing_config["num_control_points"]),
        "H_T": int(timing_config["num_phase_points"]),
    }

    checked = 0
    unique_base_paths = set()
    all_base_paths = set()
    violations = []
    maxima = {"v_ratio_max": 0.0, "a_ratio_max": 0.0, "relative_timing_rmse": 0.0}
    for shard_path in sorted((dataset_root / "shards").glob("part-*.hdf5")):
        with h5py.File(str(shard_path), "r") as shard:
            if shard.attrs.get("schema_version") != SCHEMA_VERSION:
                violations.append(f"{shard_path.name}: schema_version")
                continue
            for hash_key in (
                "urdf_sha256",
                "joint_limits_sha256",
                "collision_spheres_sha256",
                "collision_parent_bounds_sha256",
            ):
                if shard.attrs.get(hash_key) != robot[hash_key]:
                    violations.append(f"{shard_path.name}: {hash_key}")
            missing = sorted(set(CANONICAL_SAMPLE_FIELDS) - set(_all_dataset_paths(shard)))
            if missing:
                violations.append(f"{shard_path.name}: missing={missing}")
                continue
            row_counts = {
                shard[name].shape[0] for name in CANONICAL_SAMPLE_FIELDS
            }
            if len(row_counts) != 1:
                violations.append(f"{shard_path.name}: inconsistent field lengths={row_counts}")
                continue
            for name, (dtype, symbolic_shape) in CANONICAL_SAMPLE_FIELDS.items():
                expected_shape = tuple(dimensions[value] for value in symbolic_shape)
                if shard[name].dtype != dtype:
                    violations.append(
                        f"{shard_path.name}: {name} dtype={shard[name].dtype}, expected={dtype}"
                    )
                if shard[name].shape[1:] != expected_shape:
                    violations.append(
                        f"{shard_path.name}: {name} shape={shard[name].shape[1:]}, "
                        f"expected={expected_shape}"
                    )
            all_base_paths.update(
                np.unique(shard["index/base_path_id"][:]).astype(np.int64).tolist()
            )
            for row in range(shard["timing/control_points"].shape[0]):
                if max_samples is not None and checked >= max_samples:
                    break
                checked += 1
                unique_base_paths.add(int(shard["index/base_path_id"][row]))
                prefix = f"{shard_path.name}:{row}"
                control_points = np.asarray(shard["timing/control_points"][row], dtype=np.float64)
                timing = timing_spline.evaluate(control_points)
                spatial_control_points = np.asarray(
                    shard["spatial/control_points"][row], dtype=np.float64
                )
                spatial = interpolate.BSpline(
                    knots,
                    spatial_control_points,
                    int(spatial_config["degree"]),
                    axis=0,
                )
                q, dq, ddq = attach_spatial_derivatives(spatial, timing)
                v_ratio = float(np.max(np.abs(dq) / dq_max[None, :]))
                a_ratio = float(np.max(np.abs(ddq) / ddq_max[None, :]))
                reference = np.asarray(shard["timing/reference_time"][row], dtype=np.float64)
                relative_rmse = float(
                    np.sqrt(np.mean(np.square(timing.time_from_start - reference)))
                    / reference[-1]
                )
                maxima["v_ratio_max"] = max(maxima["v_ratio_max"], v_ratio)
                maxima["a_ratio_max"] = max(maxima["a_ratio_max"], a_ratio)
                maxima["relative_timing_rmse"] = max(
                    maxima["relative_timing_rmse"], relative_rmse
                )
                if not np.all(np.diff(reference) > 0.0):
                    violations.append(f"{prefix}: non_monotone_reference")
                if not bool(shard["quality/accepted"][row]):
                    violations.append(f"{prefix}: stored sample is not accepted")
                if not np.allclose(spatial_control_points[:3], spatial_control_points[0], atol=1e-6):
                    violations.append(f"{prefix}: spatial_start_boundary")
                if not np.allclose(spatial_control_points[-3:], spatial_control_points[-1], atol=1e-6):
                    violations.append(f"{prefix}: spatial_goal_boundary")
                if not np.allclose(
                    shard["condition/q_start"][row], spatial_control_points[0], atol=1e-6
                ):
                    violations.append(f"{prefix}: q_start_mismatch")
                if not np.allclose(
                    shard["condition/q_goal"][row], spatial_control_points[-1], atol=1e-6
                ):
                    violations.append(f"{prefix}: q_goal_mismatch")
                if np.any(q < q_min[None, :] - 1e-6) or np.any(q > q_max[None, :] + 1e-6):
                    violations.append(f"{prefix}: joint_position_limit")
                if not np.isclose(timing.duration, shard["timing/duration"][row], rtol=1e-5):
                    violations.append(f"{prefix}: duration_mismatch")
                if timing.duration < float(timing_config["duration_min"]):
                    violations.append(f"{prefix}: duration_below_min")
                if timing.duration > float(timing_config["duration_max"]):
                    violations.append(f"{prefix}: duration_above_max")
                if v_ratio > 1.0 + ratio_tolerance:
                    violations.append(f"{prefix}: velocity_limit={v_ratio}")
                if a_ratio > 1.0 + ratio_tolerance:
                    violations.append(f"{prefix}: acceleration_limit={a_ratio}")
            if max_samples is not None and checked >= max_samples:
                break

    split_sets = {}
    for name in ("train", "val", "test"):
        split_sets[name] = set(
            np.load(str(dataset_root / "splits" / f"{name}_base_path_ids.npy")).tolist()
        )
    if split_sets["train"] & split_sets["val"]:
        violations.append("train/val base_path_id leakage")
    if split_sets["train"] & split_sets["test"]:
        violations.append("train/test base_path_id leakage")
    if split_sets["val"] & split_sets["test"]:
        violations.append("val/test base_path_id leakage")
    source_group_size = int(manifest.get("splits", {}).get("source_group_size", 1))
    group_memberships = {}
    for split_name, base_path_ids in split_sets.items():
        for base_path_id in base_path_ids:
            group_id = int(base_path_id) // source_group_size
            previous = group_memberships.setdefault(group_id, split_name)
            if previous != split_name:
                violations.append(
                    f"source group {group_id} leaks across {previous}/{split_name}"
                )
                break
    split_union = split_sets["train"] | split_sets["val"] | split_sets["test"]
    if split_union != all_base_paths:
        violations.append(
            f"split coverage mismatch: split={len(split_union)}, shards={len(all_base_paths)}"
        )
    return {
        "schema_version": "spacetime_mpd_validation_report_v1",
        "dataset_root": str(dataset_root),
        "checked_samples": checked,
        "checked_base_paths": len(unique_base_paths),
        "maxima": maxima,
        "violation_count": len(violations),
        "violations": violations[:100],
        "passed": not violations,
    }


def _all_dataset_paths(group: h5py.Group):
    result = []
    group.visititems(lambda name, value: result.append(name) if isinstance(value, h5py.Dataset) else None)
    return result


def main() -> int:
    args = _parse_args()
    report = validate(
        args.dataset_root,
        max_samples=args.max_samples,
        ratio_tolerance=args.ratio_tolerance,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
