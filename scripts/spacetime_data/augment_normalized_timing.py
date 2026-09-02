#!/usr/bin/env python3
"""Add ``(T_min, T_max, tau, r)`` fields to existing Space-Time MPD shards.

The operation is independent from RRT and TOPP-RA.  Each completed shard is
copied, augmented, flushed, and atomically replaced so an interruption cannot
leave a partially modified HDF5 file.  Rows outside the configured duration
bounds are retained and marked with ``quality/duration_bounds_valid = False``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Dict

import numpy as np
import yaml

from mpd.datasets.spacetime_timing_augmentation import (
    NORMALIZED_TIMING_EXTENSION_VERSION,
    augment_shard_atomic,
    augmentation_config,
)
from mpd.datasets.spacetime_schema import SCHEMA_VERSION
from mpd.parametric_trajectory.normalized_timing import NormalizedTimingSplineNumpy
from mpd.parametric_trajectory.timing_fitting import TimingSplineNumpy


def _parse_args(repository_root: Path) -> argparse.Namespace:
    default_root = (
        repository_root
        / "data_trajectories_spacetime"
        / "EnvWarehouse-RobotPanda-RRTConnect-SpaceTime-v1"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, nargs="?", default=default_root)
    parser.add_argument(
        "--duration-max",
        type=float,
        default=14.0,
        help="task time limit in seconds (default: 14)",
    )
    parser.add_argument("--duration-floor", type=float, default=0.0)
    parser.add_argument("--density-floor", type=float, default=1e-3)
    parser.add_argument("--logit-clip", type=float, default=1e-3)
    parser.add_argument("--t-min-safety-factor", type=float, default=1.0)
    parser.add_argument("--bounds-tolerance", type=float, default=1e-3)
    parser.add_argument("--max-fit-rmse", type=float, default=0.05)
    parser.add_argument(
        "--shape-source",
        choices=("control_points", "reference_time"),
        default="control_points",
        help="fit r from the deployed c curve (default) or the saved teacher curve",
    )
    return parser.parse_args()


def _atomic_write_yaml(path: Path, value: Dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".updating", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        shutil.copymode(str(path), str(temporary_path))
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def augment_dataset(args: argparse.Namespace) -> Dict[str, object]:
    dataset_root = args.dataset_root.resolve()
    manifest_path = dataset_root / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    in_progress = sorted(dataset_root.rglob("*.inprogress"))
    if in_progress:
        examples = ", ".join(path.name for path in in_progress[:3])
        raise RuntimeError(
            "dataset generation appears active or interrupted; finish generation before "
            f"augmenting the dataset ({examples})"
        )
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {manifest.get('schema_version')}")
    spatial_config = manifest["spatial_spline"]
    timing_config = manifest["timing_spline"]
    robot = manifest["robot"]
    with (dataset_root / robot["joint_limits"]).open("r", encoding="utf-8") as stream:
        limits = yaml.safe_load(stream)
    joint_names = robot["active_joint_names"]
    dq_max = np.asarray([limits[name]["qdot_max"] for name in joint_names], dtype=np.float64)
    ddq_max = np.asarray(
        [limits[name]["qddot_max"] for name in joint_names], dtype=np.float64
    )
    timing_spline = TimingSplineNumpy(
        num_control_points=int(timing_config["num_control_points"]),
        degree=int(timing_config["degree"]),
        num_phase_points=int(timing_config["num_phase_points"]),
        u_min=float(timing_config["u_min"]),
    )
    normalized_spline = NormalizedTimingSplineNumpy(
        num_control_points=int(timing_config["num_control_points"]),
        degree=int(timing_config["degree"]),
        num_phase_points=int(timing_config["num_phase_points"]),
        density_floor=args.density_floor,
    )
    config = augmentation_config(
        duration_max=args.duration_max,
        duration_floor=args.duration_floor,
        density_floor=args.density_floor,
        logit_clip=args.logit_clip,
        t_min_safety_factor=args.t_min_safety_factor,
        bounds_tolerance=args.bounds_tolerance,
        max_fit_rmse=args.max_fit_rmse,
        shape_source=args.shape_source,
        timing_degree=int(timing_config["degree"]),
        num_phase_points=int(timing_config["num_phase_points"]),
    )
    shard_paths = sorted((dataset_root / "shards").glob("part-*.hdf5"))
    if not shard_paths:
        raise ValueError(f"no completed shards found in {dataset_root / 'shards'}")
    started = time.monotonic()
    totals = {"rows": 0, "valid_rows": 0, "invalid_rows": 0, "logit_clipped_rows": 0}
    fit_rmse_max = 0.0
    t_min_min = np.inf
    t_min_max = -np.inf
    for index, shard_path in enumerate(shard_paths, start=1):
        shard_report = augment_shard_atomic(
            shard_path,
            spatial_degree=int(spatial_config["degree"]),
            timing_spline=timing_spline,
            normalized_spline=normalized_spline,
            dq_max=dq_max,
            ddq_max=ddq_max,
            config=config,
        )
        for name in totals:
            totals[name] += int(shard_report[name])
        fit_rmse_max = max(fit_rmse_max, float(shard_report["fit_rmse_max"]))
        t_min_min = min(t_min_min, float(shard_report["t_min_min"]))
        t_min_max = max(t_min_max, float(shard_report["t_min_max"]))
        print(
            f"augmented={index}/{len(shard_paths)} shard={shard_path.name} "
            f"rows={shard_report['rows']} valid={shard_report['valid_rows']}",
            flush=True,
        )

    manifest["normalized_timing"] = dict(config)
    manifest["normalized_timing"]["fields"] = {
        "shape": "timing/shape_control_points",
        "duration_min": "timing/t_min",
        "duration_max": "timing/t_max",
        "duration_logit": "timing/tau",
        "training_valid": "quality/duration_bounds_valid",
    }
    _atomic_write_yaml(manifest_path, manifest)

    report: Dict[str, object] = {
        "schema_version": "spacetime_normalized_timing_augmentation_report_v1",
        "extension_version": NORMALIZED_TIMING_EXTENSION_VERSION,
        "dataset_root": str(dataset_root),
        "config": config,
        "processed_shards": len(shard_paths),
        **totals,
        "fit_rmse_max": fit_rmse_max,
        "t_min_min": t_min_min,
        "t_min_max": t_min_max,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = dataset_root / "normalized-timing-augmentation-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    args = _parse_args(repository_root)
    report = augment_dataset(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
