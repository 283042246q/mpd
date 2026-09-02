#!/usr/bin/env python3
"""Build canonical ``(P, c)`` training shards from legacy warehouse paths.

This is a standalone data-engineering path. It does not instantiate the old
``TrajectoryDatasetBspline`` and does not modify the original spatial HDF5.
Run it from the repository root in the ``mpd-splines-public`` environment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional

import h5py
import numpy as np

from mpd.datasets.spacetime_hdf5 import SpaceTimeShardWriter, write_grouped_splits
from mpd.datasets.spacetime_legacy import LegacyWarehouseReader
from mpd.datasets.spacetime_schema import (
    CANONICAL_SAMPLE_FIELDS,
    SpatialSource,
    build_manifest,
    load_panda_robot_bundle,
    materialize_dataset_contract,
)
from mpd.parametric_trajectory.timing_fitting import (
    TimingSplineNumpy,
    fit_and_validate_timing_reference,
)
from mpd.parametric_trajectory.toppra_retiming import (
    RetimingReference,
    build_multimodal_retiming_references,
)


DEFAULT_VARIANTS = (
    "toppra",
    "duration_1.2",
    "duration_1.5",
    "limits_0.85_0.80",
    "limits_0.65_0.70",
    "local_slowdown",
    "near_wait",
)


class OnlineMetric:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf

    def update(self, value: float) -> None:
        value = float(value)
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def to_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {"count": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "count": self.count,
            "mean": self.total / self.count,
            "min": self.minimum,
            "max": self.maximum,
        }


def _parse_args(repository_root: Path) -> argparse.Namespace:
    source_default = (
        repository_root
        / "data_trajectories"
        / "EnvWarehouse-RobotPanda-config_file_v01-joint_joint-one-RRTConnect"
        / "dataset_merged_doubled.hdf5"
    )
    output_default = (
        repository_root
        / "data_trajectories_spacetime"
        / "EnvWarehouse-RobotPanda-RRTConnect-SpaceTime-v1"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=source_default)
    parser.add_argument("--output-root", type=Path, default=output_default)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-paths", type=int, default=None)
    parser.add_argument("--paths-per-shard", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1726484688)
    parser.add_argument("--variant", action="append", choices=DEFAULT_VARIANTS, dest="variants")
    parser.add_argument("--spatial-num-control-points", type=int, default=29)
    parser.add_argument("--spatial-degree", type=int, default=5)
    parser.add_argument("--num-phase-points", type=int, default=128)
    parser.add_argument("--timing-num-control-points", type=int, default=8)
    parser.add_argument("--timing-degree", type=int, default=3)
    parser.add_argument("--timing-u-min", type=float, default=0.05)
    parser.add_argument("--duration-min", type=float, default=2.0)
    parser.add_argument("--duration-max", type=float, default=15.0)
    parser.add_argument("--num-toppra-gridpoints", type=int, default=256)
    parser.add_argument("--spatial-fit-rmse-max", type=float, default=0.05)
    parser.add_argument("--timing-fit-relative-rmse-max", type=float, default=0.05)
    parser.add_argument("--ratio-tolerance", type=float, default=1e-3)
    parser.add_argument("--safety-margin", type=float, default=1.02)
    parser.add_argument("--max-safety-iterations", type=int, default=5)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    args.variants = tuple(args.variants or DEFAULT_VARIANTS)
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.max_paths is not None and args.max_paths <= 0:
        parser.error("--max-paths must be positive")
    if args.paths_per_shard <= 0:
        parser.error("--paths-per-shard must be positive")
    if args.report_every <= 0:
        parser.error("--report-every must be positive")
    return args


def _variant_manifest(variants: Iterable[str]) -> List[Mapping[str, object]]:
    result = []
    for name in variants:
        variant_id = DEFAULT_VARIANTS.index(name)
        entry: Dict[str, object] = {"name": name, "variant_id": variant_id, "mode_id": variant_id}
        if name.startswith("duration_"):
            entry.update(method="duration_scaled", duration_scale=float(name.split("_")[1]))
        elif name.startswith("limits_"):
            _, velocity, acceleration = name.split("_")
            entry.update(
                method="limit_scaled_toppra",
                velocity_scale=float(velocity),
                acceleration_scale=float(acceleration),
            )
        elif name == "toppra":
            entry.update(method="toppra", velocity_scale=1.0, acceleration_scale=1.0)
        elif name == "local_slowdown":
            entry.update(method="local_density_bump", randomized=True)
        elif name == "near_wait":
            entry.update(method="near_wait_density_bump", randomized=True)
        result.append(entry)
    return result


def _records_for_path(
    *,
    spatial_control_points: np.ndarray,
    spatial_fit_rmse: float,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    task_id: int,
    base_path_id: int,
    accepted: List[tuple],
) -> Mapping[str, np.ndarray]:
    rows = len(accepted)
    result: Dict[str, np.ndarray] = {}
    result["spatial/control_points"] = np.repeat(
        spatial_control_points[None, :, :], rows, axis=0
    )
    result["timing/control_points"] = np.stack(
        [validated.fit.control_points for _, _, validated in accepted]
    )
    result["timing/duration"] = np.asarray(
        [validated.fit.duration for _, _, validated in accepted]
    )
    result["timing/reference_time"] = np.stack(
        [validated.reference_time for _, _, validated in accepted]
    )
    result["condition/q_start"] = np.repeat(q_start[None, :], rows, axis=0)
    result["condition/q_goal"] = np.repeat(q_goal[None, :], rows, axis=0)
    result["index/task_id"] = np.full(rows, task_id)
    result["index/base_path_id"] = np.full(rows, base_path_id)
    result["index/variant_id"] = np.asarray([variant_id for variant_id, _, _ in accepted])
    result["index/scene_id"] = np.full(rows, -1)
    result["index/mode_id"] = np.asarray(
        [reference.mode_id for _, reference, _ in accepted]
    )
    result["source/spatial"] = np.full(rows, int(SpatialSource.RRT_CONNECT))
    result["source/timing"] = np.asarray(
        [int(reference.source) for _, reference, _ in accepted]
    )
    result["source/parent_sample_id"] = np.full(rows, -1)
    result["quality/spatial_fit_rmse"] = np.full(rows, spatial_fit_rmse)
    result["quality/timing_fit_rmse"] = np.asarray(
        [validated.fit.rmse for _, _, validated in accepted]
    )
    result["quality/v_ratio_max"] = np.asarray(
        [validated.v_ratio_max for _, _, validated in accepted]
    )
    result["quality/a_ratio_max"] = np.asarray(
        [validated.a_ratio_max for _, _, validated in accepted]
    )
    result["quality/reference_scale"] = np.asarray(
        [validated.reference_scale for _, _, validated in accepted]
    )
    result["quality/static_clearance_min"] = np.full(rows, np.nan)
    result["quality/dynamic_clearance_min"] = np.full(rows, np.nan)
    result["quality/accepted"] = np.ones(rows, dtype=np.bool_)
    if set(result) != set(CANONICAL_SAMPLE_FIELDS):
        raise AssertionError("generator record does not match canonical schema")
    return result


def _collect_output_base_path_ids(shards_root: Path) -> np.ndarray:
    values = []
    for shard in sorted(shards_root.glob("part-*.hdf5")):
        with h5py.File(str(shard), "r") as source:
            values.append(np.unique(source["index/base_path_id"][:]))
    if not values:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(values).astype(np.int64, copy=False))


def _json_compatible_args(args: argparse.Namespace) -> Dict[str, object]:
    result = vars(args).copy()
    result["source"] = str(args.source.resolve())
    result["output_root"] = str(args.output_root.resolve())
    result["variants"] = list(args.variants)
    return result


def generate(args: argparse.Namespace, repository_root: Path) -> Dict[str, object]:
    source_path = args.source.resolve()
    output_root = args.output_root.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    robot = load_panda_robot_bundle(repository_root)
    timing_spline = TimingSplineNumpy(
        num_control_points=args.timing_num_control_points,
        degree=args.timing_degree,
        num_phase_points=args.num_phase_points,
        u_min=args.timing_u_min,
    )
    manifest = build_manifest(
        robot,
        spatial_num_control_points=args.spatial_num_control_points,
        spatial_degree=args.spatial_degree,
        spatial_num_phase_points=args.num_phase_points,
        timing_num_control_points=args.timing_num_control_points,
        timing_degree=args.timing_degree,
        timing_num_phase_points=args.num_phase_points,
        timing_u_min=args.timing_u_min,
        timing_duration_min=args.duration_min,
        timing_duration_max=args.duration_max,
        source_dataset=str(source_path),
        variants=_variant_manifest(args.variants),
    )
    manifest["generation"] = {
        "seed": args.seed,
        "num_toppra_gridpoints": args.num_toppra_gridpoints,
        "spatial_fit_rmse_max": args.spatial_fit_rmse_max,
        "timing_fit_relative_rmse_max": args.timing_fit_relative_rmse_max,
        "ratio_tolerance": args.ratio_tolerance,
        "safety_margin": args.safety_margin,
        "static_collision_validation": False,
        "notes": "accepted means spatial joint-position and timing kinematic checks passed; static clearance is NaN",
    }
    materialize_dataset_contract(output_root, robot, manifest)

    dimensions = {
        "H_P": args.spatial_num_control_points,
        "D": robot.dof,
        "K": args.timing_num_control_points,
        "H_T": args.num_phase_points,
        "spatial_degree": args.spatial_degree,
        "timing_degree": args.timing_degree,
    }
    rejection_counts: Counter = Counter()
    accepted_by_variant: Counter = Counter()
    metrics = defaultdict(OnlineMetric)
    start_time = time.monotonic()
    processed_paths = 0
    accepted_paths = 0
    accepted_samples = 0
    stop_index: Optional[int] = None
    report_tag = f"{args.start_index:07d}"
    rejects_path = output_root / f"rejects-{report_tag}.jsonl"

    with LegacyWarehouseReader(source_path, dof=robot.dof) as reader, rejects_path.open(
        "x", encoding="utf-8"
    ) as rejects:
        end_index = len(reader)
        if args.max_paths is not None:
            end_index = min(end_index, args.start_index + args.max_paths)
        if args.start_index >= end_index:
            raise ValueError(
                f"empty source range: start={args.start_index}, end={end_index}, size={len(reader)}"
            )

        for shard_start in range(args.start_index, end_index, args.paths_per_shard):
            shard_stop = min(end_index, shard_start + args.paths_per_shard)
            shard_path = output_root / "shards" / f"part-{shard_start:07d}.hdf5"
            compression = None if args.compression == "none" else args.compression
            with SpaceTimeShardWriter(
                shard_path,
                dimensions=dimensions,
                robot_hashes=robot.hashes,
                compression=compression,
            ) as writer:
                writer.set_attributes(
                    {
                        "source_start_index": shard_start,
                        "source_stop_index": shard_stop,
                    }
                )
                for base_path_id in range(shard_start, shard_stop):
                    processed_paths += 1
                    stop_index = base_path_id + 1
                    try:
                        spatial, task_id = reader.read_canonical_spatial_path(
                            base_path_id,
                            num_control_points=args.spatial_num_control_points,
                            degree=args.spatial_degree,
                        )
                        metrics["spatial_fit_rmse"].update(spatial.fit_rmse)
                        if spatial.fit_rmse > args.spatial_fit_rmse_max:
                            raise ValueError("spatial_fit_rmse")
                        rng = np.random.default_rng(
                            np.random.SeedSequence([args.seed, base_path_id])
                        )
                        references = build_multimodal_retiming_references(
                            spatial.scipy_spline(),
                            dq_max=robot.dq_max,
                            ddq_max=robot.ddq_max,
                            phase=timing_spline.phase,
                            rng=rng,
                            num_toppra_gridpoints=args.num_toppra_gridpoints,
                            variant_names=args.variants,
                        )
                        accepted = []
                        for reference in references:
                            variant_id = DEFAULT_VARIANTS.index(reference.name)
                            try:
                                validated = fit_and_validate_timing_reference(
                                    reference.time_from_start,
                                    spatial_spline=spatial.scipy_spline(),
                                    q_min=robot.q_min,
                                    q_max=robot.q_max,
                                    dq_max=robot.dq_max,
                                    ddq_max=robot.ddq_max,
                                    timing_spline=timing_spline,
                                    duration_min=args.duration_min,
                                    duration_max=args.duration_max,
                                    ratio_tolerance=args.ratio_tolerance,
                                    safety_margin=args.safety_margin,
                                    max_safety_iterations=args.max_safety_iterations,
                                    max_relative_rmse=args.timing_fit_relative_rmse_max,
                                )
                            except Exception as error:
                                reason = f"variant:{reference.name}:{type(error).__name__}:{error}"
                                rejection_counts[reason] += 1
                                rejects.write(
                                    json.dumps(
                                        {
                                            "base_path_id": base_path_id,
                                            "task_id": task_id,
                                            "variant": reference.name,
                                            "reason": reason,
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                                if args.fail_fast:
                                    raise
                                continue
                            if not validated.accepted:
                                reason = f"variant:{reference.name}:{validated.reject_reason}"
                                rejection_counts[reason] += 1
                                rejects.write(
                                    json.dumps(
                                        {
                                            "base_path_id": base_path_id,
                                            "task_id": task_id,
                                            "variant": reference.name,
                                            "reason": validated.reject_reason,
                                            "duration": validated.fit.duration,
                                            "relative_rmse": validated.fit.relative_rmse,
                                            "v_ratio_max": validated.v_ratio_max,
                                            "a_ratio_max": validated.a_ratio_max,
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                                continue
                            accepted.append((variant_id, reference, validated))
                            accepted_by_variant[reference.name] += 1
                            metrics["duration"].update(validated.fit.duration)
                            metrics["timing_relative_rmse"].update(
                                validated.fit.relative_rmse
                            )
                            metrics["v_ratio_max"].update(validated.v_ratio_max)
                            metrics["a_ratio_max"].update(validated.a_ratio_max)
                            metrics["reference_scale"].update(validated.reference_scale)
                        if accepted:
                            writer.append(
                                _records_for_path(
                                    spatial_control_points=spatial.control_points,
                                    spatial_fit_rmse=spatial.fit_rmse,
                                    q_start=spatial.q_start,
                                    q_goal=spatial.q_goal,
                                    task_id=task_id,
                                    base_path_id=base_path_id,
                                    accepted=accepted,
                                )
                            )
                            accepted_paths += 1
                            accepted_samples += len(accepted)
                        else:
                            rejection_counts["base_path:no_accepted_variant"] += 1
                    except Exception as error:
                        reason = f"base_path:{type(error).__name__}:{error}"
                        rejection_counts[reason] += 1
                        rejects.write(
                            json.dumps(
                                {
                                    "base_path_id": base_path_id,
                                    "reason": reason,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        if args.fail_fast:
                            raise

                    if processed_paths % args.report_every == 0:
                        elapsed = time.monotonic() - start_time
                        print(
                            f"processed={processed_paths} accepted_paths={accepted_paths} "
                            f"samples={accepted_samples} paths_per_s={processed_paths / elapsed:.2f}",
                            flush=True,
                        )

    all_base_path_ids = _collect_output_base_path_ids(output_root / "shards")
    splits = write_grouped_splits(
        output_root / "splits", all_base_path_ids, seed=args.seed
    )
    elapsed = time.monotonic() - start_time
    report = {
        "schema_version": "spacetime_mpd_generation_report_v1",
        "arguments": _json_compatible_args(args),
        "source_stop_index": stop_index,
        "processed_paths": processed_paths,
        "accepted_paths": accepted_paths,
        "accepted_samples": accepted_samples,
        "accepted_by_variant": dict(sorted(accepted_by_variant.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "metrics": {name: metric.to_dict() for name, metric in sorted(metrics.items())},
        "split_base_path_counts": {name: len(values) for name, values in splits.items()},
        "elapsed_seconds": elapsed,
        "paths_per_second": processed_paths / elapsed,
    }
    report_path = output_root / f"generation-report-{report_tag}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    args = _parse_args(repository_root)
    report = generate(args, repository_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
