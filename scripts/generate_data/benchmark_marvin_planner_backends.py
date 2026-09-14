#!/usr/bin/env python3
"""Paired CPU-OMPL/GPU-batch benchmark on saved Marvin endpoint pairs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import yaml

from scripts.generate_data.generate_marvin_warehouse_bimanual import (
    DEFAULT_CONFIG,
    MarvinWarehouseGenerator,
    active_arms,
)

BACKENDS = ("RRTConnect", "GpuBatchRRTConnect")


def _dataset_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    merged = path / "dataset_merged.hdf5"
    if merged.is_file():
        return [merged]
    return sorted(path.glob("shards/*/dataset_merged.hdf5"))


def load_requests(path, per_mode):
    selected = defaultdict(list)
    for dataset in _dataset_files(path):
        with h5py.File(dataset, "r") as data:
            modes = data["task_mode"].asstr()[:]
            task_ids = data["task_id"][:]
            for row, mode in enumerate(modes):
                if len(selected[mode]) >= per_mode:
                    continue
                selected[mode].append(
                    {
                        "task_id": int(task_ids[row]),
                        "task_mode": str(mode),
                        "q_start": data["q_start"][row].astype(float).tolist(),
                        "q_goal": data["q_goal"][row].astype(float).tolist(),
                        "source": str(dataset),
                    }
                )
        if all(len(selected[mode]) >= per_mode for mode in (
            "dual_independent", "left_only", "right_only"
        )):
            break
    missing = {
        mode: per_mode - len(selected[mode])
        for mode in ("dual_independent", "left_only", "right_only")
        if len(selected[mode]) < per_mode
    }
    if missing:
        raise ValueError(f"dataset does not contain enough requests: {missing}")
    return [
        request
        for mode in ("dual_independent", "left_only", "right_only")
        for request in selected[mode]
    ]


def _delta(after, before):
    return {key: float(after[key] - before[key]) for key in set(after) | set(before)}


def run_backend(config, backend, requests):
    config = dict(config)
    config["planner"] = backend
    initialized = time.perf_counter()
    generator = MarvinWarehouseGenerator(config, int(config.get("seed", 0)))
    initialization_seconds = time.perf_counter() - initialized
    records = []
    try:
        if backend == "GpuBatchRRTConnect":
            first = requests[0]
            generator._gpu_collision_mask(
                np.stack((first["q_start"], first["q_goal"]))
            )
            torch.cuda.synchronize(generator.gpu_torch_robot.q_pos_min.device)
        for index, request in enumerate(requests, 1):
            q_start = np.asarray(request["q_start"], dtype=float)
            q_goal = np.asarray(request["q_goal"], dtype=float)
            before_stats = Counter(generator.stats)
            started = time.perf_counter()
            path = generator.plan_once(q_start, q_goal, request["task_mode"])
            raw_path_valid = path is not None
            spline = generator.validated_spline(path) if raw_path_valid else None
            wall_seconds = time.perf_counter() - started
            inactive = sorted(
                set(("left", "right")) - set(active_arms(request["task_mode"]))
            )
            inactive_drift = max(
                (
                    float(np.max(np.abs(path[:, slice(0, 7) if arm == "left" else slice(7, 14)] -
                                        q_start[slice(0, 7) if arm == "left" else slice(7, 14)])))
                    for arm in inactive
                ),
                default=0.0,
            ) if path is not None else None
            record = {
                **request,
                "backend": backend,
                "raw_path_valid": raw_path_valid,
                "spline_valid": spline is not None,
                "wall_seconds": wall_seconds,
                "inactive_joint_max_drift": inactive_drift,
                "stats": _delta(generator.stats, before_stats),
            }
            records.append(record)
            print(
                f"[{backend}] {index}/{len(requests)} {request['task_mode']} "
                f"raw={raw_path_valid} spline={spline is not None} wall={wall_seconds:.3f}s",
                flush=True,
            )
    finally:
        generator.close()
    return initialization_seconds, records


def summarize(records):
    result = {}
    for backend in BACKENDS:
        backend_rows = [row for row in records if row["backend"] == backend]
        if not backend_rows:
            continue
        by_mode = {}
        for mode in ("all", "dual_independent", "left_only", "right_only"):
            rows = backend_rows if mode == "all" else [
                row for row in backend_rows if row["task_mode"] == mode
            ]
            wall = np.asarray([row["wall_seconds"] for row in rows])
            rrt = np.asarray([row["stats"].get("rrt_seconds", 0.0) for row in rows])
            spline_successes = sum(row["spline_valid"] for row in rows)
            by_mode[mode] = {
                "requests": len(rows),
                "raw_path_successes": sum(row["raw_path_valid"] for row in rows),
                "spline_successes": spline_successes,
                "success_rate": spline_successes / len(rows),
                "wall_mean_seconds": float(wall.mean()),
                "wall_median_seconds": float(np.median(wall)),
                "wall_p90_seconds": float(np.quantile(wall, 0.9)),
                "rrt_mean_seconds": float(rrt.mean()),
                "wall_seconds_per_valid_spline": (
                    float(wall.sum() / spline_successes) if spline_successes else None
                ),
            }
        result[backend] = by_mode
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--per-mode", type=int, default=10)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--allowed-time", type=float, default=10.0)
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--gpu-batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.per_mode < 1 or args.allowed_time <= 0 or args.gpu_batch_size < 1:
        raise ValueError("per-mode, allowed-time and gpu-batch-size must be positive")
    config = yaml.safe_load(args.config.read_text())
    config.update(
        planner_allowed_time=args.allowed_time,
        gpu_device=args.gpu_device,
        gpu_rrt_batch_size=args.gpu_batch_size,
    )
    requests = load_requests(args.dataset, args.per_mode)
    records = []
    initialization = {}
    for backend in args.backends:
        seconds, rows = run_backend(config, backend, requests)
        initialization[backend] = seconds
        records.extend(rows)
    report = {
        "schema": "marvin_planner_backend_benchmark/v1",
        "dataset": str(args.dataset),
        "config": str(args.config),
        "paired_requests_per_mode": args.per_mode,
        "planner_allowed_time": args.allowed_time,
        "gpu_batch_size": args.gpu_batch_size,
        "initialization_seconds": initialization,
        "summary": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
