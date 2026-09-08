#!/usr/bin/env python3
"""Paired benchmark for Marvin pre-RRT endpoint filters.

Each candidate endpoint pair is planned exactly once by the production RRT.
All filters are evaluated on that same pair before planning, so a rejected pair
that the baseline later solves provides a directly observed false rejection.
Counterfactual throughput removes RRT/audit work for rejected pairs and retains
the measured sampling and filter work.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
import yaml

from scripts.generate_data.benchmark_marvin_planner_ablation import (
    CollisionPolicy,
    atomic_json,
    gold_certify,
    path_metrics,
    resample_path,
    solve_raw,
    stable_seed,
)
from ompl import util as ou
from scripts.generate_data.generate_marvin_warehouse_bimanual import (
    DEFAULT_CONFIG,
    MarvinWarehouseGenerator,
    PRE_RRT_FILTERS,
    active_arms,
    file_sha256,
    task_spec,
    validate_config,
)


SCHEMA = "marvin_pre_rrt_filter_ablation/v1"
FILTERS = PRE_RRT_FILTERS
DEFAULT_OUTPUT = Path("benchmark_results/marvin_pre_rrt_filter_seed73")


def chunks(count, workers):
    base, remainder = divmod(count, workers)
    result = []
    start = 0
    for index in range(workers):
        size = base + (index < remainder)
        if size:
            result.append(list(range(start, start + size)))
            start += size
    return result


def _filter_result(generator, name, q_start, q_goal):
    if name == "none":
        return {
            "accepted": True,
            "seconds": 0.0,
            "rejection_reason": "accepted",
        }
    previous = generator.config.get("pre_rrt_filter", "none")
    generator.config["pre_rrt_filter"] = name
    try:
        accepted, diagnostics = generator.pre_rrt_accept(q_start, q_goal)
    finally:
        generator.config["pre_rrt_filter"] = previous
    return {
        "accepted": bool(accepted),
        "seconds": float(diagnostics["pre_rrt_filter_seconds"]),
        "rejection_reason": diagnostics["pre_rrt_rejection_reason"],
        "endpoint_environment_clearance": float(diagnostics["endpoint_environment_clearance"]),
        "endpoint_self_clearance": float(diagnostics["endpoint_self_clearance"]),
        **(
            {"sparse_line_valid_fraction": float(diagnostics["sparse_line_valid_fraction"])}
            if "sparse_line_valid_fraction" in diagnostics
            else {}
        ),
    }


def _region_bucket(mode, direction, source, goal):
    parts = [mode, direction]
    for arm in active_arms(mode):
        parts.extend((arm, source[arm], goal[arm]))
    return "|".join(parts)


def collect_chunk(config, output, seed, pair_ids, planner_time, endpoint_attempts):
    ou.RNG.setSeed(stable_seed(seed, "pre-filter-rrt-worker", pair_ids[0]))
    started = time.perf_counter()
    generator = MarvinWarehouseGenerator(
        deepcopy(config), seed=stable_seed(seed, "pre-filter-sampling-worker", pair_ids[0])
    )
    initialization_seconds = time.perf_counter() - started
    policy = CollisionPolicy(generator, "sphere_100")
    try:
        for offset, pair_id in enumerate(pair_ids):
            destination = Path(output) / "pairs" / f"pair_{pair_id:05d}.json"
            if destination.exists():
                continue
            mode, direction = task_spec(pair_id)
            sampling_started = time.perf_counter()
            sampled = None
            attempts = 0
            while sampled is None and attempts < endpoint_attempts:
                attempts += 1
                generator.rng = np.random.default_rng(stable_seed(seed, "endpoint", pair_id, attempts))
                generator.deadline = time.perf_counter() + float(config.get("task_timeout_seconds", 300))
                sampled = generator._sample_task(mode, direction)
            sampling_seconds = time.perf_counter() - sampling_started
            if sampled is None:
                raise RuntimeError(f"pair {pair_id}: no endpoint pair after {endpoint_attempts} attempts")
            q_start, q_goal, source, goal = sampled
            filters = {name: _filter_result(generator, name, q_start, q_goal) for name in FILTERS}

            result = solve_raw(
                generator,
                q_start,
                q_goal,
                mode,
                float(config.get("state_validity_resolution", 0.002)),
                float(config.get("planner_range", 0.35)),
                policy,
                planner_time,
                stable_seed(seed, "pre-filter-rrt", pair_id),
            )
            record = {
                "schema": SCHEMA,
                "pair_id": pair_id,
                "chunk_id": pair_ids[0],
                "mode": mode,
                "direction": direction,
                "source": source,
                "goal": goal,
                "region_bucket": _region_bucket(mode, direction, source, goal),
                "endpoint_attempts": attempts,
                "sampling_seconds": sampling_seconds,
                "filters": filters,
                "exact": bool(result["exact"]),
                "solve_seconds": float(result["solve_seconds"]),
                "collision_stats": result["collision_stats"],
                "gold_path_valid": False,
                "gold_spline_valid": False,
                "gold_final_valid": False,
                "gold_path_audit_seconds": 0.0,
                "gold_spline_audit_seconds": 0.0,
                "joint_path_length": None,
                "normalized_path_length": None,
            }
            if result["exact"]:
                path = resample_path(result["full_vertices"])
                if path is not None:
                    record.update(gold_certify(generator, path))
                    record["joint_path_length"] = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                    record["normalized_path_length"] = path_metrics(
                        path,
                        generator.robot.joint_bounds_low_np,
                        generator.robot.joint_bounds_high_np,
                    )["normalized_length"]
            atomic_json(destination, record)
            print(
                f"[pre-filter {pair_ids[0]:03d}] {offset + 1}/{len(pair_ids)} pair={pair_id} "
                f"filters={''.join('Y' if filters[name]['accepted'] else 'N' for name in FILTERS)} "
                f"exact={record['exact']} final={record['gold_final_valid']} "
                f"rrt={record['solve_seconds']:.3f}s",
                flush=True,
            )
        return {
            "chunk_id": pair_ids[0],
            "count": len(pair_ids),
            "initialization_seconds": initialization_seconds,
            "collision_policy": policy.metadata,
        }
    finally:
        generator.close()


def _distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _active_region_counts(records):
    source, goal = Counter(), Counter()
    for record in records:
        for arm in active_arms(record["mode"]):
            source[record["source"][arm]] += 1
            goal[record["goal"][arm]] += 1
    return {"source": dict(source), "goal": dict(goal)}


def summarize_filter(records, chunks_meta, name, workers):
    entered = [record for record in records if record["filters"][name]["accepted"]]
    rejected = [record for record in records if not record["filters"][name]["accepted"]]
    exact = [record for record in entered if record["exact"]]
    final = [record for record in entered if record["gold_final_valid"]]
    baseline_exact = [record for record in records if record["exact"]]
    baseline_final = [record for record in records if record["gold_final_valid"]]
    exact_false_rejects = [record for record in rejected if record["exact"]]
    final_false_rejects = [record for record in rejected if record["gold_final_valid"]]

    chunk_times = {item["chunk_id"]: float(item["initialization_seconds"]) for item in chunks_meta}
    for record in records:
        elapsed = record["sampling_seconds"] + record["filters"][name]["seconds"]
        if record["filters"][name]["accepted"]:
            elapsed += (
                record["solve_seconds"]
                + record["gold_path_audit_seconds"]
                + record["gold_spline_audit_seconds"]
            )
        chunk_times[record["chunk_id"]] += elapsed
    counterfactual_wall = max(chunk_times.values())
    worker_seconds = sum(chunk_times.values())
    final_buckets = {record["region_bucket"] for record in final}
    baseline_buckets = {record["region_bucket"] for record in baseline_final}
    candidate_regions = {
        region
        for record in records
        for arm in active_arms(record["mode"])
        for region in (record["source"][arm], record["goal"][arm])
        if region != "random"
    }
    final_regions = {
        region
        for record in final
        for arm in active_arms(record["mode"])
        for region in (record["source"][arm], record["goal"][arm])
        if region != "random"
    }
    rejection_reasons = Counter(record["filters"][name]["rejection_reason"] for record in rejected)
    return {
        "filter": name,
        "candidates": len(records),
        "entered_rrt": len(entered),
        "pre_rrt_accept_rate": len(entered) / len(records),
        "rrt_exact": len(exact),
        "rrt_exact_rate_after_filter": len(exact) / len(entered) if entered else None,
        "final_valid": len(final),
        "final_rate_per_candidate": len(final) / len(records),
        "counterfactual_three_worker_wall_seconds": counterfactual_wall,
        "counterfactual_final_trajectories_per_minute": 60.0 * len(final) / counterfactual_wall,
        "worker_seconds": worker_seconds,
        "filter_seconds": sum(record["filters"][name]["seconds"] for record in records),
        "rejection_reasons": dict(rejection_reasons),
        "exact_false_rejects": len(exact_false_rejects),
        "exact_false_reject_rate_of_baseline_exact": (
            len(exact_false_rejects) / len(baseline_exact) if baseline_exact else None
        ),
        "exact_fraction_among_rejections": len(exact_false_rejects) / len(rejected) if rejected else 0.0,
        "final_false_rejects": len(final_false_rejects),
        "final_false_reject_rate_of_baseline_final": (
            len(final_false_rejects) / len(baseline_final) if baseline_final else None
        ),
        "joint_path_length": _distribution([record["joint_path_length"] for record in final]),
        "normalized_path_length": _distribution([record["normalized_path_length"] for record in final]),
        "region_transition_buckets": len(final_buckets),
        "baseline_final_region_transition_buckets_retained": (
            len(final_buckets & baseline_buckets) / len(baseline_buckets) if baseline_buckets else None
        ),
        "placement_regions_covered": sorted(final_regions),
        "placement_region_coverage_rate": len(final_regions) / len(candidate_regions),
        "region_counts": _active_region_counts(final),
        "by_mode": {
            mode: {
                "entered_rrt": sum(record["mode"] == mode for record in entered),
                "rrt_exact": sum(record["mode"] == mode for record in exact),
                "final_valid": sum(record["mode"] == mode for record in final),
            }
            for mode in ("dual_independent", "left_only", "right_only")
        },
        "workers": workers,
    }


def write_report(output, summary):
    rows = []
    for name in FILTERS:
        item = summary["filters"][name]
        rows.append(
            f"| {name} | {item['entered_rrt']}/{item['candidates']} | "
            f"{100 * item['rrt_exact_rate_after_filter']:.1f}% | {item['final_valid']} | "
            f"{item['counterfactual_final_trajectories_per_minute']:.2f} | "
            f"{100 * item['exact_false_reject_rate_of_baseline_exact']:.1f}% | "
            f"{100 * item['final_false_reject_rate_of_baseline_final']:.1f}% | "
            f"{item['joint_path_length']['mean']:.3f} | {item['region_transition_buckets']} | "
            f"{100 * item['baseline_final_region_transition_buckets_retained']:.1f}% | "
            f"{100 * item['placement_region_coverage_rate']:.1f}% |"
        )
    lines = [
        "# Marvin pre-RRT filter paired ablation",
        "",
        f"Candidates: {summary['count']}; workers: {summary['workers']}; seed: {summary['seed']}.",
        "",
        "Every row uses the same endpoint pairs and their one observed production RRT result. "
        "Throughput is a paired counterfactual: measured sampling/filter time plus RRT and audit time only "
        "for pairs admitted by that filter.",
        "",
        "| filter | entered RRT | exact/entered | final | final/min (3 workers) | exact false reject | final false reject | path length mean rad | region buckets | bucket retention | placement coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "False-reject denominators are all baseline-exact or baseline-final pairs, respectively.",
        "Region buckets encode mode, direction, and every active arm's source/goal region. "
        "Placement coverage is the fraction of candidate placement-region names represented by final trajectories.",
    ]
    (Path(output) / "REPORT.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--planner-time", type=float, default=10.0)
    parser.add_argument("--endpoint-attempts", type=int, default=30)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed pair_*.json records in an existing output directory.",
    )
    args = parser.parse_args(argv)
    if args.count <= 0 or args.count % 10 or args.workers <= 0:
        raise ValueError("count must be a positive multiple of ten; workers must be positive")
    config = yaml.safe_load(args.config.read_text())
    validate_config(config)
    run_contract = {
        "schema": SCHEMA,
        "count": args.count,
        "workers": args.workers,
        "seed": args.seed,
        "planner_time_seconds": args.planner_time,
        "endpoint_attempts": args.endpoint_attempts,
        # Compare parsed values rather than the file hash so comment-only YAML
        # edits do not invalidate an otherwise identical interrupted run.
        "config": config,
    }
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        contract_path = args.output / "run_contract.json"
        summary_path = args.output / "summary.json"
        if contract_path.is_file():
            prior = json.loads(contract_path.read_text())
        elif summary_path.is_file():
            old_summary = json.loads(summary_path.read_text())
            for key in ("schema", "count", "workers", "seed", "planner_time_seconds"):
                if old_summary.get(key) != run_contract[key]:
                    raise ValueError(f"resume setting {key} differs from existing summary")
            expected_thresholds = {
                key: config[key]
                for key in (
                    "pre_rrt_endpoint_environment_clearance",
                    "pre_rrt_endpoint_self_clearance",
                    "pre_rrt_sparse_line_points",
                    "pre_rrt_sparse_line_min_valid_fraction",
                )
            }
            if old_summary.get("thresholds") != expected_thresholds:
                raise ValueError("resume filter thresholds differ from existing summary")
            prior = run_contract
        elif any((args.output / "pairs").glob("pair_*.json")):
            raise ValueError("cannot safely resume legacy partial output without run metadata")
        else:
            prior = run_contract
        if prior != run_contract:
            raise ValueError(f"resume settings differ from existing run: {prior} != {run_contract}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    atomic_json(args.output / "run_contract.json", run_contract)
    pair_chunks = chunks(args.count, min(args.workers, args.count))
    jobs = [
        (config, str(args.output), args.seed, pair_ids, args.planner_time, args.endpoint_attempts)
        for pair_ids in pair_chunks
    ]
    wall_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(collect_chunk, *job) for job in jobs]
        chunks_meta = [future.result() for future in as_completed(futures)]
    measured_wall = time.perf_counter() - wall_started
    records = [
        json.loads((args.output / "pairs" / f"pair_{pair_id:05d}.json").read_text())
        for pair_id in range(args.count)
    ]
    summary = {
        "schema": SCHEMA,
        "platform": platform.platform(),
        "count": args.count,
        "workers": args.workers,
        "seed": args.seed,
        "planner_time_seconds": args.planner_time,
        "resumed": args.resume,
        "measured_invocation_wall_seconds": measured_wall,
        "measured_baseline_wall_seconds": None if args.resume else measured_wall,
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "thresholds": {
            key: config[key]
            for key in (
                "pre_rrt_endpoint_environment_clearance",
                "pre_rrt_endpoint_self_clearance",
                "pre_rrt_sparse_line_points",
                "pre_rrt_sparse_line_min_valid_fraction",
            )
        },
        "chunks": sorted(chunks_meta, key=lambda item: item["chunk_id"]),
        "filters": {
            name: summarize_filter(records, chunks_meta, name, args.workers) for name in FILTERS
        },
    }
    atomic_json(args.output / "summary.json", summary)
    write_report(args.output, summary)
    print(json.dumps(summary["filters"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
