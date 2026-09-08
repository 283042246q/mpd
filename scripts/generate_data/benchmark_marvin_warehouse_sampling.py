#!/usr/bin/env python3
"""Compare joint+FK+filter and region+bounded-IK+filter with equal time budgets.

Endpoint experiment: other arm at home, fixed time per arm/region/seed.
Pipeline experiment: fresh random reference, one sampling call per scheduled
task, one RRT per valid pair, then dense path and spline validation. Failed
tasks are counted rather than retried, so rates have a fixed denominator.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.spatial.distance import pdist
import yaml

from scripts.generate_data.generate_marvin_warehouse_bimanual import (
    DEFAULT_CONFIG,
    ARM_REGIONS,
    ARM_SLICES,
    MarvinWarehouseGenerator,
    task_spec,
    _write_dataset,
    file_sha256,
)


def diversity(values):
    if len(values) < 2:
        return {"count": len(values), "pairwise_rms": None, "mean_std": None}
    x = np.array(values).reshape(len(values), -1)
    return dict(
        count=len(values),
        pairwise_rms=float(np.mean(pdist(x)) / np.sqrt(x.shape[1])),
        mean_std=float(np.mean(x.std(axis=0))),
    )


def benchmark(config, sampler, seed, endpoint_seconds, task_seconds, tasks, output, equal_task_time=False):
    config = deepcopy(config)
    config["sampler"] = sampler
    generator = MarvinWarehouseGenerator(config, seed)
    report = dict(sampler=sampler, seed=seed, regions={}, pipeline={})
    endpoint_states = {}
    try:
        if not generator.valid(np.zeros(14)):
            raise ValueError("endpoint benchmark requires a valid home reference")
        for arm, regions in ARM_REGIONS.items():
            for name in regions:
                states, positions = [], []
                generator.stats.clear()
                before = time.perf_counter()
                generator.deadline = before + endpoint_seconds
                while time.perf_counter() < generator.deadline:
                    q = generator._target_state(np.zeros(14), arm, name)
                    if q is not None:
                        states.append(q.tolist())
                        positions.append(generator._pose(q, arm).translation.copy().tolist())
                elapsed = time.perf_counter() - before
                endpoint_states[name] = dict(q=states, xyz=positions)
                q_norm = [
                    (np.array(q)[ARM_SLICES[arm]] / generator.robot.joint_ranges_np[ARM_SLICES[arm]]).tolist()
                    for q in states
                ]
                report["regions"][name] = dict(
                    seconds=elapsed,
                    stats=dict(generator.stats),
                    accepted_per_second=len(states) / elapsed,
                    acceptance_per_candidate=len(states) / max(1, generator.stats["target_candidates"]),
                    joint_diversity=diversity(q_norm),
                    tcp_diversity=diversity(positions),
                )
                print(
                    sampler,
                    seed,
                    name,
                    "accepted",
                    len(states),
                    "candidates",
                    generator.stats["target_candidates"],
                    flush=True,
                )
        generator.stats.clear()
        if equal_task_time:
            # A separate comparison removes production attempt caps: both
            # samplers may use the entire per-task wall-clock budget.
            config.update(ik_tries=1000000000, fk_tries=1000000000)
        generator.rng = np.random.default_rng(seed + 100000)  # decouple timing-based endpoint sample counts
        paths, metadata, events = [], [], []
        before = time.perf_counter()
        for task_id in range(tasks):
            mode, direction = task_spec(task_id)
            event = dict(task_id=task_id, mode=mode, direction=direction)
            start = time.perf_counter()
            generator.deadline = start + task_seconds
            sampled = generator._sample_task(mode, direction)
            event["sampling_seconds"] = time.perf_counter() - start
            event["status"] = "endpoint_failure"
            if sampled is not None:
                q_start, q_goal, source, goal = sampled
                start = time.perf_counter()
                path = generator.plan_once(q_start, q_goal, mode)
                event["planning_and_path_check_seconds"] = time.perf_counter() - start
                event["status"] = "rrt_or_path_failure"
                if path is not None:
                    start = time.perf_counter()
                    spline = generator.validated_spline(path)
                    event["spline_check_seconds"] = time.perf_counter() - start
                    event["status"] = "spline_failure"
                    if spline is not None:
                        event["status"] = "accepted"
                        paths.append(path.copy())
                        item = dict(
                            task_id=task_id,
                            task_mode=mode,
                            direction=direction,
                            q_start=q_start,
                            q_goal=q_goal,
                            bspline=spline,
                            planning_time=event["planning_and_path_check_seconds"],
                        )
                        for arm in ARM_SLICES:
                            item[f"source_region_{arm}"] = source.get(arm, "inactive")
                            item[f"goal_region_{arm}"] = goal.get(arm, "inactive")
                        metadata.append(item)
            events.append(event)
            print(sampler, seed, "task", task_id, event["status"], flush=True)
        elapsed = time.perf_counter() - before
        normalized = np.asarray(paths) / generator.robot.joint_ranges_np if paths else []
        residuals = []
        for path in normalized:
            line = np.linspace(path[0], path[-1], len(path))
            residuals.append(path - line)
        report["pipeline"] = dict(
            seconds=elapsed,
            tasks=tasks,
            accepted=len(paths),
            accepted_per_second=len(paths) / elapsed,
            statuses=dict(Counter(e["status"] for e in events)),
            stats=dict(generator.stats),
            events=events,
            trajectory_diversity=diversity(normalized),
            trajectory_residual_diversity=diversity(residuals),
            mean_joint_path_length=(
                float(np.mean([np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in paths])) if paths else None
            ),
        )
        if paths:
            _write_dataset(output / f"{sampler}_{seed}_paths", config, paths, metadata, seed, generator.stats)
        (output / f"{sampler}_{seed}_endpoints.json").write_text(json.dumps(endpoint_states, indent=2))
        (output / f"{sampler}_{seed}.json").write_text(json.dumps(report, indent=2))
    finally:
        generator.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--endpoint-seconds", type=float, default=10.0)
    parser.add_argument("--task-seconds", type=float, default=15.0)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--equal-task-time", action="store_true")
    parser.add_argument("--samplers", nargs="+", choices=("joint_fk", "region_ik"), default=["joint_fk", "region_ik"])
    args = parser.parse_args(argv)
    if args.tasks <= 0 or args.tasks % 10 or args.endpoint_seconds < 0 or args.task_seconds <= 0:
        raise ValueError("positive time budgets and task count divisible by 10 required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text())
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config))
    reports = []
    for seed in args.seeds:
        for sampler in args.samplers:
            reports.append(
                benchmark(
                    config,
                    sampler,
                    seed,
                    args.endpoint_seconds,
                    args.task_seconds,
                    args.tasks,
                    args.output_dir,
                    args.equal_task_time,
                )
            )
    payload = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        config_sha256=file_sha256(args.config),
        endpoint_seconds=args.endpoint_seconds,
        task_seconds=args.task_seconds,
        equal_task_time=args.equal_task_time,
        results=reports,
    )
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
