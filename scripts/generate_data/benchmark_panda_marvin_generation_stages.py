#!/usr/bin/env python3
"""Stage-level benchmark of the native Panda and Marvin warehouse generators.

The benchmark uses a deterministic 5:5 direction schedule and exactly one RRT
call for each successfully sampled endpoint pair.  Panda's native launcher does
not validate fitted splines, so an additional, separately reported spline audit
is performed to expose that difference without changing the native outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.interpolate import BSpline
import torch
import yaml

from experiment_launcher.utils import fix_random_seed
from pb_ompl.pb_ompl import fit_bspline_to_path
from scripts.generate_data.generate_marvin_warehouse_bimanual import (
    DEFAULT_CONFIG as MARVIN_CONFIG,
    MarvinWarehouseGenerator,
    file_sha256,
    task_spec,
)
from scripts.generate_data.generate_trajectories import GenerateDataOMPL, get_random_pose_from_region

PANDA_CONFIG = Path(__file__).resolve().parents[2] / "data_generation_cfgs/EnvWarehouse-RobotPanda_v01.yaml"


def _dense(path, max_step=0.025):
    samples = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / max_step)))
        samples.append(a + np.linspace(0.0, 1.0, n + 1)[1:, None] * (b - a))
    return np.concatenate(samples)


def _summary(events, native_success_key="native_success", final_success_key="final_success"):
    totals = Counter()
    for event in events:
        for key, value in event.items():
            if key.endswith("_seconds") and isinstance(value, (float, int)):
                totals[key] += value
    native = sum(bool(event.get(native_success_key)) for event in events)
    final = sum(bool(event.get(final_success_key)) for event in events)
    return {
        "tasks": len(events),
        "statuses": dict(Counter(event["status"] for event in events)),
        "native_success": native,
        "final_success": final,
        "native_success_rate": native / len(events),
        "final_success_rate": final / len(events),
        "stage_totals_seconds": dict(totals),
        "stage_means_per_task_seconds": {key: value / len(events) for key, value in totals.items()},
    }


def benchmark_panda(config, seed, tasks, planner_time):
    regions = config["pose_regions"]
    region_names = tuple(regions)
    events = []
    wall_start = time.perf_counter()
    for task_id in range(tasks):
        event = {
            "task_id": task_id,
            "direction": "random_to_placement" if task_id % 10 < 5 else "placement_to_placement",
            "native_success": False,
            "final_success": False,
        }
        task_seed = int(np.random.SeedSequence([seed, task_id]).generate_state(1)[0])
        fix_random_seed(task_seed)
        started = time.perf_counter()
        worker = GenerateDataOMPL(
            env_id="EnvWarehouse",
            robot_id="RobotPanda",
            planner="RRTConnect",
            min_distance_robot_env=0.02,
            tensor_args={"device": "cpu", "dtype": torch.float32},
            gripper=True,
            pybullet_mode="DIRECT",
            debug=False,
        )
        event["initialization_seconds"] = time.perf_counter() - started
        try:
            started = time.perf_counter()
            if event["direction"] == "random_to_placement":
                source_name = "random"
                goal_name = region_names[int(np.random.randint(len(region_names)))]
                ee_start = None
            else:
                source_name, goal_name = np.random.choice(region_names, size=2, replace=False)
                ee_start = get_random_pose_from_region(regions[source_name])
            ee_goal = get_random_pose_from_region(regions[goal_name])
            try:
                starts, goals = worker.get_start_and_goal_states(
                    q_pos_start=None,
                    ee_pose_start=ee_start,
                    q_pos_goal=None,
                    ee_pose_goal=ee_goal,
                    n_joint_position_goal=1,
                    sample_joint_position_goals_with_same_ee_pose=False,
                    min_distance_q_pos_start_goal=0.5,
                    debug=False,
                )
            except Exception as error:
                starts, goals = [], []
                event["sampling_error"] = repr(error)
            event["endpoint_sampling_seconds"] = time.perf_counter() - started
            event["source_region"] = str(source_name)
            event["goal_region"] = str(goal_name)
            event["status"] = "endpoint_failure"
            if starts and goals:
                started = time.perf_counter()
                result = worker.run(
                    num_trajectories=1,
                    max_tries=1,
                    joint_position_start=starts[0],
                    joint_position_goal=goals[0],
                    planner_allowed_time=planner_time,
                    interpolate_num=128,
                    simplify_path=True,
                    fit_bspline=False,
                    debug=False,
                )
                event["rrt_and_native_postprocess_seconds"] = time.perf_counter() - started
                event["status"] = "rrt_no_exact_solution"
                if result:
                    event["native_success"] = True
                    event["status"] = "native_accepted"
                    path = np.asarray(result[0]["sol_path"])
                    started = time.perf_counter()
                    try:
                        tt, cc, degree = fit_bspline_to_path(
                            path,
                            bspline_degree=5,
                            bspline_num_control_points=22,
                            bspline_zero_vel_at_start_and_goal=True,
                            bspline_zero_acc_at_start_and_goal=True,
                        )
                        spline = BSpline(tt, np.asarray(cc).T, degree)(np.linspace(0.0, 1.0, 512))
                        event["final_success"] = bool(
                            np.allclose(spline[[0, -1]], path[[0, -1]], atol=1e-5)
                            and all(
                                worker.pbompl_interface.is_state_valid(q, max_distance=0.02, check_bounds=True)
                                for q in _dense(spline)
                            )
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        event["final_success"] = False
                    event["extra_matched_spline_audit_seconds"] = time.perf_counter() - started
                    event["status"] = "final_accepted" if event["final_success"] else "extra_spline_rejected"
        finally:
            worker.terminate()
        event["total_seconds"] = sum(
            value for key, value in event.items() if key.endswith("_seconds") and key != "total_seconds"
        )
        events.append(event)
        print("panda", task_id, event["status"], flush=True)
    summary = _summary(events)
    summary.update(
        wall_seconds=time.perf_counter() - wall_start,
        native_accounted_seconds=sum(
            event.get("initialization_seconds", 0.0)
            + event.get("endpoint_sampling_seconds", 0.0)
            + event.get("rrt_and_native_postprocess_seconds", 0.0)
            for event in events
        ),
        matched_final_accounted_seconds=sum(event["total_seconds"] for event in events),
        execution_model="new PyBullet/OMPL instance per task, matching task_batch_size=1",
        extra_spline_audit_is_native=False,
    )
    return {"events": events, "summary": summary}


def benchmark_marvin(config, seed, tasks, planner_time):
    config = dict(config)
    config["planner_allowed_time"] = planner_time
    init_start = time.perf_counter()
    generator = MarvinWarehouseGenerator(config, seed, progress_label="benchmark")
    initialization = time.perf_counter() - init_start
    events = []
    wall_start = time.perf_counter()
    try:
        for task_id in range(tasks):
            mode, direction = task_spec(task_id)
            event = {
                "task_id": task_id,
                "mode": mode,
                "direction": direction,
                "native_success": False,
                "final_success": False,
                "initialization_seconds": initialization if task_id == 0 else 0.0,
            }
            generator.deadline = time.perf_counter() + float(config.get("task_timeout_seconds", 300))
            before_target = generator.stats["target_seconds"]
            started = time.perf_counter()
            sampled = generator._sample_task(mode, direction)
            event["endpoint_sampling_seconds"] = time.perf_counter() - started
            event["target_ik_seconds"] = generator.stats["target_seconds"] - before_target
            event["random_and_endpoint_filter_seconds"] = max(
                0.0, event["endpoint_sampling_seconds"] - event["target_ik_seconds"]
            )
            event["status"] = "endpoint_failure"
            if sampled is not None:
                q_start, q_goal, _, _ = sampled
                detail_keys = (
                    "path_simplify_seconds",
                    "path_resample_seconds",
                    "path_densify_seconds",
                    "path_torch_audit_seconds",
                    "path_pybullet_audit_seconds",
                    "path_dense_states",
                )
                before_detail = {key: generator.stats[key] for key in detail_keys}
                before_rrt = generator.stats["rrt_seconds"]
                before_no_exact = generator.stats["rrt_no_exact_solution"]
                before_path_rejected = generator.stats["path_rejected"]
                started = time.perf_counter()
                path = generator.plan_once(q_start, q_goal, mode)
                planning_elapsed = time.perf_counter() - started
                event["rrt_solve_seconds"] = generator.stats["rrt_seconds"] - before_rrt
                event["rrt_postprocess_and_path_audit_seconds"] = max(
                    0.0, planning_elapsed - event["rrt_solve_seconds"]
                )
                for key in detail_keys:
                    event[key] = generator.stats[key] - before_detail[key]
                if generator.stats["rrt_no_exact_solution"] > before_no_exact:
                    event["status"] = "rrt_no_exact_solution"
                elif generator.stats["path_rejected"] > before_path_rejected:
                    event["status"] = "path_rejected"
                else:
                    event["status"] = "path_accepted"
                if path is not None:
                    event["native_success"] = True
                    detail_keys = (
                        "spline_fit_evaluate_seconds",
                        "spline_densify_seconds",
                        "spline_torch_audit_seconds",
                        "spline_pybullet_audit_seconds",
                        "spline_dense_states",
                    )
                    before_detail = {key: generator.stats[key] for key in detail_keys}
                    started = time.perf_counter()
                    spline = generator.validated_spline(path)
                    event["spline_fit_and_audit_seconds"] = time.perf_counter() - started
                    for key in detail_keys:
                        event[key] = generator.stats[key] - before_detail[key]
                    event["final_success"] = spline is not None
                    event["status"] = "final_accepted" if spline is not None else "spline_rejected"
            event["total_seconds"] = (
                event.get("initialization_seconds", 0.0)
                + event.get("endpoint_sampling_seconds", 0.0)
                + event.get("rrt_solve_seconds", 0.0)
                + event.get("rrt_postprocess_and_path_audit_seconds", 0.0)
                + event.get("spline_fit_and_audit_seconds", 0.0)
            )
            events.append(event)
            print("marvin", task_id, mode, event["status"], flush=True)
    finally:
        stats = dict(generator.stats)
        generator.close()
    summary = _summary(events)
    summary.update(
        wall_seconds=time.perf_counter() - wall_start + initialization,
        accounted_seconds=sum(event["total_seconds"] for event in events),
        initialization_once_seconds=initialization,
        execution_model="one PyBullet scene per shard; a fresh OMPL SimpleSetup per RRT",
        generator_stats=stats,
    )
    return {"events": events, "summary": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--planner-time", type=float, default=10.0)
    parser.add_argument("--robots", nargs="+", choices=("panda", "marvin"), default=("panda", "marvin"))
    parser.add_argument("--marvin-no-simplify", action="store_true")
    args = parser.parse_args(argv)
    if args.tasks <= 0 or args.tasks % 10 or args.planner_time <= 0:
        raise ValueError("tasks must be a positive multiple of 10 and planner time must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    panda_config = yaml.safe_load(PANDA_CONFIG.read_text())
    marvin_config = yaml.safe_load(MARVIN_CONFIG.read_text())
    if args.marvin_no_simplify:
        marvin_config["simplify_path"] = False
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "seed": args.seed,
        "tasks": args.tasks,
        "planner_time_seconds": args.planner_time,
        "schedule": "first 5/10 random-to-placement; last 5/10 placement-to-different-placement",
        "panda_config_sha256": file_sha256(PANDA_CONFIG),
        "marvin_config_sha256": file_sha256(MARVIN_CONFIG),
    }
    if "panda" in args.robots:
        report["panda"] = benchmark_panda(panda_config, args.seed, args.tasks, args.planner_time)
    if "marvin" in args.robots:
        report["marvin"] = benchmark_marvin(marvin_config, args.seed, args.tasks, args.planner_time)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({name: report[name]["summary"] for name in args.robots}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
