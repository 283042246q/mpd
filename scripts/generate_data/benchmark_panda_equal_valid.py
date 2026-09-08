#!/usr/bin/env python3
"""Generate a fixed number of final-valid Panda warehouse trajectories.

The native Panda pipeline ends after RRTConnect + PathSimplifier + interpolation.
For an apples-to-apples success count with Marvin, this benchmark additionally
audits the dense raw path and a 22-control-point degree-5 B-spline in PyBullet.
Those extra stages are reported separately from native Panda generation work.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
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
from pb_ompl.pb_ompl import fit_bspline_to_path, ob, og
from scripts.generate_data.generate_trajectories import GenerateDataOMPL, get_random_pose_from_region


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "data_generation_cfgs/EnvWarehouse-RobotPanda_v01.yaml"
STAGE_KEYS = (
    "initialization_seconds",
    "endpoint_sampling_seconds",
    "rrt_solve_seconds",
    "path_simplify_seconds",
    "path_interpolate_seconds",
    "path_densify_seconds",
    "path_pybullet_audit_seconds",
    "spline_fit_evaluate_seconds",
    "spline_densify_seconds",
    "spline_pybullet_audit_seconds",
    "cleanup_seconds",
    "other_seconds",
)


def dense_path(path, max_step):
    started = time.perf_counter()
    pieces = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / max_step)))
        pieces.append(a + np.linspace(0.0, 1.0, count + 1)[1:, None] * (b - a))
    return np.concatenate(pieces), time.perf_counter() - started


def panda_plan_once(worker, start, goal, planner_time, interpolate_num):
    """Equivalent to PbOMPL.plan_start_goal, with timers around its stages."""
    interface = worker.pbompl_interface
    interface.robot.set_state(start)
    interface.ss.clear()
    state_start, state_goal = ob.State(interface.space), ob.State(interface.space)
    for index, (q_start, q_goal) in enumerate(zip(start, goal)):
        state_start[index], state_goal[index] = float(q_start), float(q_goal)
    interface.ss.setStartAndGoalStates(state_start, state_goal)

    started = time.perf_counter()
    interface.ss.solve(planner_time)
    rrt_seconds = time.perf_counter() - started
    if not interface.ss.haveExactSolutionPath():
        return None, {
            "rrt_solve_seconds": rrt_seconds,
            "path_simplify_seconds": 0.0,
            "path_interpolate_seconds": 0.0,
        }

    geometric_path = interface.ss.getSolutionPath()
    started = time.perf_counter()
    og.PathSimplifier(interface.si).simplify(geometric_path, maxTime=0.1)
    simplify_seconds = time.perf_counter() - started
    started = time.perf_counter()
    geometric_path.interpolate(interpolate_num)
    path = np.asarray([interface.state_to_list(state) for state in geometric_path.getStates()])
    interpolate_seconds = time.perf_counter() - started
    return path, {
        "rrt_solve_seconds": rrt_seconds,
        "path_simplify_seconds": simplify_seconds,
        "path_interpolate_seconds": interpolate_seconds,
    }


def run_shard(config, seed, start_task_id, final_count, planner_time, max_attempt_factor):
    regions = config["pose_regions"]
    names = tuple(regions)
    accepted = 0
    attempts = 0
    events = []
    shard_started = time.perf_counter()
    max_attempts = final_count * max_attempt_factor
    while accepted < final_count:
        if attempts >= max_attempts:
            raise RuntimeError(f"Panda shard {start_task_id}: only {accepted}/{final_count} after {attempts} attempts")
        attempts += 1
        task_id = start_task_id + accepted
        attempt_started = time.perf_counter()
        task_seed = int(np.random.SeedSequence([seed, start_task_id, task_id, attempts]).generate_state(1)[0])
        fix_random_seed(task_seed)
        event = {
            "attempt": attempts,
            "task_id": task_id,
            "direction": "random_to_placement" if task_id % 10 < 5 else "placement_to_placement",
            "native_success": False,
            "final_success": False,
        }
        for key in STAGE_KEYS:
            event[key] = 0.0

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
            event["validity_resolution"] = float(worker.pbompl_interface.si.getStateValidityCheckingResolution())
            event["planner_range_configured"] = "automatic"

            started = time.perf_counter()
            if event["direction"] == "random_to_placement":
                source_name = "random"
                goal_name = names[int(np.random.randint(len(names)))]
                ee_start = None
            else:
                source_name, goal_name = np.random.choice(names, size=2, replace=False)
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
                event["error"] = repr(error)
            event["endpoint_sampling_seconds"] = time.perf_counter() - started
            event["source_region"] = str(source_name)
            event["goal_region"] = str(goal_name)
            event["status"] = "endpoint_failure"
            if starts and goals:
                path, timings = panda_plan_once(worker, starts[0], goals[0], planner_time, 128)
                event.update(timings)
                event["planner_range_detected"] = float(worker.pbompl_interface.planner.getRange())
                event["status"] = "rrt_no_exact_solution"
                if path is not None:
                    event["native_success"] = True
                    dense, event["path_densify_seconds"] = dense_path(path, 0.025)
                    started = time.perf_counter()
                    raw_valid = all(
                        worker.pbompl_interface.is_state_valid(q, max_distance=0.02, check_bounds=True) for q in dense
                    )
                    event["path_pybullet_audit_seconds"] = time.perf_counter() - started
                    event["path_dense_states"] = len(dense)
                    event["status"] = "path_rejected"
                    if raw_valid:
                        started = time.perf_counter()
                        try:
                            knots, controls, degree = fit_bspline_to_path(
                                path,
                                bspline_degree=5,
                                bspline_num_control_points=22,
                                bspline_zero_vel_at_start_and_goal=True,
                                bspline_zero_acc_at_start_and_goal=True,
                            )
                            spline = BSpline(knots, np.asarray(controls).T, degree)(np.linspace(0.0, 1.0, 512))
                            endpoint_valid = np.allclose(spline[[0, -1]], path[[0, -1]], atol=1e-5)
                        except (ValueError, np.linalg.LinAlgError):
                            spline = None
                            endpoint_valid = False
                        event["spline_fit_evaluate_seconds"] = time.perf_counter() - started
                        if spline is not None and endpoint_valid:
                            spline_dense, event["spline_densify_seconds"] = dense_path(spline, 0.025)
                            started = time.perf_counter()
                            event["final_success"] = all(
                                worker.pbompl_interface.is_state_valid(q, max_distance=0.02, check_bounds=True)
                                for q in spline_dense
                            )
                            event["spline_pybullet_audit_seconds"] = time.perf_counter() - started
                            event["spline_dense_states"] = len(spline_dense)
                        event["status"] = "final_accepted" if event["final_success"] else "spline_rejected"
        finally:
            started = time.perf_counter()
            worker.terminate()
            event["cleanup_seconds"] = time.perf_counter() - started

        measured = sum(event[key] for key in STAGE_KEYS if key != "other_seconds")
        event["attempt_wall_seconds"] = time.perf_counter() - attempt_started
        event["other_seconds"] = max(0.0, event["attempt_wall_seconds"] - measured)
        events.append(event)
        if event["final_success"]:
            accepted += 1
        print(
            f"[panda shard {start_task_id:09d}] accepted {accepted}/{final_count}; "
            f"attempt {attempts}: {event['status']}",
            flush=True,
        )
    return {"start_task_id": start_task_id, "wall_seconds": time.perf_counter() - shard_started, "events": events}


def summarize(shards, global_wall_seconds):
    events = [event for shard in shards for event in shard["events"]]
    totals = {key: sum(event.get(key, 0.0) for event in events) for key in STAGE_KEYS}
    accounted = sum(totals.values())
    return {
        "final_trajectories": sum(event["final_success"] for event in events),
        "native_exact_trajectories": sum(event["native_success"] for event in events),
        "attempts": len(events),
        "statuses": dict(Counter(event["status"] for event in events)),
        "global_wall_seconds": global_wall_seconds,
        "slowest_shard_wall_seconds": max(shard["wall_seconds"] for shard in shards),
        "worker_accumulated_seconds": accounted,
        "stage_totals_seconds": totals,
        "stage_percent_worker_time": {key: 100.0 * value / accounted for key, value in totals.items()},
        "validity_resolution": sorted({event["validity_resolution"] for event in events}),
        "planner_range_configured": "automatic",
        "planner_range_detected": sorted(
            {event["planner_range_detected"] for event in events if "planner_range_detected" in event}
        ),
        "definition": "RRT exact + simplified raw path dense PyBullet audit + fitted spline dense PyBullet audit",
        "native_difference": "raw/spline audits are benchmark additions; native Panda generation ends after interpolation",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-trajectories", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--planner-time", type=float, default=10.0)
    parser.add_argument("--max-attempt-factor", type=int, default=30)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.num_trajectories <= 0 or args.num_trajectories % 10 or args.workers <= 0:
        raise ValueError("num-trajectories must be a positive multiple of 10 and workers must be positive")
    config = yaml.safe_load(args.config.read_text())
    blocks = args.num_trajectories // 10
    worker_count = min(args.workers, blocks)
    base, remainder = divmod(blocks, worker_count)
    counts = [(base + (index < remainder)) * 10 for index in range(worker_count)]
    jobs = []
    start = 0
    for count in counts:
        jobs.append((start, count))
        start += count

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn")) as pool:
        futures = [
            pool.submit(
                run_shard,
                config,
                args.seed,
                start_task_id,
                count,
                args.planner_time,
                args.max_attempt_factor,
            )
            for start_task_id, count in jobs
        ]
        shards = [future.result() for future in as_completed(futures)]
    global_wall = time.perf_counter() - started
    shards.sort(key=lambda shard: shard["start_task_id"])
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "config": str(args.config),
        "seed": args.seed,
        "workers": worker_count,
        "planner_time_seconds": args.planner_time,
        "shards": shards,
    }
    report["summary"] = summarize(shards, global_wall)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
