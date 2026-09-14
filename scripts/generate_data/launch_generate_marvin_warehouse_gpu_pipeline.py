#!/usr/bin/env python3
"""Stable mixed-shard Marvin generation with isolated CPU/GPU actors."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import yaml

from scripts.generate_data.marvin_gpu_pipeline_actors import actor_main
from scripts.generate_data.marvin_gpu_task_contract import (
    TaskContract,
    stable_seed,
)


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY
    / "data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-independent.yaml"
)
MODE_BATCH_CONFIG = {
    "dual_independent": "gpu_query_batch_size_dual",
    "left_only": "gpu_query_batch_size_left",
    "right_only": "gpu_query_batch_size_right",
}


class RestartableActor:
    """One-request-at-a-time spawned actor with bounded IPC and restart."""

    def __init__(self, context, role, config, name, timeout, max_restarts):
        self.context = context
        self.role = role
        self.config = config
        self.name = name
        self.timeout = float(timeout)
        self.max_restarts = int(max_restarts)
        if self.timeout <= 0 or self.max_restarts < 0:
            raise ValueError("actor timeout must be positive and restart count nonnegative")
        self.restart_count = 0
        self.process = None
        self.requests = None
        self.responses = None
        self.pending = None
        self._start()

    def _start(self):
        self.requests = self.context.Queue(maxsize=1)
        self.responses = self.context.Queue(maxsize=1)
        self.process = self.context.Process(
            target=actor_main,
            args=(self.role, self.config, self.requests, self.responses),
            name=self.name,
        )
        self.process.start()
        ready = self._wait_response()
        if ready is None:
            self._terminate()
            raise RuntimeError(f"{self.name} did not initialize within {self.timeout}s")
        if not ready.get("ready"):
            self._terminate()
            raise RuntimeError(
                f"{self.name} initialization failed: {ready.get('error_type')}: "
                f"{ready.get('message')}\n{ready.get('traceback', '')}"
            )

    def send(self, message):
        if self.pending is not None:
            raise RuntimeError(f"{self.name} already has an in-flight request")
        self.pending = message
        self.requests.put(message, timeout=self.timeout)

    def receive(self):
        if self.pending is None:
            raise RuntimeError(f"{self.name} has no in-flight request")
        message = self.pending
        for retry in range(2):
            response = self._wait_response()
            if response is not None:
                self.pending = None
                if response.get("ok"):
                    return response["result"]
                raise RuntimeError(
                    f"{self.name} request failed: {response.get('error_type')}: "
                    f"{response.get('message')}\n{response.get('traceback', '')}"
                )
            if retry:
                break
            self._restart_after_failure()
            self.pending = message
            self.requests.put(message, timeout=self.timeout)
        self.pending = None
        raise RuntimeError(f"{self.name} request timed out after restart")

    def _wait_response(self):
        """Stop waiting promptly after a native crash, otherwise honor timeout."""
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                return self.responses.get(timeout=min(1.0, remaining))
            except Empty:
                if self.process is None or not self.process.is_alive():
                    return None

    def request(self, message):
        self.send(message)
        return self.receive()

    def _restart_after_failure(self):
        self._terminate()
        self.restart_count += 1
        if self.restart_count > self.max_restarts:
            raise RuntimeError(
                f"{self.name} exceeded {self.max_restarts} automatic restarts"
            )
        print(
            f"[{self.name}] actor exited or hung; restart "
            f"{self.restart_count}/{self.max_restarts}",
            flush=True,
        )
        self.pending = None
        self._start()

    def recycle(self):
        if self.pending is not None:
            raise RuntimeError(f"cannot recycle busy actor {self.name}")
        self._terminate(graceful=True)
        self._start()

    def _terminate(self, graceful=False):
        process = self.process
        if process is None:
            return
        if graceful and process.is_alive():
            try:
                self.requests.put(None, timeout=1)
            except Exception:
                pass
            process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        for queue in (self.requests, self.responses):
            if queue is not None:
                queue.close()
                queue.join_thread()
        self.process = self.requests = self.responses = None

    def close(self):
        self.pending = None
        self._terminate(graceful=True)


def _chunks(values, size):
    for begin in range(0, len(values), int(size)):
        yield values[begin : begin + int(size)]


def _parallel_proposals(actors, jobs):
    partitions = [[] for _ in actors]
    for index, job in enumerate(jobs):
        partitions[index % len(actors)].append(job)
    active = []
    for actor, partition in zip(actors, partitions):
        if partition:
            actor.send({"op": "propose", "jobs": partition})
            active.append(actor)
    return [item for actor in active for item in actor.receive()]


def _factor_dual_candidates(proposals, definitions_by_id, maximum):
    """Cross compatible per-arm IK pairs before the GPU endpoint gate.

    Marvin's two kinematic branches are independent, while inter-arm
    collision is not. Crossing left start/goal pairs with right start/goal
    pairs restores the conditional search that the legacy collision-guided
    sampler performed, without loading collision libraries in IK actors.
    """
    grouped = defaultdict(list)
    singles = []
    for item in proposals:
        if definitions_by_id[item["task_id"]].mode == "dual_independent":
            grouped[item["task_id"]].append(item)
        else:
            singles.append(item)
    result = list(singles)
    for task_id in sorted(grouped):
        values = sorted(grouped[task_id], key=lambda item: item["candidate_index"])
        produced = 0
        for left in values:
            for right in values:
                q_start = np.r_[left["q_start"][:7], right["q_start"][7:]]
                q_goal = np.r_[left["q_goal"][:7], right["q_goal"][7:]]
                result.append(
                    {
                        **left,
                        "candidate_index": int(left["candidate_index"]) * len(values)
                        + int(right["candidate_index"]),
                        "q_start": q_start,
                        "q_goal": q_goal,
                        "ee_goal_pose": np.stack(
                            (left["ee_goal_pose"][0], right["ee_goal_pose"][1])
                        ).astype(np.float32, copy=False),
                    }
                )
                produced += 1
                if produced >= int(maximum):
                    break
            if produced >= int(maximum):
                break
    return result


def generate_checkpoint(config, definitions, endpoint_actors, gpu_actor, bullet_actor):
    checkpoint_started = time.perf_counter()
    base_seed = int(config["seed"])
    attempts = Counter()
    accepted = {}
    stats = Counter()
    max_attempts = int(
        config.get(
            "gpu_pipeline_max_attempts_per_task",
            config.get("max_attempts_per_task", 30),
        )
    )
    candidates_per_attempt = int(
        config.get("gpu_pipeline_endpoint_candidates_per_attempt", 8)
    )
    started = {task.task_id: time.perf_counter() for task in definitions}
    definitions_by_id = {task.task_id: task for task in definitions}

    while len(accepted) < len(definitions):
        stats["coordinator_rounds"] += 1
        pending = [task for task in definitions if task.task_id not in accepted]
        exhausted = [task.task_id for task in pending if attempts[task.task_id] >= max_attempts]
        if exhausted:
            raise RuntimeError(
                f"GPU pipeline task(s) exhausted {max_attempts} attempts: {exhausted}; "
                f"stats={dict(stats)}"
            )
        jobs = []
        for task in pending:
            attempts[task.task_id] += 1
            stats["task_attempts"] += 1
            jobs.append(
                {
                    "task": asdict(task),
                    "attempt": attempts[task.task_id],
                    "candidate_seeds": [
                        stable_seed(
                            base_seed,
                            task.task_id,
                            attempts[task.task_id],
                            f"endpoint-{candidate}",
                        )
                        for candidate in range(candidates_per_attempt)
                    ],
                }
            )
        stage_started = time.perf_counter()
        proposed = _parallel_proposals(endpoint_actors, jobs)
        stats["endpoint_proposal_wall_milliseconds"] += round(
            1000 * (time.perf_counter() - stage_started)
        )
        proposals = []
        for item in proposed:
            if item.get("proposal") is None:
                stats["endpoint_proposal_failures"] += 1
                stats[f"task/{item['task_id']}/endpoint_proposal_failures"] += 1
            else:
                proposals.append(item)
        if not proposals:
            continue
        proposals = _factor_dual_candidates(
            proposals,
            definitions_by_id,
            int(config.get("gpu_pipeline_dual_cross_candidates", 64)),
        )
        stats["endpoint_candidates_after_dual_cross"] += len(proposals)

        stage_started = time.perf_counter()
        gpu_valid = gpu_actor.request({"op": "endpoints", "items": proposals})
        stats["gpu_endpoint_audit_wall_milliseconds"] += round(
            1000 * (time.perf_counter() - stage_started)
        )
        gpu_endpoints = []
        for item, valid in zip(proposals, gpu_valid):
            if valid:
                gpu_endpoints.append(item)
            else:
                stats["endpoint_gpu_rejected"] += 1
                stats[f"task/{item['task_id']}/endpoint_gpu_rejected"] += 1
        if not gpu_endpoints:
            continue
        stage_started = time.perf_counter()
        bullet_valid = bullet_actor.request(
            {"op": "endpoints", "items": gpu_endpoints}
        )
        stats["pybullet_endpoint_audit_wall_milliseconds"] += round(
            1000 * (time.perf_counter() - stage_started)
        )
        endpoints_by_task = defaultdict(list)
        planning_candidates = int(
            config.get("gpu_pipeline_planning_candidates_per_task", 4)
        )
        for item, valid in zip(gpu_endpoints, bullet_valid):
            if valid:
                # Preserve a few endpoint alternatives. RRT/spline rejection
                # should not throw away every other endpoint that already paid
                # for Pinocchio, GPU and PyBullet validation in this round.
                if len(endpoints_by_task[item["task_id"]]) < planning_candidates:
                    endpoints_by_task[item["task_id"]].append(item)
            else:
                stats["endpoint_pybullet_rejected"] += 1
                stats[f"task/{item['task_id']}/endpoint_pybullet_rejected"] += 1
        endpoints = [
            item for values in endpoints_by_task.values() for item in values
        ]
        if not endpoints:
            continue

        by_mode = defaultdict(list)
        for item in endpoints:
            by_mode[definitions_by_id[item["task_id"]].mode].append(item)
        planned = []
        for mode in MODE_BATCH_CONFIG:
            batch_size = int(config.get(MODE_BATCH_CONFIG[mode], 1))
            for batch in _chunks(by_mode[mode], batch_size):
                query_seeds = [
                    stable_seed(
                        base_seed,
                        item["task_id"],
                        item["attempt"],
                        f"rrt-{item['candidate_index']}",
                    )
                    for item in batch
                ]
                stage_started = time.perf_counter()
                result = gpu_actor.request(
                    {
                        "op": "plan",
                        "mode": mode,
                        "items": batch,
                        "query_seeds": query_seeds,
                    }
                )
                stats["gpu_plan_audit_wall_milliseconds"] += round(
                    1000 * (time.perf_counter() - stage_started)
                )
                stats["gpu_plan_batches"] += 1
                stats["gpu_plan_queries"] += len(batch)
                for item, plan in zip(batch, result):
                    if plan["path"] is None:
                        stats[f"gpu_rejected/{plan['rejection_reason']}"] += 1
                        stats[
                            f"task/{item['task_id']}/gpu_rejected/{plan['rejection_reason']}"
                        ] += 1
                    else:
                        planned.append({**item, **plan})
        if not planned:
            continue

        stage_started = time.perf_counter()
        mesh_valid = bullet_actor.request(
            {"op": "trajectories", "items": planned}
        )
        stats["pybullet_trajectory_audit_wall_milliseconds"] += round(
            1000 * (time.perf_counter() - stage_started)
        )
        for item, valid in zip(planned, mesh_valid):
            if not valid:
                stats["trajectory_pybullet_rejected"] += 1
                stats[f"task/{item['task_id']}/trajectory_pybullet_rejected"] += 1
                continue
            task = definitions_by_id[item["task_id"]]
            if task.task_id in accepted:
                stats["valid_alternate_not_selected"] += 1
                continue
            path = np.asarray(item["path"])
            metadata = {
                "task_id": task.task_id,
                "task_mode": task.mode,
                "direction": task.direction,
                "q_start": np.asarray(item["q_start"]),
                "q_goal": np.asarray(item["q_goal"]),
                "planning_time": float(item["batch_seconds"]),
                "bspline": item["spline"],
                "ee_goal_pose": np.asarray(item["ee_goal_pose"], dtype=np.float32),
                "joint_path_length": float(
                    np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                ),
                "source_region_left": task.source["left"],
                "source_region_right": task.source["right"],
                "goal_region_left": task.goal["left"],
                "goal_region_right": task.goal["right"],
            }
            accepted[task.task_id] = (path, metadata)
            stats["accepted"] += 1
            stats[f"accepted/{task.mode}"] += 1
            stats["rrt_iterations"] += int(item["rrt_iterations"])
            stats["rrt_sampled_edges"] += int(item["rrt_sampled_edges"])
            stats["rrt_checked_states"] += int(item["rrt_checked_states"])
            stats["accepted_task_wall_milliseconds"] += int(
                1000 * (time.perf_counter() - started[task.task_id])
            )
            print(
                f"[gpu pipeline] accepted {len(accepted)}/{len(definitions)} "
                f"task={task.task_id} mode={task.mode} attempt={attempts[task.task_id]}",
                flush=True,
            )
    ordered = [accepted[task.task_id] for task in definitions]
    stats["checkpoint_wall_milliseconds"] = round(
        1000 * (time.perf_counter() - checkpoint_started)
    )
    return [item[0] for item in ordered], [item[1] for item in ordered], stats


def _publish_checkpoint(path, config, start, paths, metadata, stats):
    from scripts.generate_data.generate_marvin_warehouse_bimanual import _write_dataset
    from scripts.generate_data.launch_generate_marvin_warehouse_bimanual import (
        fsync_tree,
        quarantine_incomplete_shard,
    )

    if path.exists():
        quarantine_incomplete_shard(path)
    staging = path.parent / f".{path.name}.incomplete-{os.getpid()}"
    if staging.exists():
        quarantine = staging.parent / (
            f".{path.name}.stale-incomplete-{time.time_ns()}-{os.getpid()}"
        )
        os.replace(staging, quarantine)
    _write_dataset(staging, config, paths, metadata, int(config["seed"]) + start, stats)
    fsync_tree(staging)
    os.replace(staging, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-trajectories", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--endpoint-workers", type=int)
    parser.add_argument("--gpu-device")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    config["planner"] = "GpuMultiQueryRRTConnect"
    if args.gpu_device:
        config["gpu_device"] = args.gpu_device
    if args.endpoint_workers is not None:
        config["gpu_pipeline_endpoint_workers"] = args.endpoint_workers
    count = int(args.num_trajectories or config.get("num_trajectories", 1000))
    endpoint_workers = int(config.get("gpu_pipeline_endpoint_workers", 2))
    checkpoint_size = int(config.get("gpu_pipeline_checkpoint_size", 10))
    if count <= 0 or count % checkpoint_size or checkpoint_size % 10:
        raise ValueError("trajectory count/checkpoint size must preserve ten-task blocks")
    if endpoint_workers < 1 or endpoint_workers > 8:
        raise ValueError("endpoint worker count must lie in [1, 8]")
    if config.get("pre_rrt_filter", "none") != "none":
        raise ValueError("the GPU pipeline currently requires pre_rrt_filter=none")
    root = args.output_dir or Path(config["output_dir"])
    print(
        f"GPU pipeline: {count} trajectories, {endpoint_workers} endpoint actors, "
        f"1 GPU actor, 1 PyBullet actor, checkpoint={checkpoint_size} -> {root}",
        flush=True,
    )
    if args.dry_run:
        return 0
    if (root / "dataset_merged.hdf5").exists():
        raise FileExistsError(root / "dataset_merged.hdf5")
    root.mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)

    # Dynamic imports keep all native OMPL setup out of spawned actor modules.
    from scripts.generate_data.launch_generate_marvin_warehouse_bimanual import (
        merge_shards,
        shard_complete,
    )

    checkpoints = [(start, checkpoint_size) for start in range(0, count, checkpoint_size)]
    completed = []
    pending = []
    for start, size in checkpoints:
        path = root / "shards" / f"{start:09d}"
        if shard_complete(path, config, start, size):
            completed.append(path)
        else:
            pending.append((start, size, path))
    if not pending:
        merge_shards(root, completed, config)
        return 0

    endpoint_actors = []
    gpu_actor = None
    bullet_actor = None
    try:
        context = mp.get_context("spawn")
        timeout = float(config.get("gpu_pipeline_actor_timeout_seconds", 600))
        restarts = int(config.get("gpu_pipeline_max_actor_restarts", 3))
        for index in range(endpoint_workers):
            endpoint_actors.append(
                RestartableActor(
                    context, "endpoint", config, f"endpoint-{index}", timeout, restarts
                )
            )
        gpu_actor = RestartableActor(context, "gpu", config, "gpu", timeout, restarts)
        bullet_actor = RestartableActor(
            context, "pybullet", config, "pybullet", timeout, restarts
        )
        contract = TaskContract(config, int(config["seed"]))
        recycle = int(config.get("gpu_pipeline_pybullet_recycle_trajectories", 50))
        generated_since_recycle = 0
        for start, size, path in pending:
            definitions = [contract.build(task_id) for task_id in range(start, start + size)]
            paths, metadata, stats = generate_checkpoint(
                config, definitions, endpoint_actors, gpu_actor, bullet_actor
            )
            stats["endpoint_actor_restarts"] = sum(
                actor.restart_count for actor in endpoint_actors
            )
            stats["gpu_actor_restarts"] = gpu_actor.restart_count
            stats["pybullet_actor_restarts"] = bullet_actor.restart_count
            _publish_checkpoint(path, config, start, paths, metadata, stats)
            completed.append(path)
            generated_since_recycle += size
            if recycle > 0 and generated_since_recycle >= recycle:
                bullet_actor.recycle()
                generated_since_recycle = 0
    finally:
        for actor in endpoint_actors:
            actor.close()
        if gpu_actor is not None:
            gpu_actor.close()
        if bullet_actor is not None:
            bullet_actor.close()
    merge_shards(root, completed, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
