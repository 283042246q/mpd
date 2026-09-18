#!/usr/bin/env python3
"""Run one isolated Marvin pipeline stage under a sustained, real workload.

The stage imports are deliberately local.  Endpoint and PyBullet soaks hide
CUDA completely, the GPU soak reads fixed endpoints before constructing its
single CUDA backend, and no mode constructs either of the other two stages.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import sys
import time

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

import h5py
import numpy as np
import yaml


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY
    / "data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-independent.yaml"
)
DEFAULT_DATASET = (
    REPOSITORY
    / "data_public/data_trajectories/"
    "EnvWarehouse-RobotMarvinBimanual-independent-v3-res002-nosimplifier-150k-3/"
    "dataset_merged.hdf5"
)
MODES = ("dual_independent", "left_only", "right_only")
MODE_BATCH_KEYS = {
    "dual_independent": "gpu_query_batch_size_dual",
    "left_only": "gpu_query_batch_size_left",
    "right_only": "gpu_query_batch_size_right",
}
MINIMUM_SOAK_SECONDS = 40 * 60


class DurableEventLog:
    """Append diagnostic events durably so a hard reset leaves a last marker."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, event, **fields):
        payload = {
            "event": event,
            "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "unix_seconds": time.time(),
            "monotonic_seconds": time.monotonic(),
            **fields,
        }
        self.stream.write(json.dumps(payload, sort_keys=True) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def _decode_strings(dataset):
    return np.asarray(dataset.asstr()[:], dtype=object)


def load_fixed_inputs(dataset_path, config, rows_per_mode=None, trajectory_rows=16):
    """Copy all soak inputs out of HDF5 before any native worker is created."""
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if rows_per_mode is None:
        rows_per_mode = {
            mode: int(config.get(MODE_BATCH_KEYS[mode], 1)) for mode in MODES
        }
    required = {
        "q_start",
        "q_goal",
        "task_mode",
        "sol_path",
        "bspline_params_tt",
        "bspline_params_cc",
        "bspline_params_k",
    }
    with h5py.File(dataset_path, "r") as handle:
        missing = required - set(handle)
        if missing:
            raise ValueError(f"dataset is missing required fields: {sorted(missing)}")
        task_modes = _decode_strings(handle["task_mode"])
        endpoints = {}
        endpoint_rows = {}
        for mode in MODES:
            selected = np.flatnonzero(task_modes == mode)[: int(rows_per_mode[mode])]
            if len(selected) != int(rows_per_mode[mode]):
                raise ValueError(
                    f"dataset has only {len(selected)} {mode} rows; "
                    f"need {rows_per_mode[mode]}"
                )
            endpoint_rows[mode] = selected.astype(np.int64)
            endpoints[mode] = (
                np.asarray(handle["q_start"][selected], dtype=np.float64),
                np.asarray(handle["q_goal"][selected], dtype=np.float64),
            )
        trajectory_rows = min(int(trajectory_rows), len(task_modes))
        selected = np.arange(trajectory_rows, dtype=np.int64)
        trajectories = [
            {
                "row": int(row),
                "path": np.asarray(handle["sol_path"][row], dtype=np.float64),
                "spline": (
                    np.asarray(handle["bspline_params_tt"][row], dtype=np.float64),
                    np.asarray(handle["bspline_params_cc"][row], dtype=np.float64),
                    int(handle["bspline_params_k"][row]),
                ),
            }
            for row in selected
        ]
    return endpoints, endpoint_rows, trajectories


def save_fixed_endpoint_snapshot(path, source, endpoints, rows):
    values = {"source_dataset": np.asarray(str(Path(source).resolve()))}
    for mode, (q_start, q_goal) in endpoints.items():
        values[f"{mode}_rows"] = rows[mode]
        values[f"{mode}_q_start"] = q_start
        values[f"{mode}_q_goal"] = q_goal
    np.savez_compressed(path, **values)


def _proc_status(pid):
    result = {"pid": int(pid)}
    try:
        for line in Path(f"/proc/{int(pid)}/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"Name", "State", "VmRSS", "VmHWM", "Threads"}:
                result[key] = value.strip()
    except (FileNotFoundError, ProcessLookupError):
        result["missing"] = True
    return result


def _base_heartbeat(started, counters, pids):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "counters": dict(counters),
        "load_average": [round(value, 3) for value in os.getloadavg()],
        "self_peak_rss_kib": int(usage.ru_maxrss),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "processes": [_proc_status(pid) for pid in pids],
    }


def _validate_isolation(mode, excluded_cpu):
    affinity = os.sched_getaffinity(0)
    if excluded_cpu is not None and int(excluded_cpu) in affinity:
        raise RuntimeError(
            f"CPU {excluded_cpu} is still in affinity {sorted(affinity)}; "
            "start this command through the taskset wrapper"
        )
    if mode in {"endpoint", "pybullet"} and os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ) not in {"", "-1"}:
        raise RuntimeError(
            f"{mode} soak requires CUDA_VISIBLE_DEVICES='' or '-1'"
        )


def run_endpoint(config, duration_seconds, log, heartbeat_seconds):
    """Exercise the production Pinocchio/IK actor pool and nothing downstream."""
    import multiprocessing as mp

    from scripts.generate_data.launch_generate_marvin_warehouse_gpu_pipeline import (
        RestartableActor,
    )
    from scripts.generate_data.marvin_gpu_task_contract import (
        TaskContract,
        stable_seed,
    )

    count = int(config.get("gpu_pipeline_endpoint_workers", 2))
    timeout = float(config.get("gpu_pipeline_actor_timeout_seconds", 600))
    recycle_jobs = int(config.get("gpu_pipeline_endpoint_recycle_jobs", 100))
    context = mp.get_context("spawn")
    actors = []
    busy = set()
    actor_jobs = Counter()
    counters = Counter()
    next_task_id = 0
    contract = TaskContract(config, int(config["seed"]))
    candidate_chunk = int(
        config.get("gpu_pipeline_endpoint_candidate_chunk_size", 2)
    )
    if candidate_chunk < 1:
        raise ValueError("endpoint candidate chunk size must be positive")
    try:
        for index in range(count):
            actors.append(
                RestartableActor(
                    context,
                    "endpoint",
                    config,
                    f"soak-endpoint-{index}",
                    timeout,
                    0,
                    0,
                )
            )
        started = time.monotonic()
        next_heartbeat = started
        log.write(
            "stage_ready",
            stage="endpoint",
            actors=[_proc_status(actor.process.pid) for actor in actors],
            recycle_jobs=recycle_jobs,
            cpu_sphere_filter=bool(
                config.get("gpu_pipeline_endpoint_cpu_sphere_filter", True)
            ),
        )
        deadline = started + float(duration_seconds)
        while time.monotonic() < deadline:
            for index, actor in enumerate(actors):
                if index in busy:
                    continue
                task = contract.build(next_task_id)
                candidate_indices = list(range(candidate_chunk))
                actor.send(
                    {
                        "op": "propose",
                        "jobs": [
                            {
                                "task": asdict(task),
                                "attempt": 1,
                                "candidate_indices": candidate_indices,
                                "candidate_seeds": [
                                    stable_seed(
                                        int(config["seed"]),
                                        next_task_id,
                                        1,
                                        f"endpoint-{candidate}",
                                    )
                                    for candidate in candidate_indices
                                ],
                            }
                        ],
                    }
                )
                busy.add(index)
                actor_jobs[index] += 1
                counters["jobs_submitted"] += 1
                next_task_id += 1
            progressed = False
            for index in tuple(busy):
                ready, result = actors[index].poll()
                if not ready:
                    continue
                busy.remove(index)
                progressed = True
                counters["jobs_completed"] += 1
                counters["proposals"] += sum(
                    item.get("proposal") is not None for item in result
                )
                counters["proposal_failures"] += sum(
                    item.get("proposal") is None for item in result
                )
                if recycle_jobs > 0 and actor_jobs[index] >= recycle_jobs:
                    actors[index].recycle()
                    actor_jobs[index] = 0
                    counters["proactive_recycles"] += 1
            now = time.monotonic()
            if now >= next_heartbeat:
                log.write(
                    "heartbeat",
                    stage="endpoint",
                    **_base_heartbeat(
                        started,
                        counters,
                        [actor.process.pid for actor in actors],
                    ),
                )
                next_heartbeat = now + heartbeat_seconds
            if not progressed:
                time.sleep(0.002)
    finally:
        for actor in actors:
            actor.close()
    return counters


def run_gpu(config, endpoints, duration_seconds, log, heartbeat_seconds):
    """Exercise only the CUDA RRT/shortcut/Torch-audit backend."""
    from scripts.generate_data.marvin_gpu_planning_backend import (
        MarvinGpuPlanningBackend,
    )

    import torch

    worker = MarvinGpuPlanningBackend(config)
    device = worker.robot.q_pos_min.device
    started = time.monotonic()
    next_heartbeat = started
    counters = Counter()
    batch_index = 0
    log.write(
        "stage_ready",
        stage="gpu",
        pid=os.getpid(),
        gpu_name=torch.cuda.get_device_name(device),
        gpu_total_memory_bytes=int(
            torch.cuda.get_device_properties(device).total_memory
        ),
        batch_sizes={mode: len(endpoints[mode][0]) for mode in MODES},
    )
    deadline = started + float(duration_seconds)
    while time.monotonic() < deadline:
        mode = MODES[batch_index % len(MODES)]
        q_starts, q_goals = endpoints[mode]
        seeds = [
            int(config["seed"]) + batch_index * 100_003 + query
            for query in range(len(q_starts))
        ]
        log.write(
            "gpu_batch_started",
            stage="gpu",
            batch_index=batch_index,
            mode=mode,
            queries=len(q_starts),
        )
        plans = worker.plan_mode_batch(mode, q_starts, q_goals, seeds)
        counters["batches"] += 1
        counters["queries"] += len(plans)
        counters["accepted"] += sum(plan.path is not None for plan in plans)
        for plan in plans:
            counters[f"result/{plan.rejection_reason or 'accepted'}"] += 1
        log.write(
            "gpu_batch_finished",
            stage="gpu",
            batch_index=batch_index,
            mode=mode,
            accepted=sum(plan.path is not None for plan in plans),
            gpu_allocated_bytes=int(torch.cuda.memory_allocated(device)),
            gpu_reserved_bytes=int(torch.cuda.memory_reserved(device)),
            gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            gpu_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )
        batch_index += 1
        now = time.monotonic()
        if now >= next_heartbeat:
            log.write(
                "heartbeat",
                stage="gpu",
                **_base_heartbeat(started, counters, [os.getpid()]),
                gpu_allocated_bytes=int(torch.cuda.memory_allocated(device)),
                gpu_reserved_bytes=int(torch.cuda.memory_reserved(device)),
                gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                gpu_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            )
            next_heartbeat = now + heartbeat_seconds
    torch.cuda.synchronize(device)
    return counters


def run_pybullet(config, trajectories, duration_seconds, log, heartbeat_seconds):
    """Exercise one persistent DIRECT auditor against published trajectories."""
    from scripts.generate_data.marvin_pybullet_auditor import MarvinPyBulletAuditor

    worker = MarvinPyBulletAuditor(config)
    started = time.monotonic()
    next_heartbeat = started
    counters = Counter()
    audit_index = 0
    log.write("stage_ready", stage="pybullet", pid=os.getpid())
    deadline = started + float(duration_seconds)
    try:
        while time.monotonic() < deadline:
            item = trajectories[audit_index % len(trajectories)]
            valid = worker.trajectory_valid(item["path"], item["spline"])
            counters["trajectories"] += 1
            counters["valid"] += bool(valid)
            counters["invalid"] += not bool(valid)
            audit_index += 1
            now = time.monotonic()
            if now >= next_heartbeat:
                log.write(
                    "heartbeat",
                    stage="pybullet",
                    current_dataset_row=item["row"],
                    **_base_heartbeat(started, counters, [os.getpid()]),
                )
                next_heartbeat = now + heartbeat_seconds
    finally:
        worker.close()
    return counters


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("endpoint", "gpu", "pybullet"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=MINIMUM_SOAK_SECONDS)
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("--excluded-cpu", type=int, default=2)
    parser.add_argument("--gpu-device")
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Permit less than 40 minutes for smoke testing only.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.duration_seconds <= 0:
        raise ValueError("duration must be positive")
    if args.duration_seconds < MINIMUM_SOAK_SECONDS and not args.allow_short:
        raise ValueError(
            f"isolation soaks must run at least {MINIMUM_SOAK_SECONDS} seconds; "
            "use --allow-short only for a smoke test"
        )
    if args.heartbeat_seconds <= 0:
        raise ValueError("heartbeat interval must be positive")
    _validate_isolation(args.mode, args.excluded_cpu)
    config = yaml.safe_load(args.config.read_text())
    if args.gpu_device:
        config["gpu_device"] = args.gpu_device
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir = args.output_dir / ".matplotlib"
    matplotlib_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    log = DurableEventLog(args.output_dir / "events.jsonl")
    started = time.monotonic()
    log.write(
        "soak_started",
        stage=args.mode,
        pid=os.getpid(),
        duration_seconds=args.duration_seconds,
        config=str(args.config.resolve()),
        dataset=str(args.dataset.resolve()),
        python=sys.executable,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    try:
        if args.mode == "endpoint":
            counters = run_endpoint(
                config, args.duration_seconds, log, args.heartbeat_seconds
            )
        else:
            endpoints, rows, trajectories = load_fixed_inputs(
                args.dataset, config
            )
            save_fixed_endpoint_snapshot(
                args.output_dir / "fixed_endpoints.npz",
                args.dataset,
                endpoints,
                rows,
            )
            if args.mode == "gpu":
                counters = run_gpu(
                    config,
                    endpoints,
                    args.duration_seconds,
                    log,
                    args.heartbeat_seconds,
                )
            else:
                counters = run_pybullet(
                    config,
                    trajectories,
                    args.duration_seconds,
                    log,
                    args.heartbeat_seconds,
                )
        elapsed = time.monotonic() - started
        log.write(
            "soak_completed",
            stage=args.mode,
            elapsed_seconds=round(elapsed, 3),
            counters=dict(counters),
        )
        (args.output_dir / "summary.yaml").write_text(
            yaml.safe_dump(
                {
                    "stage": args.mode,
                    "completed": True,
                    "elapsed_seconds": elapsed,
                    "counters": dict(counters),
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                },
                sort_keys=False,
            )
        )
        return 0
    except BaseException as error:
        log.write(
            "soak_failed",
            stage=args.mode,
            error_type=type(error).__name__,
            message=str(error),
        )
        raise
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
