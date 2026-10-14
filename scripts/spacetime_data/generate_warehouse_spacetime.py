#!/usr/bin/env python3
"""Generate validated single-arm Warehouse spacetime trajectories in batches.

Run as a module from the MPD root. The two backends are deliberately named:
ompl-strrtstar is genuine ST-RRT* in isolated CPU processes; torch-forest is
the native GPU multi-query spacetime teacher with arrival-root recycling.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

from mpd.datasets.spacetime_schema import load_panda_robot_bundle, materialize_dataset_contract, sha256_file
from scripts.spacetime_data.batch_spacetime_rrt import TorchSpaceTimeForest, edge_collision_mask
from scripts.spacetime_data.warehouse_world import (
    MOTION_MODES, WORLD_SAMPLING_POLICY, WarehouseCollision, make_forecast,
)
from scripts.spacetime_data.warehouse_trajectory_fit import fit_teacher


REPOSITORY = Path(__file__).resolve().parents[2]
SCHEMA = "warehouse_spacetime_joint_v1"
SOURCE = REPOSITORY / "data_trajectories/EnvWarehouse-RobotPanda-config_file_v01-joint_joint-one-RRTConnect/dataset_merged_doubled.hdf5"


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".inprogress")
    with temporary.open("w") as stream:
        json.dump(json_value(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--backend", choices=("torch-forest", "ompl-strrtstar"), default="torch-forest")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--torch-batch-policy", choices=("fill", "legacy"), default="fill",
                        help="Torch only: fill valid-query batches, or limit candidate attempts per batch")
    parser.add_argument("--checkpoint-batches", type=int, default=10,
                        help="publish derived index/report every N batches; shards commit every batch")
    parser.add_argument("--edges-per-query", type=int, default=16)
    parser.add_argument("--goal-roots", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=512)
    parser.add_argument("--solve-seconds", type=float, default=5.)
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--ompl-scheduling", choices=("streaming", "batch"), default="streaming",
                        help="OMPL only: continuous valid-query queue, or legacy batch barrier")
    parser.add_argument("--ompl-queue-size", type=int, default=0,
                        help="OMPL streaming in-flight cap; 0 means max(query-batch-size, 2 * cpu-workers)")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--source-group-size", type=int, default=2,
                        help="forward/reverse source rows per family; doubled warehouse default=2")
    parser.add_argument("--max-attempts", type=int, default=0, help="0 means 20 * requested trajectories")
    parser.add_argument("--max-world-attempts", type=int, default=12,
                        help="maximum dynamic-world samples per attempt; does not retry failed planning")
    parser.add_argument("--obstacles", type=int, default=3,
                        help="dynamic object count; 0 retains only static Warehouse and self collision")
    parser.add_argument("--motion-modes", nargs="+", choices=MOTION_MODES, default=list(MOTION_MODES),
                        help="mode pool: each dynamic object samples independently with replacement")
    parser.add_argument("--horizon", type=float, default=14.)
    parser.add_argument("--nominal-duration", type=float, default=8.)
    parser.add_argument("--forecast-dt", type=float, default=0.02)
    parser.add_argument("--edge-dt", type=float, default=0.02)
    parser.add_argument("--joint-step", type=float, default=0.04)
    parser.add_argument("--velocity-scale", type=float, default=0.4)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--collision-chunk", type=int, default=512)
    parser.add_argument("--spatial-rmse-max", type=float, default=0.08)
    parser.add_argument("--timing-rmse-max", type=float, default=0.08)
    parser.add_argument("--no-root-recycling", action="store_true")
    parser.add_argument("--require-tau-r", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for name in ("num_trajectories", "query_batch_size", "checkpoint_batches", "edges_per_query", "goal_roots",
                 "max_iterations", "cpu_workers", "source_group_size", "collision_chunk",
                 "max_world_attempts"):
        if getattr(args, name) < 1:
            parser.error(name + " must be positive")
    if args.start_index < 0 or args.max_attempts < 0 or args.seed < 0:
        parser.error("indices, seed and max-attempts must be non-negative")
    if args.obstacles < 0:
        parser.error("obstacles must be non-negative")
    if args.ompl_queue_size < 0:
        parser.error("ompl-queue-size must be non-negative")
    if not 2 < args.nominal_duration <= args.horizon or not 0 < args.velocity_scale <= 1:
        parser.error("require 2 < nominal-duration <= horizon and 0 < velocity-scale <= 1")
    if min(args.solve_seconds, args.forecast_dt, args.edge_dt, args.joint_step,
           args.spatial_rmse_max, args.timing_rmse_max) <= 0 or args.margin < 0:
        parser.error("invalid budget, resolution, fit threshold or margin")
    args.source, args.output_root = args.source.resolve(), args.output_root.resolve()
    return args


def run_identity(args, robot):
    config = vars(args).copy()
    for key in ("output_root", "resume", "num_trajectories", "max_attempts"):
        config.pop(key)
    stat = args.source.stat()
    config.update(schema=SCHEMA, source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                  robot_hashes=robot.hashes,
                  runtime={"python": sys.version.split()[0], "torch": str(torch.__version__),
                           "numpy": np.__version__, "cuda": torch.version.cuda},
                  reused_tree_sha256=sha256_file(REPOSITORY / "scripts/generate_data/gpu_batch_rrt.py"),
                  static_environment="EnvWarehouse", goal_type="joint", oracle_forecast=True,
                  world_sampling=WORLD_SAMPLING_POLICY,
                  terminal_hold="none; task ends at decoded trajectory duration",
                  software={name: sha256_file(REPOSITORY / "scripts/spacetime_data" / name) for name in
                            ("warehouse_world.py", "batch_spacetime_rrt.py", "ompl_strrt_batch.py", "streaming_spacetime.py", "warehouse_trajectory_fit.py",
                             "generate_warehouse_spacetime.py")},
                  static_geometry_hashes={name: sha256_file(REPOSITORY / "mpd/torch_robotics/torch_robotics/environments" / name)
                                          for name in ("env_warehouse.py", "env_table_shelf.py", "primitives.py")})
    return json_value(config)


def read_progress(output, config_hash):
    accepted, attempts, failures, entries = 0, 0, Counter(), []
    expected_attempt = 0
    for path in sorted((output / "shards").glob("*.hdf5")):
        with h5py.File(path, "r") as shard:
            if shard.attrs["config_hash"] != config_hash or shard.attrs["schema"] != SCHEMA:
                raise ValueError("shard contract mismatch: " + str(path))
            if int(shard.attrs["attempt_start"]) != expected_attempt:
                raise ValueError("non-contiguous committed attempt ranges")
            expected_attempt = attempts = int(shard.attrs["attempt_stop"])
            failures.update(json.loads(shard.attrs["failures_json"]))
            for key, sample in shard["samples"].items():
                entries.append({"file": str(path.relative_to(output)), "group": "samples/" + key,
                                "family_id": int(sample.attrs["family_id"]), "attempt_id": int(sample.attrs["attempt_id"]),
                                "duration_s": json.loads(sample.attrs["quality_json"])["duration"],
                                "modes": json.loads(sample.attrs["world_metadata_json"])["modes"],
                                "tau_r_valid": bool(sample.attrs["tau_r_valid"])})
                accepted += 1
    return accepted, attempts, failures, entries


def write_shard(output, shard_index, attempt_start, attempt_stop, records, failures, config_hash):
    destination = output / "shards" / ("shard-%06d.hdf5" % shard_index)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(".hdf5.inprogress")
    with h5py.File(temporary, "w") as shard:
        shard.attrs.update(schema=SCHEMA, config_hash=config_hash, attempt_start=attempt_start,
                           attempt_stop=attempt_stop, failures_json=json.dumps(dict(failures)))
        samples = shard.create_group("samples")
        for record in records:
            job, result, fitted = record
            sample = samples.create_group("%09d" % job["attempt_id"])
            sample.attrs.update(attempt_id=job["attempt_id"], source_row=job["source_row"], family_id=job["family_id"],
                                source_task_id=job["source_task_id"], seed=job["seed"], backend=result.statistics["backend"],
                                world_metadata_json=json.dumps(job["forecast"].metadata()),
                                quality_json=json.dumps(json_value(fitted["quality"])),
                                planner_statistics_json=json.dumps(json_value(result.statistics)),
                                tau_r_valid=fitted["tau_r_valid"], tau_r_reject_reason=str(fitted["tau_r_reject_reason"]),
                                tau_r_quality_json=json.dumps(json_value(fitted["tau_r_quality"])),
                                degree=fitted["degree"], phase_eta=fitted["phase_eta"],
                                accepted=True, fit_wall_s=job["fit_wall_s"])
            data = {"condition/q_start": job["q_start"], "condition/q_goal": job["q_goal"],
                    "world/times": job["forecast"].times, "world/positions": job["forecast"].positions,
                    "world/shape": job["forecast"].shape, "world/size": job["forecast"].size,
                    "world/event_times": job["forecast"].event_times, "world/anchors": job["forecast"].anchors,
                    "spatial/control_points": fitted["P"], "spatial/knots": fitted["knots"],
                    "timing/control_points": fitted["c"], "timing/tau_r": fitted["tau_r"],
                    "timing/T_min": np.asarray(fitted["T_min"]),
                    "timing/reference_phase": fitted["reference_phase"], "timing/reference_time": fitted["reference_time"],
                    "execution/q": fitted["execution_q"], "execution/dq": fitted["execution_dq"],
                    "execution/ddq": fitted["execution_ddq"], "execution/time": fitted["execution_time"]}
            for index, raw in enumerate(result.paths):
                data["raw/solution_%02d" % index] = raw
            data["raw/fitted_solution_index"] = np.asarray(job["fitted_solution_index"])
            for key, array in data.items():
                value = np.asarray(array)
                sample.create_dataset(key, data=value, compression="lzf" if value.ndim else None)
        shard.flush()
    os.replace(temporary, destination)
    return destination


def publish_index(output, config_hash, accepted, attempts, failures, entries, elapsed, requested,
                  pipeline_statistics=None):
    splits = {"train": [], "validation": [], "test": []}
    durations = [entry["duration_s"] for entry in entries]
    mode_counts = Counter(mode for entry in entries for mode in entry["modes"])
    for entry in entries:
        bucket = int(hashlib.sha256(str(entry["family_id"]).encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 80 else ("validation" if bucket < 90 else "test")
        splits[split].append(entry)
    atomic_json(output / "index.json", {"schema": SCHEMA, "config_hash": config_hash, "splits": splits})
    report = {"schema": SCHEMA, "config_hash": config_hash,
                "accepted": accepted, "attempts": attempts, "requested": requested,
                "complete": accepted >= requested, "failure_counts": dict(failures),
                "session_wall_s": elapsed, "split_counts": {k: len(v) for k, v in splits.items()},
                "object_motion_mode_counts": dict(mode_counts),
                "tau_r_valid_samples": sum(entry["tau_r_valid"] for entry in entries),
                "duration_s": {"min": min(durations), "mean": float(np.mean(durations)), "max": max(durations)} if durations else None,
                "positive_definition": "fitted c codec; joint limits, v/a, static/self/dynamic and interaction checks passed",
                "safety_scope": "sampled sphere/primitive collision with margin; not continuous collision certification"}
    if pipeline_statistics is not None:
        report["session_pipeline_statistics"] = pipeline_statistics
    atomic_json(output / "report.json", report)


def prepare_job(args, source, collision, current):
    """Prepare one candidate; a rejected candidate consumes an attempt, not a queue slot."""
    paths = source["sol_path"]
    row = (args.start_index + current * args.source_group_size) % len(paths)
    reference = np.asarray(paths[row], dtype=float)
    if collision.static_collision(reference).any().item():
        return None, "source_static_self_or_limits"
    seed = int(np.random.SeedSequence([args.seed, current]).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    centers = None
    if args.obstacles:
        anchor_indices = np.linspace(0, len(reference) - 1, min(64, len(reference))).astype(int)
        centers = collision.centers(reference[anchor_indices]).cpu().numpy()
    forecast = None
    for _ in range(args.max_world_attempts):
        try:
            candidate = make_forecast(centers, args.nominal_duration, args.horizon, rng,
                                      object_count=args.obstacles, modes=args.motion_modes, forecast_dt=args.forecast_dt)
        except ValueError:
            continue
        collision.set_worlds([candidate])
        if not collision(reference[:1], [0.], [0]).item():
            forecast = candidate
            break
    if forecast is None:
        return None, "world_anchor_or_initial_collision"
    return {"attempt_id": current, "source_row": row, "source_task_id": int(source["task_id"][row]),
            "family_id": row // args.source_group_size, "seed": seed,
            "q_start": reference[0], "q_goal": reference[-1],
            "reference": reference, "forecast": forecast}, None


def validate_result(args, collision, query, job, result):
    """Same raw and fitted acceptance gates for both scheduling modes/backends."""
    fitted, reason = None, "teacher_no_solution"
    fit_started = time.perf_counter()
    fine_paths = []
    for raw in result.paths:
        delta_t = np.diff(raw[:, -1])
        velocity = collision.robot.dq_max.cpu().numpy() * args.velocity_scale
        if (not np.isfinite(raw).all() or np.any(delta_t <= 0)
            or abs(raw[0, -1]) > 1e-6
            or not np.allclose(raw[0, :7], job["q_start"], atol=1e-6)
            or not np.allclose(raw[-1, :7], job["q_goal"], atol=1e-6)
            or np.any(np.abs(np.diff(raw[:, :7], axis=0)) > delta_t[:, None] * velocity + 1e-5)):
            reason = "raw_time_boundary_or_velocity"
            continue
        tensor = torch.as_tensor(raw, **collision.tensor_args)
        enabled = torch.ones((1, len(raw) - 1), dtype=torch.bool, device=tensor.device)
        invalid, _ = edge_collision_mask(
            tensor[:-1][None], tensor[1:][None],
            torch.full_like(enabled, query, dtype=torch.long), collision,
            joint_step=args.joint_step / 2, time_step=args.edge_dt / 2)
        if bool(invalid.any().item()):
            reason = "raw_dense_collision"
            continue
        fine_paths.append(raw)
    result.statistics["raw_dense_rejected"] = len(result.paths) - len(fine_paths)
    result.paths = fine_paths
    for raw_index, raw in enumerate(result.paths):
        fitted, reason = fit_teacher(raw, collision, query, job["forecast"],
                                     spatial_rmse_max=args.spatial_rmse_max,
                                     timing_rmse_max=args.timing_rmse_max,
                                     require_tau_r=args.require_tau_r)
        if fitted is not None:
            job["fitted_solution_index"] = raw_index
            break
    job["fit_wall_s"] = time.perf_counter() - fit_started
    return (None, str(reason)) if fitted is None else ((job, result, fitted), None)


def generate_ompl_stream(args, source, collision, backend, config_hash,
                         accepted, attempt, failures, entries, started):
    from scripts.spacetime_data.streaming_spacetime import StreamingSpaceTime
    import math

    def validate(job, result):
        # The preparation step also changes the parent world. Restore the exact
        # forecast belonging to this result before any collision/fit checks.
        collision.set_worlds([job["forecast"]])
        return validate_result(args, collision, 0, job, result)

    capacity = args.ompl_queue_size or max(args.query_batch_size, 2 * args.cpu_workers)
    stream = StreamingSpaceTime(
        lambda current: prepare_job(args, source, collision, current), backend.submit, validate,
        start_attempt=attempt, max_attempts=args.max_attempts or args.num_trajectories * 20,
        target=args.num_trajectories - accepted, capacity=capacity,
        timeout=60 + math.ceil(capacity / args.cpu_workers) * args.solve_seconds)
    stats = stream.statistics
    stats.update(scheduling="streaming", worker_count=args.cpu_workers, shard_write_wall_s=0.)
    output = args.output_root
    shard_index = len(list((output / "shards").glob("*.hdf5")))
    session_shards, begin = 0, attempt
    records, shard_failures = [], Counter()

    def publish():
        elapsed = time.perf_counter() - started
        # Sum of all completed queries, including rejected ones; this is an
        # estimate including callback/setup time, not OS-measured CPU occupancy.
        stats["planner_worker_utilization_estimate"] = stats["planner_wall_s_total"] / max(
            elapsed * args.cpu_workers, 1e-9)
        stats["planner_mean_wall_s"] = stats["planner_wall_s_total"] / max(stats["completed"], 1)
        stats["accepted_trajectories_per_minute"] = stats["accepted"] * 60 / max(elapsed, 1e-9)
        publish_index(output, config_hash, accepted, attempt, failures, entries,
                      elapsed, args.num_trajectories, stats)

    def commit():
        nonlocal accepted, shard_index, session_shards, begin, records, shard_failures
        write_started = time.perf_counter()
        destination = write_shard(output, shard_index, begin, attempt, records, shard_failures, config_hash)
        stats["shard_write_wall_s"] += time.perf_counter() - write_started
        accepted += len(records)
        failures.update(shard_failures)
        entries.extend({"file": str(destination.relative_to(output)), "group": "samples/%09d" % job["attempt_id"],
                        "family_id": job["family_id"], "attempt_id": job["attempt_id"],
                        "duration_s": fitted["quality"]["duration"], "modes": job["forecast"].modes,
                        "tau_r_valid": fitted["tau_r_valid"]} for job, _, fitted in records)
        shard_index += 1
        session_shards += 1
        if session_shards % args.checkpoint_batches == 0:
            publish()
        print(json.dumps({"accepted": accepted, "attempts": attempt, "batch_accepted": len(records),
                          "batch_failures": dict(shard_failures), "wall_s": round(time.perf_counter() - started, 2),
                          "submitted": stats["submitted"], "completed": stats["completed"]}), flush=True)
        begin, records, shard_failures = attempt, [], Counter()

    for outcome in stream:
        if outcome.attempt_id != attempt:
            raise RuntimeError("non-contiguous streaming outcome")
        attempt += 1
        if outcome.record is None:
            shard_failures[outcome.reason] += 1
        else:
            records.append(outcome.record)
        if attempt - begin >= args.query_batch_size:
            commit()
    if attempt > begin:
        commit()
    publish()
    return 0 if accepted >= args.num_trajectories else 2


def _generate(args):
    torch.set_num_threads(1)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu explicitly for a small check")
    robot_bundle = load_panda_robot_bundle(REPOSITORY)
    config = run_identity(args, robot_bundle)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = args.output_root
    contract = output / "config.json"
    if contract.exists():
        if not args.resume:
            raise FileExistsError("output already initialized; use --resume")
        if json.loads(contract.read_text()) != config:
            raise ValueError("resume configuration/assets/software/source do not match")
    else:
        if output.exists() and any(path.name != ".writer.lock" for path in output.iterdir()):
            raise ValueError("refusing to initialize a nonempty output without a contract")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(contract, config)
    materialize_dataset_contract(output, robot_bundle, {
        "schema": SCHEMA, "config_hash": config_hash, "robot": robot_bundle.manifest_robot_section(),
        "static_environment": "EnvWarehouse", "world": "frozen oracle piecewise-linear forecast",
        "obstacle_static_collision": "ignored", "goal_type": "joint", "codec": "P29-degree5/c8; optional tau_r6",
        "codec_dense_phase_points": 2049, "collision_margin_m": args.margin,
        "backend": args.backend, "normalizers": "not fitted; fit on train only"})
    (output / "shards").mkdir(exist_ok=True)
    accepted, attempt, failures, entries = read_progress(output, config_hash)
    started, backend = time.perf_counter(), None
    if accepted >= args.num_trajectories:
        # A shard may have committed immediately before an interrupted index
        # update. Rebuild the derived index even if no more planning is needed.
        previous = json.loads((output / "report.json").read_text()) if (output / "report.json").exists() else {}
        unchanged = previous.get("accepted") == accepted and previous.get("attempts") == attempt
        publish_index(output, config_hash, accepted, attempt, failures, entries,
                      previous.get("session_wall_s", 0.) if unchanged else 0., args.num_trajectories,
                      previous.get("session_pipeline_statistics") if unchanged else None)
        print("Already complete: %d trajectories" % accepted, flush=True)
        return 0
    collision = WarehouseCollision(args.device, margin=args.margin, chunk_size=args.collision_chunk)
    if args.backend == "ompl-strrtstar":
        from scripts.spacetime_data.ompl_strrt_batch import OmplSTRRTBatch
        backend = OmplSTRRTBatch(REPOSITORY, workers=args.cpu_workers, seed=args.seed,
                                 margin=args.margin, collision_chunk=args.collision_chunk,
                                 velocity_scale=args.velocity_scale, time_step=args.edge_dt, joint_step=args.joint_step,
                                 solve_seconds=args.solve_seconds, planner_range=1.5)
    max_attempts = args.max_attempts or args.num_trajectories * 20
    shard_index = len(list((output / "shards").glob("*.hdf5")))
    session_batches = 0
    pipeline = None
    if backend is None:
        pipeline = dict(scheduling="torch-batch", batch_policy=args.torch_batch_policy,
                        query_batch_size=args.query_batch_size, prepared=0, submitted=0,
                        completed=0, raw_solved_queries=0, accepted=0, planned_batches=0,
                        full_batches=0, batch_size_histogram={}, preparation_wall_s=0.,
                        world_setup_wall_s=0., planner_wall_s=0., postprocess_wall_s=0.,
                        shard_write_wall_s=0.)

    def sync_stage():
        if pipeline is not None and collision.tensor_args["device"].type == "cuda":
            torch.cuda.synchronize(collision.tensor_args["device"])

    def publish():
        elapsed = time.perf_counter() - started
        if pipeline is not None:
            pipeline["effective_batch_size_mean"] = pipeline["submitted"] / max(pipeline["planned_batches"], 1)
            pipeline["batch_fill_fraction"] = pipeline["submitted"] / max(
                pipeline["planned_batches"] * args.query_batch_size, 1)
            pipeline["preparation_fraction"] = pipeline["preparation_wall_s"] / max(elapsed, 1e-9)
            pipeline["accepted_trajectories_per_minute"] = pipeline["accepted"] * 60 / max(elapsed, 1e-9)
        publish_index(output, config_hash, accepted, attempt, failures, entries,
                      elapsed, args.num_trajectories, pipeline)

    try:
        with h5py.File(args.source, "r") as source:
            paths = source["sol_path"]
            if paths.ndim != 3 or paths.shape[-1] != 7 or paths.shape[0] < 1 or paths.shape[1] < 3:
                raise ValueError("source sol_path must be nonempty [N,waypoints>=3,7]")
            if backend is not None and args.ompl_scheduling == "streaming":
                return generate_ompl_stream(args, source, collision, backend, config_hash,
                                            accepted, attempt, failures, entries, started)
            while accepted < args.num_trajectories and attempt < max_attempts:
                begin = attempt
                count = min(args.query_batch_size, args.num_trajectories - accepted, max_attempts - attempt)
                jobs, batch_failures = [], Counter()
                sync_stage()
                preparation_started = time.perf_counter()
                fill_valid = backend is None and args.torch_batch_policy == "fill"
                while attempt < max_attempts and (len(jobs) < count if fill_valid else attempt - begin < count):
                    current = attempt
                    attempt += 1
                    job, reason = prepare_job(args, source, collision, current)
                    if job is None:
                        batch_failures[reason] += 1
                    else:
                        jobs.append(job)
                sync_stage()
                if pipeline is not None:
                    pipeline["prepared"] += attempt - begin
                    pipeline["preparation_wall_s"] += time.perf_counter() - preparation_started
                records = []
                if jobs:
                    world_started = time.perf_counter()
                    collision.set_worlds([job["forecast"] for job in jobs])
                    sync_stage()
                    if pipeline is not None:
                        pipeline["world_setup_wall_s"] += time.perf_counter() - world_started
                        pipeline["submitted"] += len(jobs)
                        pipeline["planned_batches"] += 1
                        pipeline["full_batches"] += int(len(jobs) == args.query_batch_size)
                        histogram = pipeline["batch_size_histogram"]
                        histogram[str(len(jobs))] = histogram.get(str(len(jobs)), 0) + 1
                    planner_started = time.perf_counter()
                    if backend is None:
                        planner = TorchSpaceTimeForest(
                            collision, collision.robot.q_pos_min, collision.robot.q_pos_max,
                            collision.robot.dq_max * args.velocity_scale, edges_per_query=args.edges_per_query,
                            roots=args.goal_roots, max_iterations=args.max_iterations, joint_step=args.joint_step,
                            time_step=args.edge_dt, recycle=not args.no_root_recycling)
                        results = planner.plan_batch(
                            np.stack([j["q_start"] for j in jobs]), np.stack([j["q_goal"] for j in jobs]),
                            np.full(len(jobs), args.horizon),
                            np.array([float(np.max(j["forecast"].event_times)) + 0.25
                                      if len(j["forecast"].event_times) else 0. for j in jobs]),
                            allowed_time=args.solve_seconds, seed=jobs[0]["seed"],
                            reference_paths=np.stack([j["reference"] for j in jobs]))
                    else:
                        results = backend.plan_batch(jobs)
                    sync_stage()
                    if pipeline is not None:
                        pipeline["planner_wall_s"] += time.perf_counter() - planner_started
                        pipeline["completed"] += len(results)
                        pipeline["raw_solved_queries"] += sum(bool(result.paths) for result in results)
                    postprocess_started = time.perf_counter()
                    for query, (job, result) in enumerate(zip(jobs, results)):
                        record, reason = validate_result(args, collision, query, job, result)
                        if record is None:
                            batch_failures[reason] += 1
                        else:
                            records.append(record)
                    sync_stage()
                    if pipeline is not None:
                        pipeline["postprocess_wall_s"] += time.perf_counter() - postprocess_started
                        pipeline["accepted"] += len(records)
                write_started = time.perf_counter()
                destination = write_shard(output, shard_index, begin, attempt, records, batch_failures, config_hash)
                if pipeline is not None:
                    pipeline["shard_write_wall_s"] += time.perf_counter() - write_started
                shard_index += 1
                session_batches += 1
                accepted += len(records)
                failures.update(batch_failures)
                entries.extend({"file": str(destination.relative_to(output)), "group": "samples/%09d" % job["attempt_id"],
                                "family_id": job["family_id"], "attempt_id": job["attempt_id"],
                                "duration_s": fitted["quality"]["duration"], "modes": job["forecast"].modes,
                                "tau_r_valid": fitted["tau_r_valid"]} for job, _, fitted in records)
                if session_batches % args.checkpoint_batches == 0 or accepted >= args.num_trajectories or attempt >= max_attempts:
                    publish()
                print(json.dumps({"accepted": accepted, "attempts": attempt, "batch_accepted": len(records),
                                  "batch_failures": dict(batch_failures), "effective_queries": len(jobs),
                                  "wall_s": round(time.perf_counter() - started, 2)}), flush=True)
        publish()
        return 0 if accepted >= args.num_trajectories else 2
    finally:
        if backend is not None:
            # On an exceptional exit don't wait indefinitely for outstanding work.
            if sys.exc_info()[0] is None:
                backend.close()
            else:
                backend.terminate()


def main(argv=None):
    import fcntl
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another writer owns this output directory")
        try:
            return _generate(args)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
