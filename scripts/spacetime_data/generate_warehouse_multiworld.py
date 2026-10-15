#!/usr/bin/env python3
"""Generate 0..5-obstacle worlds for sampled, previously solved Warehouse families.

Independent entry point: original generators, samplers and resume contracts are
unchanged. Every family uses one direction across all six world slots. GPU batches
contain one obstacle count. A finite failed slot is recorded, never replaced by
another family or silently retried until successful.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from mpd.datasets.spacetime_schema import load_panda_robot_bundle, materialize_dataset_contract, sha256_file
from scripts.spacetime_data import generate_warehouse_spacetime as base
from scripts.spacetime_data.batch_spacetime_rrt import TorchSpaceTimeForest
from scripts.spacetime_data.multiworld_sampling import make_multiworld_forecast
from scripts.spacetime_data.warehouse_world import WarehouseCollision


COUNTS = tuple(range(6))
DEFAULT_ELIGIBLE_ROOTS = tuple(
    base.REPOSITORY / f"data_trajectories_spacetime/Warehouse-TorchSpaceTime-v1-{n}obstacle"
    for n in (3, 4, 5)
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=base.SOURCE)
    parser.add_argument("--num-families", type=int, required=True)
    parser.add_argument("--eligible-roots", type=Path, nargs="+", default=list(DEFAULT_ELIGIBLE_ROOTS),
                        help="historical accepted datasets with >=3 obstacles; union their family IDs")
    parser.add_argument("--direction", choices=("random", "forward", "reverse"), default="random",
                        help="random assigns one direction independently per family; all six slots share it")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--device", default="cuda:0")
    for name, kind, default in (
        ("query-batch-size", int, 128), ("solve-seconds", float, 10.),
        ("max-world-attempts", int, 12), ("checkpoint-batches", int, 10),
        ("edges-per-query", int, 16), ("goal-roots", int, 4), ("max-iterations", int, 512),
        ("horizon", float, 14.), ("forecast-dt", float, .02), ("edge-dt", float, .02),
        ("joint-step", float, .04), ("velocity-scale", float, .4), ("margin", float, .005),
        ("collision-chunk", int, 32768), ("spatial-rmse-max", float, .08),
        ("timing-rmse-max", float, .08),
    ):
        parser.add_argument("--" + name, type=kind, default=default)
    parser.add_argument("--require-tau-r", action="store_true")
    parser.add_argument("--no-root-recycling", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="freeze family plan and contract without allocating a GPU")
    cli = parser.parse_args(argv)
    if cli.num_families < 1 or cli.seed < 0:
        parser.error("num-families must be positive and seed nonnegative")
    # Reuse the original argument validation without exposing its task traversal,
    # count, backend, or accepted-trajectory target in this new CLI.
    common = ["--output-root", str(cli.output_root), "--source", str(cli.source),
              "--device", cli.device, "--seed", str(cli.seed)]
    for name in ("query_batch_size", "solve_seconds", "max_world_attempts", "checkpoint_batches",
                 "edges_per_query", "goal_roots", "max_iterations", "horizon", "forecast_dt",
                 "edge_dt", "joint_step", "velocity_scale", "margin", "collision_chunk",
                 "spatial_rmse_max", "timing_rmse_max"):
        common += ["--" + name.replace("_", "-"), str(getattr(cli, name))]
    args = base.parse_args(common)
    vars(args).update(vars(cli))
    args.source, args.output_root = cli.source.resolve(), cli.output_root.resolve()
    args.eligible_roots = [path.resolve() for path in cli.eligible_roots]
    args.num_trajectories = cli.num_families * len(COUNTS)
    return args


def split_for(family):
    bucket = int(hashlib.sha256(str(int(family)).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else ("validation" if bucket < 90 else "test")


def keyed_seed(seed, source_namespace, *keys):
    return int(np.random.SeedSequence([seed, source_namespace, *map(int, keys)]).generate_state(1)[0])


def select_families(args):
    """Read accepted historical indexes, validate source identity, sample without replacement."""
    stat = args.source.stat()
    source_identity = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "source_group_size": 2}
    namespace = int(hashlib.sha256(json.dumps(source_identity, sort_keys=True).encode()).hexdigest()[:8], 16)
    pool, evidence = set(), []
    with h5py.File(args.source, "r") as source:
        shape = source["sol_path"].shape
        if len(shape) != 3 or shape[0] % 2 or shape[1] < 3 or shape[2] != 7:
            raise ValueError("source must be a doubled [2*families,waypoints,7] dataset")
        source_ids = np.asarray(source["task_id"])
        if source_ids.shape != (shape[0],):
            raise ValueError("source task_id shape mismatch")
    for root in args.eligible_roots:
        config = json.loads((root / "config.json").read_text())
        index = json.loads((root / "index.json").read_text())
        expected = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if index.get("config_hash") != expected or config.get("schema") != base.SCHEMA:
            raise ValueError("historical dataset contract mismatch: " + str(root))
        if (config.get("obstacles", -1) < 3 or config.get("source_group_size") != 2
            or config.get("source_size") != stat.st_size or config.get("source_mtime_ns") != stat.st_mtime_ns):
            raise ValueError("historical dataset must use >=3 obstacles and the same doubled source: " + str(root))
        families = {int(entry["family_id"]) for entries in index["splits"].values() for entry in entries}
        if any(family < 0 or 2 * family + 1 >= shape[0] for family in families):
            raise ValueError("historical family ID outside source")
        pool.update(families)
        evidence.append({"root": str(root), "obstacles": config["obstacles"], "families": len(families),
                         "config_sha256": sha256_file(root / "config.json"),
                         "index_sha256": sha256_file(root / "index.json")})
    if args.num_families > len(pool):
        raise ValueError(f"requested {args.num_families} families, but eligible union has {len(pool)}")
    rng = np.random.default_rng(keyed_seed(args.seed, namespace, 10))
    selected = rng.permutation(np.array(sorted(pool), dtype=np.int64))[:args.num_families]
    plan = []
    for family in selected:
        family = int(family)
        direction = (keyed_seed(args.seed, namespace, 11, family) % 2 if args.direction == "random"
                     else int(args.direction == "reverse"))
        row = 2 * family + direction
        plan.append({"family_id": family, "direction": direction, "source_row": row,
                     "source_task_id": int(source_ids[row]), "split": split_for(family)})
    return plan, {"eligible_families": len(pool), "source_identity": source_identity,
                  "source_namespace": namespace, "historical_datasets": evidence}


def make_config(args, robot, plan, selection):
    config = base.run_identity(args, robot)
    for key in ("dry_run", "obstacles", "start_index", "max_attempts", "torch_batch_policy",
                "cpu_workers", "ompl_scheduling", "ompl_queue_size"):
        config.pop(key, None)
    config.update(generator="warehouse_family_multiworld_v1", obstacle_counts=list(COUNTS),
                  requested_worlds=len(plan) * len(COUNTS), selection=selection,
                  family_plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                  scheduling="round-robin count buckets; valid batches; no cross-count padding",
                  world_sampling={"shape": "iid_uniform_sphere_box_capsule",
                                  "motion_mode": "iid_uniform_with_replacement_from_motion_modes",
                                  "event_fraction": "iid_uniform; same fraction selects reference waypoint anchor",
                                  "event_fraction_ranges": {"0": None, "1-2": [.25, .65], "3-5": [.25, .80]},
                                  "nominal_duration": 8., "reference_retime": False},
                  seed_policy="source namespace/family/direction/count/scene draw; separate solver namespace",
                  slot_failure_policy="at most max_world_attempts scene draws; submit first initially valid scene once")
    for filename in ("generate_warehouse_multiworld.py", "multiworld_sampling.py"):
        config["software"][filename] = sha256_file(Path(__file__).with_name(filename))
    return base.json_value(config)


def restore(output, config_hash, plan):
    accepted, processed, failures, entries = base.read_progress(output, config_hash)
    cursors, outcomes, batch_stats = {n: 0 for n in COUNTS}, [], []
    for path in sorted((output / "shards").glob("*.hdf5")):
        with h5py.File(path) as shard:
            rows = [json.loads(value) for value in shard["task_outcomes"].asstr()[:]]
            accepted_ids = {int(sample.attrs["attempt_id"]) for sample in shard["samples"].values()}
            if {row["attempt_id"] for row in rows if row["status"] == "accepted"} != accepted_ids:
                raise ValueError("accepted outcomes and saved samples differ")
            for row in rows:
                count = row["obstacle_count"]
                cursor = cursors[count]
                if cursor >= len(plan) or row["family_id"] != plan[cursor]["family_id"]:
                    raise ValueError("world ledger differs from frozen family plan")
                if row["direction"] != plan[cursor]["direction"] or row["attempt_id"] != len(outcomes):
                    raise ValueError("world ledger direction or attempt sequence differs")
                cursors[count] += 1
                outcomes.append(row)
            batch_stats.append(json.loads(shard.attrs["batch_statistics_json"]))
    if processed != len(outcomes):
        raise ValueError("committed attempts and world ledger differ")
    return accepted, processed, failures, entries, cursors, outcomes, batch_stats


def commit_batch(output, shard_index, begin, outcomes, records, failures, config_hash, statistics):
    """Reuse the sample writer, then atomically publish samples and their task ledger together."""
    staging = output / ".batch-staging"
    (staging / "shards").mkdir(parents=True, exist_ok=True)
    # Only this output's locked, unpublished staging file may be discarded.
    for path in (staging / "shards").glob("shard-*.hdf5*"):
        path.unlink()
    path = base.write_shard(staging, shard_index, begin, begin + len(outcomes), records, failures, config_hash)
    by_attempt = {row["attempt_id"]: row for row in outcomes}
    with h5py.File(path, "a") as shard:
        shard.create_dataset("task_outcomes", data=[json.dumps(row) for row in outcomes],
                             dtype=h5py.string_dtype("utf-8"))
        shard.attrs["batch_statistics_json"] = json.dumps(statistics)
        for sample in shard["samples"].values():
            row = by_attempt[int(sample.attrs["attempt_id"])]
            sample.attrs.update(world_id=row["obstacle_count"], obstacle_count=row["obstacle_count"],
                                direction=row["direction"], world_seed=row["world_seed"],
                                scene_draw=row["scene_draw"], solver_seed=row["solver_seed"],
                                batch_seed=statistics["batch_seed"])
        shard.flush()
    destination = output / "shards" / path.name
    if destination.exists():
        raise FileExistsError(destination)
    os.replace(path, destination)


def publish(output, config_hash, plan, state, elapsed):
    accepted, processed, failures, entries, cursors, outcomes, batches = state
    requested = len(plan) * len(COUNTS)
    base.publish_index(output, config_hash, accepted, processed, failures, entries, elapsed, requested)
    report = json.loads((output / "report.json").read_text())
    per_count = {}
    family_successes = Counter(row["family_id"] for row in outcomes if row["status"] == "accepted")
    for count in COUNTS:
        rows = [row for row in outcomes if row["obstacle_count"] == count]
        group_batches = [batch for batch in batches if batch["obstacle_count"] == count]
        per_count[str(count)] = {"requested": len(plan), "processed": len(rows),
                                 "pending": len(plan) - cursors[count],
                                 "status_counts": dict(Counter(row["status"] for row in rows)),
                                 "accepted": sum(row["status"] == "accepted" for row in rows),
                                 "submitted": sum(batch["submitted"] for batch in group_batches),
                                 "batches": len(group_batches),
                                 "batch_size_histogram": dict(Counter(str(batch["submitted"]) for batch in group_batches))}
    report.update(generator="warehouse_family_multiworld_v1", requested_families=len(plan),
                  complete=accepted == requested, run_finished=processed == requested,
                  processed_worlds=processed, pending_worlds=requested - processed,
                  failed_worlds=processed - accepted, per_obstacle_count=per_count,
                  direction_family_counts=dict(Counter("reverse" if row["direction"] else "forward" for row in plan)),
                  family_success_histogram=dict(Counter(str(family_successes[row["family_id"]]) for row in plan)),
                  complete_families=sum(family_successes[row["family_id"]] == len(COUNTS) for row in plan),
                  pipeline_totals={key: sum(batch.get(key, 0) for batch in batches) for key in
                                   ("preparation_wall_s", "world_setup_wall_s", "planner_wall_s", "postprocess_wall_s")})
    base.atomic_json(output / "report.json", report)


def _generate(args):
    torch.set_num_threads(1)
    plan, selection = select_families(args)
    robot = load_panda_robot_bundle(base.REPOSITORY)
    config = make_config(args, robot, plan, selection)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = args.output_root
    if (output / "config.json").exists():
        if not args.resume:
            raise FileExistsError("output already initialized; use --resume")
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("resume configuration/assets/software/source/selection do not match")
        if json.loads((output / "families.json").read_text()) != plan:
            raise ValueError("frozen family plan changed")
    else:
        if any(path.name != ".writer.lock" for path in output.iterdir()):
            raise ValueError("refusing to initialize nonempty output without a contract")
        base.atomic_json(output / "families.json", plan)
        base.atomic_json(output / "config.json", config)
    materialize_dataset_contract(output, robot, {
        "schema": base.SCHEMA, "config_hash": config_hash, "robot": robot.manifest_robot_section(),
        "static_environment": "EnvWarehouse", "world": "frozen oracle piecewise-linear forecast",
        "obstacle_static_collision": "ignored", "goal_type": "joint", "codec": "P29-degree5/c8; optional tau_r6",
        "codec_dense_phase_points": 2049, "collision_margin_m": args.margin,
        "backend": "torch-forest", "normalizers": "not fitted; fit on train only"})
    (output / "shards").mkdir(exist_ok=True)
    state = restore(output, config_hash, plan)
    accepted, processed, failures, entries, cursors, outcomes, batches = state
    if args.dry_run or processed == args.num_trajectories:
        previous = json.loads((output / "report.json").read_text()) if (output / "report.json").exists() else {}
        elapsed = previous.get("session_wall_s", 0.) if previous.get("processed_worlds") == processed else 0.
        publish(output, config_hash, plan, state, elapsed)
        print(json.dumps({"dry_run": args.dry_run, "families": len(plan), "eligible": selection["eligible_families"],
                          "requested_worlds": args.num_trajectories, "accepted": accepted, "processed": processed}), flush=True)
        return 0 if args.dry_run or accepted == args.num_trajectories else 2
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu only for a small check")
    collision = WarehouseCollision(args.device, margin=args.margin, chunk_size=args.collision_chunk)
    started = time.perf_counter()

    def sync():
        if collision.tensor_args["device"].type == "cuda":
            torch.cuda.synchronize(collision.tensor_args["device"])

    with h5py.File(args.source) as source:
        @lru_cache(maxsize=max(2 * args.query_batch_size, 16))
        def reference_data(row):
            reference = np.asarray(source["sol_path"][row], dtype=float)
            valid = not bool(collision.static_collision(reference).any().item())
            return reference, valid

        @lru_cache(maxsize=max(2 * args.query_batch_size, 16))
        def reference_centers(row):
            reference, _ = reference_data(row)
            indices = np.linspace(0, len(reference) - 1, min(64, len(reference))).astype(int)
            return collision.centers(reference[indices]).cpu().numpy()

        while processed < args.num_trajectories:
            for count in COUNTS:
                if cursors[count] >= len(plan):
                    continue
                sync()
                stage_started = time.perf_counter()
                batch_rows, jobs = [], []
                begin = processed
                while cursors[count] < len(plan) and len(jobs) < args.query_batch_size:
                    family = plan[cursors[count]]
                    cursors[count] += 1
                    attempt = begin + len(batch_rows)
                    keys = (family["family_id"], family["direction"], count)
                    row = dict(family, obstacle_count=count, world_id=count, attempt_id=attempt,
                               status="invalid_scene", reason=None, scene_draw=None, world_seed=None,
                               solver_seed=keyed_seed(args.seed, selection["source_namespace"], 21, *keys))
                    batch_rows.append(row)
                    reference, valid = reference_data(family["source_row"])
                    if not valid:
                        row["reason"] = "source_static_self_or_limits"
                        continue
                    forecast = None
                    for draw in range(args.max_world_attempts):
                        world_seed = keyed_seed(args.seed, selection["source_namespace"], 20, *keys, draw)
                        row.update(scene_draw=draw, world_seed=world_seed)
                        try:
                            candidate = make_multiworld_forecast(
                                reference_centers(family["source_row"]) if count else None,
                                args.nominal_duration, args.horizon, np.random.default_rng(world_seed),
                                object_count=count, modes=args.motion_modes, forecast_dt=args.forecast_dt)
                        except ValueError:
                            continue
                        collision.set_worlds([candidate])
                        if not bool(collision(reference[:1], [0.], [0]).item()):
                            forecast = candidate
                            break
                    if forecast is None:
                        row["reason"] = "world_anchor_or_initial_collision"
                        continue
                    job = dict(family, attempt_id=attempt, seed=row["world_seed"],
                               q_start=reference[0], q_goal=reference[-1], reference=reference, forecast=forecast,
                               outcome=row)
                    jobs.append(job)
                sync()
                statistics = {"obstacle_count": count, "submitted": len(jobs), "accepted": 0,
                              "preparation_wall_s": time.perf_counter() - stage_started,
                              "batch_seed": jobs[0]["outcome"]["solver_seed"] if jobs else None}
                records = []
                if jobs:
                    stage_started = time.perf_counter()
                    collision.set_worlds([job["forecast"] for job in jobs])
                    sync()
                    statistics["world_setup_wall_s"] = time.perf_counter() - stage_started
                    planner = TorchSpaceTimeForest(
                        collision, collision.robot.q_pos_min, collision.robot.q_pos_max,
                        collision.robot.dq_max * args.velocity_scale, edges_per_query=args.edges_per_query,
                        roots=args.goal_roots, max_iterations=args.max_iterations, joint_step=args.joint_step,
                        time_step=args.edge_dt, recycle=not args.no_root_recycling)
                    stage_started = time.perf_counter()
                    results = planner.plan_batch(
                        np.stack([job["q_start"] for job in jobs]), np.stack([job["q_goal"] for job in jobs]),
                        np.full(len(jobs), args.horizon),
                        np.array([float(np.max(job["forecast"].event_times)) + .25 if count else 0. for job in jobs]),
                        allowed_time=args.solve_seconds, seed=statistics["batch_seed"],
                        reference_paths=np.stack([job["reference"] for job in jobs]))
                    sync()
                    statistics["planner_wall_s"] = time.perf_counter() - stage_started
                    if len(results) != len(jobs):
                        raise RuntimeError("planner returned wrong number of query results")
                    stage_started = time.perf_counter()
                    for query, (job, result) in enumerate(zip(jobs, results)):
                        had_solution = bool(result.paths)
                        record, reason = base.validate_result(args, collision, query, job, result)
                        row = job["outcome"]
                        row.update(status="accepted" if record is not None else
                                   ("fit_rejected" if had_solution else "teacher_unsolved"), reason=reason)
                        if record is not None:
                            records.append(record)
                    sync()
                    statistics["postprocess_wall_s"] = time.perf_counter() - stage_started
                batch_failures = Counter(row["reason"] for row in batch_rows if row["status"] != "accepted")
                statistics["accepted"] = len(records)
                commit_batch(output, len(batches), begin, batch_rows, records, batch_failures, config_hash, statistics)
                processed += len(batch_rows)
                accepted += len(records)
                failures.update(batch_failures)
                outcomes.extend(batch_rows)
                batches.append(statistics)
                entries.extend({"file": f"shards/shard-{len(batches)-1:06d}.hdf5", "group": f"samples/{job['attempt_id']:09d}",
                                "family_id": job["family_id"], "attempt_id": job["attempt_id"],
                                "duration_s": fitted["quality"]["duration"], "modes": job["forecast"].modes,
                                "tau_r_valid": fitted["tau_r_valid"]} for job, _, fitted in records)
                state = accepted, processed, failures, entries, cursors, outcomes, batches
                if len(batches) % args.checkpoint_batches == 0 or processed == args.num_trajectories:
                    publish(output, config_hash, plan, state, time.perf_counter() - started)
                print(json.dumps({"obstacle_count": count, "batch_queries": len(jobs), "batch_accepted": len(records),
                                  "processed_worlds": processed, "accepted": accepted,
                                  "requested": args.num_trajectories, "failures": dict(batch_failures)}), flush=True)
    return 0 if accepted == args.num_trajectories else 2


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
