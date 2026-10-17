"""Refit warehouse raw q-t paths to public-compatible P21 / EE-goal labels.

Run from the MPD root with ``python -m scripts.spacetime_data.refit_warehouse_eegoal``.
Inputs are committed warehouse_spacetime_joint_v1 grouped shards (including
multiworld), never the older row-major spacetime_mpd_v1 timing-only dataset.
Sources are read-only. A frozen input list and atomic output shards allow resume
even while the original generator publishes additional samples.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import h5py
import numpy as np
import scipy
import torch
import yaml

from mpd.datasets.spacetime_schema import (
    load_panda_robot_bundle, materialize_dataset_contract, sha256_file,
)
from scripts.spacetime_data import generate_warehouse_spacetime as base
from scripts.spacetime_data.warehouse_trajectory_fit import fit_teacher
from scripts.spacetime_data.warehouse_world import Forecast, WarehouseCollision


SCHEMA = "warehouse_spacetime_eegoal_v1"
FORMAT = "warehouse_eegoal_raw_refit_v1"
SPLITS = ("train", "validation", "test")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def within(path, parent):
    return path == parent or parent in path.parents


def learnable_points(full):
    """15 internal points and one generated terminal joint configuration."""
    full = np.asarray(full)
    if full.shape != (21, 7) or not np.isfinite(full).all():
        raise ValueError("expected finite P21 [21,7]")
    if not np.allclose(full[:3], full[0], atol=1e-10, rtol=0):
        raise ValueError("non-rest start")
    if not np.allclose(full[-3:], full[-1], atol=1e-10, rtol=0):
        raise ValueError("non-rest goal")
    return np.concatenate((full[3:-3], full[-1:]), axis=0)


def read_forecast(group):
    metadata = json.loads(group.attrs["world_metadata_json"])
    expected = {"frame": "panda_link0", "time_origin": "execution_start",
                "evaluator": "piecewise_linear_v1"}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported world frame/time/evaluator")
    w = group["world"]
    times, positions = np.asarray(w["times"]), np.asarray(w["positions"])
    shape, size = np.asarray(w["shape"]), np.asarray(w["size"])
    events, anchors = np.asarray(w["event_times"]), np.asarray(w["anchors"])
    n = len(shape)
    if (times.ndim != 1 or len(times) < 2 or abs(times[0]) > 1e-8
        or np.any(np.diff(times) <= 0) or positions.shape != (len(times), n, 3)
        or shape.shape != (n,) or size.shape != (n, 3)
        or events.shape != (n,) or anchors.shape != (n, 3)
        or len(metadata["modes"]) != n or not np.isin(shape, (0, 1, 2)).all()
        or any(not np.isfinite(x).all() for x in (times, positions, size, events, anchors))
        or np.any(size < 0) or (n and np.any(size[:, 0] <= 0))):
        raise ValueError("malformed saved world")
    return Forecast(times, positions, shape, size, events, metadata["modes"], anchors)


def freeze_inputs(roots, limit, bundle):
    """Snapshot published entries, preserving source family splits verbatim."""
    sources, entries, file_stats, family_splits = [], [], {}, {}
    for root_id, root in enumerate(roots):
        config = json.loads((root / "config.json").read_text())
        index = json.loads((root / "index.json").read_text())
        manifest = yaml.safe_load((root / "manifest.yaml").read_text())
        config_hash = digest(config)
        if (config.get("schema") != base.SCHEMA or index.get("schema") != base.SCHEMA
            or index.get("config_hash") != config_hash or manifest.get("schema") != base.SCHEMA
            or manifest.get("config_hash") != config_hash or not config.get("oracle_forecast")):
            raise ValueError(f"expected committed {base.SCHEMA} dataset: {root}")
        if config.get("robot_hashes") != bundle.hashes or config.get("static_environment") != "EnvWarehouse":
            raise ValueError(f"robot/static environment mismatch: {root}")
        static_hashes = config.get("static_geometry_hashes")
        if not static_hashes:
            raise ValueError(f"missing static geometry identity: {root}")
        for name, expected in static_hashes.items():
            asset = base.REPOSITORY / "mpd/torch_robotics/torch_robotics/environments" / name
            if sha256_file(asset) != expected:
                raise ValueError(f"static geometry changed: {asset}")
        if config.get("source_group_size") != 2:
            raise ValueError("direction recovery requires the doubled source_group_size=2 contract")
        source_identity = {k: config[k] for k in ("source_size", "source_mtime_ns", "source_group_size")}
        namespace = digest(source_identity)
        sources.append({"root": str(root), "config_hash": config_hash,
                        "config_file_sha256": sha256_file(root / "config.json"),
                        "index_at_freeze_sha256": sha256_file(root / "index.json"),
                        "source_identity": source_identity, "source_namespace": namespace,
                        "static_geometry_hashes": static_hashes})
        seen = set()
        for split in SPLITS:
            for e in index["splits"][split]:
                key = (e["file"], e["group"])
                if key in seen:
                    raise ValueError(f"duplicate source entry: {root}/{key}")
                seen.add(key)
                family = int(e["family_id"])
                family_key = (namespace, family)
                if family_key in family_splits and family_splits[family_key] != split:
                    raise ValueError(f"cross-root family split conflict: {family_key}")
                family_splits[family_key] = split
                if limit and len(entries) >= limit:
                    continue
                path = (root / e["file"]).resolve()
                if (not within(path, root / "shards") or path.suffix != ".hdf5"
                    or not e["group"].startswith("samples/")):
                    raise ValueError("index must reference committed shards/samples")
                if str(path) not in file_stats:
                    stat = path.stat()
                    file_stats[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                entries.append({"root_id": root_id, "file": e["file"], "group": e["group"],
                                "family_id": family, "split": split})
    if not entries:
        raise ValueError("no published input samples selected")
    return {"sources": sources, "files": file_stats, "entries": entries}


def check_frozen_sources(snapshot):
    for source in snapshot["sources"]:
        if sha256_file(Path(source["root"]) / "config.json") != source["config_file_sha256"]:
            raise ValueError("source config changed since input freeze")
        for name, expected in source["static_geometry_hashes"].items():
            asset = base.REPOSITORY / "mpd/torch_robotics/torch_robotics/environments" / name
            if sha256_file(asset) != expected:
                raise ValueError(f"static geometry changed: {asset}")
    for name, expected in snapshot["files"].items():
        stat = Path(name).stat()
        if expected != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
            raise ValueError(f"committed input shard changed: {name}")


def refit_sample(group, entry, collision, *, spatial_rmse_max, timing_rmse_max):
    """One accepted label per source world; try selected raw first, then others."""
    if not bool(group.attrs.get("accepted", False)):
        return None, {"reason": "source_not_accepted", "raw_attempts": []}
    forecast = read_forecast(group)
    if abs(forecast.horizon - 14.) > 1e-7:
        raise ValueError("unified v1 requires a 14-second forecast")
    q_start, q_goal = np.asarray(group["condition/q_start"]), np.asarray(group["condition/q_goal"])
    if q_start.shape != (7,) or q_goal.shape != (7,) or not np.isfinite([q_start, q_goal]).all():
        raise ValueError("invalid joint boundary labels")
    if int(group.attrs["family_id"]) != entry["family_id"]:
        raise ValueError("index/group family mismatch")
    names = sorted(k for k in group["raw"] if k.startswith("solution_"))
    if not names:
        return None, {"reason": "missing_raw_solutions", "raw_attempts": []}
    selected = int(group["raw/fitted_solution_index"][()])
    if not 0 <= selected < len(names):
        raise ValueError("invalid original fitted raw index")
    paths = [np.asarray(group["raw"][k]) for k in names]
    collision.set_worlds([forecast])
    attempts = []
    started = time.perf_counter()
    for raw_id in [selected] + [i for i in range(len(paths)) if i != selected]:
        path = paths[raw_id]
        if (path.ndim != 2 or path.shape[1] != 8 or len(path) < 2
            or not np.isfinite(path).all() or not np.allclose(path[0, :7], q_start, atol=1e-6, rtol=0)
            or not np.allclose(path[-1, :7], q_goal, atol=1e-6, rtol=0)):
            attempts.append({"raw_index": raw_id, "reason": "raw_boundary_or_shape"})
            continue
        fitted, reason = fit_teacher(path, collision, 0, forecast,
            spatial_control_points=21, spatial_rmse_max=spatial_rmse_max,
            timing_rmse_max=timing_rmse_max, duration_floor=2., require_tau_r=True)
        attempts.append({"raw_index": raw_id, "reason": reason})
        if fitted is None:
            continue
        learned = learnable_points(fitted["P"])
        # Match the public dataset's RobotPanda.get_EE_pose convention exactly.
        with torch.no_grad():
            ee_pose = collision.robot.get_EE_pose(torch.as_tensor(q_goal, **collision.tensor_args))
        pose = ee_pose.detach().cpu().numpy().reshape(-1, 3, 4)[0]
        if not np.isfinite(pose).all() or not np.allclose(pose[:, :3].T @ pose[:, :3], np.eye(3), atol=1e-5):
            raise ValueError("invalid EE FK pose")
        row = int(group.attrs["source_row"])
        if row // 2 != entry["family_id"]:
            raise ValueError("source row/family mismatch")
        job = {"attempt_id": entry["conversion_id"], "source_row": row,
               "family_id": entry["family_id"], "source_task_id": int(group.attrs["source_task_id"]),
               "seed": int(group.attrs["seed"]), "forecast": forecast,
               "q_start": q_start, "q_goal": q_goal, "fitted_solution_index": raw_id,
               "fit_wall_s": time.perf_counter() - started}
        result = SimpleNamespace(paths=paths, statistics=json.loads(group.attrs["planner_statistics_json"]))
        extra = {"ee_pose": pose, "learnable": learned, "original_raw_index": selected,
                 "source_world_metadata_json": str(group.attrs["world_metadata_json"]),
                 "attributes": {k: group.attrs[k] for k in
                    ("world_id", "obstacle_count", "direction", "world_seed", "scene_draw", "solver_seed", "batch_seed")
                    if k in group.attrs}}
        extra["attributes"].setdefault("direction", row % 2)
        extra["attributes"].setdefault("obstacle_count", len(forecast.shape))
        return (job, result, fitted, extra), {"reason": None, "raw_attempts": attempts}
    return None, {"reason": attempts[-1]["reason"], "raw_attempts": attempts}


def commit(output, number, begin, rows, records, config_hash, snapshot):
    staging = output / ".refit-staging"
    (staging / "shards").mkdir(parents=True, exist_ok=True)
    for old in (staging / "shards").glob("shard-*.hdf5*"):
        old.unlink()  # Only unpublished files owned by this locked converter.
    failures = Counter(row["reason"] for row in rows if row["status"] != "accepted")
    path = base.write_shard(staging, number, begin, begin + len(rows),
                            [record[:3] for record in records], failures, config_hash)
    with h5py.File(path, "a") as shard:
        shard.attrs["schema"] = SCHEMA
        shard.create_dataset("conversion_outcomes", data=[json.dumps(row) for row in rows],
                             dtype=h5py.string_dtype("utf-8"))
        for job, _, _, extra in records:
            sample = shard[f"samples/{job['attempt_id']:09d}"]
            e = snapshot["entries"][job["attempt_id"]]
            source = snapshot["sources"][e["root_id"]]
            sample.attrs.update(**extra["attributes"], split=e["split"], goal_type="ee_pose",
                source_root=source["root"], source_file=e["file"], source_group=e["group"],
                source_config_hash=source["config_hash"], source_namespace=source["source_namespace"],
                original_fitted_solution_index=extra["original_raw_index"],
                source_world_metadata_json=extra["source_world_metadata_json"],
                refit_format=FORMAT)
            sample.create_dataset("spatial/learnable_control_points", data=extra["learnable"])
            sample.create_dataset("condition/ee_goal_pose", data=extra["ee_pose"])
            sample.create_dataset("condition/ee_goal_position", data=extra["ee_pose"][:, 3])
            sample.create_dataset("condition/ee_goal_orientation", data=extra["ee_pose"][:, :3].reshape(9))
        shard.flush()
    destination = output / "shards" / path.name
    if destination.exists():
        raise FileExistsError(destination)
    os.replace(path, destination)
    return destination


def restore(output, config_hash, snapshot, *, outcomes=None, entries=None, paths=None):
    outcomes = [] if outcomes is None else outcomes
    entries = [] if entries is None else entries
    paths = sorted((output / "shards").glob("shard-*.hdf5")) if paths is None else paths
    for path in paths:
        with h5py.File(path, "r") as shard:
            if shard.attrs["schema"] != SCHEMA or shard.attrs["config_hash"] != config_hash:
                raise ValueError("output shard identity mismatch")
            if int(shard.attrs["attempt_start"]) != len(outcomes):
                raise ValueError("non-contiguous conversion shard ranges")
            rows = [json.loads(x) for x in shard["conversion_outcomes"].asstr()[:]]
            if int(shard.attrs["attempt_stop"]) != len(outcomes) + len(rows):
                raise ValueError("conversion ledger range mismatch")
            for offset, row in enumerate(rows, len(outcomes)):
                if row["conversion_id"] != offset or row["input"] != snapshot["entries"][offset]:
                    raise ValueError("conversion ledger differs from frozen input")
            accepted_ids = {row["conversion_id"] for row in rows if row["status"] == "accepted"}
            if accepted_ids != {int(g.attrs["attempt_id"]) for g in shard["samples"].values()}:
                raise ValueError("accepted ledger/sample mismatch")
            for name, sample in shard["samples"].items():
                entries.append({"file": str(path.relative_to(output)), "group": "samples/" + name,
                    "family_id": int(sample.attrs["family_id"]), "split": str(sample.attrs["split"]),
                    "source_namespace": str(sample.attrs["source_namespace"]),
                    "obstacle_count": int(sample.attrs["obstacle_count"]), "direction": int(sample.attrs["direction"]),
                    "duration_s": json.loads(sample.attrs["quality_json"])["duration"], "tau_r_valid": True})
            outcomes.extend(rows)
    return outcomes, entries


def publish(output, config_hash, snapshot, outcomes, entries):
    splits = {s: [e for e in entries if e["split"] == s] for s in SPLITS}
    base.atomic_json(output / "index.json", {"schema": SCHEMA, "config_hash": config_hash, "splits": splits})
    per_count = {}
    for row in outcomes:
        key = str(row.get("obstacle_count", "unknown"))
        counter = per_count.setdefault(key, Counter())
        counter[row["status"]] += 1
    base.atomic_json(output / "report.json", {
        "schema": SCHEMA, "config_hash": config_hash, "selected": len(snapshot["entries"]),
        "processed": len(outcomes), "accepted": len(entries),
        "rejected": len(outcomes) - len(entries), "run_finished": len(outcomes) == len(snapshot["entries"]),
        "failure_counts": dict(Counter(r["reason"] for r in outcomes if r["status"] != "accepted")),
        "per_obstacle_count": per_count, "split_counts": {s: len(v) for s, v in splits.items()},
        "positive_definition": "P21 fitted from raw; c and tau_r independently revalidated against original world",
        "safety_scope": "sampled sphere/primitive phase-and-time checks; not continuous collision certification"})


def run(args, *, collision=None):
    output = args.output_root
    bundle = load_panda_robot_bundle(base.REPOSITORY)
    options = {"format": FORMAT, "roots": [str(p) for p in args.dataset_root],
               "max_samples": args.max_samples, "shard_size": args.shard_size,
               "publish_every": args.publish_every,
               "spatial_rmse_max": args.spatial_rmse_max, "timing_rmse_max": args.timing_rmse_max,
               "device": args.device, "margin": args.margin, "collision_chunk": args.collision_chunk,
               "torch_threads": args.torch_threads, "robot_hashes": bundle.hashes,
               "runtime": {"numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)},
               "software": {name: sha256_file(base.REPOSITORY / name) for name in (
                   "scripts/spacetime_data/refit_warehouse_eegoal.py",
                   "scripts/spacetime_data/warehouse_trajectory_fit.py",
                   "scripts/spacetime_data/warehouse_world.py",
                   "scripts/spacetime_data/generate_warehouse_spacetime.py",
                   "mpd/parametric_trajectory/timing_fitting.py",
                   "mpd/parametric_trajectory/normalized_timing.py",
                   "mpd/datasets/spacetime_legacy.py")}}
    config_path, snapshot_path = output / "config.json", output / "input_snapshot.json"
    if args.resume:
        config = json.loads(config_path.read_text())
        if config["options"] != options:
            raise ValueError("resume options/software differ; use the same command and code or a new output root")
        snapshot = json.loads(snapshot_path.read_text())
        if digest(snapshot) != config["input_snapshot_hash"]:
            raise ValueError("frozen input snapshot changed")
    else:
        # A prior interrupted initialization without committed shards is recoverable
        # by rerunning into a new root; never overwrite a possibly active dataset.
        if any(p.name != ".writer.lock" for p in output.iterdir()):
            raise ValueError("output root must be empty; use --resume for an existing conversion")
        snapshot = freeze_inputs(args.dataset_root, args.max_samples, bundle)
        config = {"schema": SCHEMA, "options": options, "input_snapshot_hash": digest(snapshot)}
        base.atomic_json(snapshot_path, snapshot)
        base.atomic_json(config_path, config)
    check_frozen_sources(snapshot)
    config_hash = digest(config)
    manifest = {"schema": SCHEMA, "config_hash": config_hash, "goal_type": "ee_pose",
        "robot": bundle.manifest_robot_section(), "static_environment": "EnvWarehouse",
        "spatial": {"num_control_points": 21, "num_learnable_control_points": 16,
                    "degree": 5, "phase_eta": .05, "context_qs": True, "context_ee_goal_pose": True,
                    "layout": "q_start*3, learnable[:15], learnable[-1]*3"},
        "condition": {"q_start": [7], "ee_goal_pose": [3, 4], "ee_goal_position": [3],
                      "ee_goal_orientation": [9], "q_goal_role": "label/analysis only; not a model condition",
                      "ee_link": bundle.ee_link, "frame": "panda_link0", "pose_convention": "[R|p]"},
        "timing": {"representations": ["c", "tau_r"], "latent_dim": 6, "full_control_points": 8,
                   "degree": 3, "codec_grid_size": 2049, "duration_min": 2., "duration_max": 14.,
                   "u_min": .05, "density_floor": .001},
        "world": {"evaluator": "piecewise_linear_v1", "time_origin": "execution_start"},
        "collision_margin_m": args.margin, "split_policy": "preserve frozen parent index split",
        "raw_policy": "one valid refit per world; original fitted raw first, then remaining raw"}
    materialize_dataset_contract(output, bundle, manifest)
    outcomes, entries = restore(output, config_hash, snapshot)
    publish(output, config_hash, snapshot, outcomes, entries)
    if len(outcomes) == len(snapshot["entries"]):
        return 0
    collision = collision or WarehouseCollision(args.device, margin=args.margin, chunk_size=args.collision_chunk)
    number = len(list((output / "shards").glob("shard-*.hdf5")))
    while len(outcomes) < len(snapshot["entries"]):
        begin = len(outcomes)
        rows, records = [], []
        for conversion_id in range(begin, min(begin + args.shard_size, len(snapshot["entries"]))):
            entry = snapshot["entries"][conversion_id]
            source = snapshot["sources"][entry["root_id"]]
            path = Path(source["root"]) / entry["file"]
            with h5py.File(path, "r") as shard:
                if shard.attrs["schema"] != base.SCHEMA or shard.attrs["config_hash"] != source["config_hash"]:
                    raise ValueError("source shard contract mismatch")
                group = shard[entry["group"]]
                record, detail = refit_sample(group, dict(entry, conversion_id=conversion_id), collision,
                    spatial_rmse_max=args.spatial_rmse_max, timing_rmse_max=args.timing_rmse_max)
                count = len(group["world/shape"])
            rows.append({"conversion_id": conversion_id, "input": entry,
                         "status": "accepted" if record is not None else "fit_rejected",
                         "obstacle_count": count, **detail})
            if record is not None:
                records.append(record)
        destination = commit(output, number, begin, rows, records, config_hash, snapshot)
        number += 1
        # Restore is also the crash-recovery path; the committed shard is truth.
        outcomes, entries = restore(output, config_hash, snapshot,
            outcomes=outcomes, entries=entries, paths=[destination])
        if number % args.publish_every == 0 or len(outcomes) == len(snapshot["entries"]):
            publish(output, config_hash, snapshot, outcomes, entries)
        print(json.dumps({"processed": len(outcomes), "selected": len(snapshot["entries"]),
                          "accepted": len(entries), "rejected": len(outcomes) - len(entries)}), flush=True)
    return 0  # completion and per-sample rejection are separate report fields


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, action="append", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-samples", type=int, default=0, help="0=all; deterministic root/split/index prefix")
    p.add_argument("--shard-size", type=int, default=64)
    p.add_argument("--publish-every", type=int, default=16, help="publish derived index/report every N shards")
    p.add_argument("--collision-chunk", type=int, default=512)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--margin", type=float, default=.005)
    p.add_argument("--spatial-rmse-max", type=float, default=.08)
    p.add_argument("--timing-rmse-max", type=float, default=.08)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    args.dataset_root = [root.resolve() for root in args.dataset_root]
    args.output_root = args.output_root.resolve()
    if len(set(args.dataset_root)) != len(args.dataset_root):
        p.error("duplicate input roots")
    if (args.max_samples < 0 or min(args.shard_size, args.publish_every, args.collision_chunk, args.torch_threads) < 1
        or not np.isfinite([args.margin, args.spatial_rmse_max, args.timing_rmse_max]).all()
        or args.margin < 0 or min(args.spatial_rmse_max, args.timing_rmse_max) <= 0):
        p.error("invalid sample count, size, fit tolerance, or margin")
    if any(within(args.output_root, root) or within(root, args.output_root)
           for root in args.dataset_root):
        p.error("output must be separate from input roots, not an ancestor or descendant")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
