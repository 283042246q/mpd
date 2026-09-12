#!/usr/bin/env python3
"""Benchmark Marvin IK and planning beyond the training workspace.

The benchmark has three resumable stages:

1. expand disjoint translation intervals into atomic workspace regions, audit
   single-arm IK reachability, and generate collision-valid dual-arm requests;
2. run RRTConnect on the exact joint-space endpoint pairs;
3. run MPD on the same requests and aggregate the results.

Run this entrypoint from the ``mpd-splines-public`` environment.  Region
endpoint sampling uses system entropy; every accepted request is persisted so
both planners always receive identical endpoints.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from itertools import product
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.interpolate import BSpline
import yaml

from mpd.bimanual.runtime_contract import BimanualRequest
from mpd.bimanual.start_goal_sources import _base_request, _pose_from_matrix


ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ROOT / "scripts/inference/inference_marvin_bimanual.py"
DEFAULT_MPD_CONFIG = (
    ROOT
    / "scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml"
)
DEFAULT_REGIONS = (
    ROOT
    / "scripts/inference/cfgs/start_goal_regions/EnvWarehouse-RobotMarvinBimanual-generalization-regions.yaml"
)
DEFAULT_TRAINING_REGIONS = (
    ROOT / "data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-independent.yaml"
)
ARMS = ("left", "right")


# Each value is (base region name, z-interval index).  Both arms remain active
# in every case.  The non-crossing arm moves away from the target table during
# unilateral crossing tests so the crossed workspace is not occupied at goal.
BUILTIN_CASES = {
    "table_to_shelf_lower": {
        "start": {"left": ("left_table", 0), "right": ("right_table", 0)},
        "goal": {"left": ("left_cabinet", 0), "right": ("right_cabinet", 0)},
    },
    "table_to_shelf_upper": {
        "start": {"left": ("left_table", 0), "right": ("right_table", 0)},
        "goal": {"left": ("left_cabinet", 1), "right": ("right_cabinet", 1)},
    },
    "shelf_lower_to_upper": {
        "start": {"left": ("left_cabinet", 0), "right": ("right_cabinet", 0)},
        "goal": {"left": ("left_cabinet", 1), "right": ("right_cabinet", 1)},
    },
    "left_cross_to_right_table": {
        "start": {"left": ("left_table", 0), "right": ("right_table", 0)},
        "goal": {"left": ("right_table", 0), "right": ("right_cabinet", 0)},
    },
    "right_cross_to_left_table": {
        "start": {"left": ("left_table", 0), "right": ("right_table", 0)},
        "goal": {"left": ("left_cabinet", 0), "right": ("left_table", 0)},
    },
    "bilateral_cross_table": {
        "start": {"left": ("left_table", 0), "right": ("right_table", 0)},
        "goal": {"left": ("right_table", 0), "right": ("left_table", 0)},
    },
}


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _translation_contains(region, point):
    for axis_index, axis in enumerate("xyz"):
        intervals = np.asarray(region["translation"][axis], dtype=float)
        value = float(point[axis_index])
        if not np.any((intervals[:, 0] <= value) & (value <= intervals[:, 1])):
            return False
    return True


def expand_atomic_translation_regions(placement_regions):
    """Expand Cartesian products of disjoint XYZ intervals into named boxes."""
    expanded = {}
    metadata = {}
    for base_name, source in placement_regions.items():
        intervals = {
            axis: np.asarray(source["translation"][axis], dtype=float)
            for axis in "xyz"
        }
        if any(values.ndim != 2 or values.shape[1] != 2 or not len(values) for values in intervals.values()):
            raise ValueError(f"{base_name}.translation axes must contain [low, high] intervals")
        counts = tuple(len(intervals[axis]) for axis in "xyz")
        for indices in product(*(range(count) for count in counts)):
            suffix = "_".join(
                f"{axis}{index}"
                for axis, index, count in zip("xyz", indices, counts)
                if count > 1
            )
            alias = base_name if not suffix else f"{base_name}__{suffix}"
            region = deepcopy(source)
            for axis, index in zip("xyz", indices):
                region["translation"][axis] = [intervals[axis][index].tolist()]
            expanded[alias] = region
            metadata[alias] = {
                "base_region": base_name,
                "interval_indices": dict(zip("xyz", indices)),
                "translation": deepcopy(region["translation"]),
                "rotation": deepcopy(region["rotation"]),
            }
    return expanded, metadata


def _atomic_alias(metadata, base_name, z_index):
    matches = [
        alias
        for alias, item in metadata.items()
        if item["base_region"] == base_name
        and item["interval_indices"] == {"x": 0, "y": 0, "z": z_index}
    ]
    if len(matches) != 1:
        raise ValueError(
            f"case requires exactly one {base_name} atomic region at z interval {z_index}; "
            f"found {matches}"
        )
    return matches[0]


def resolve_cases(case_names, atomic_metadata):
    resolved = {}
    for case_name in case_names:
        specification = BUILTIN_CASES[case_name]
        resolved[case_name] = {
            endpoint: {
                arm: _atomic_alias(atomic_metadata, base_name, z_index)
                for arm, (base_name, z_index) in arm_regions.items()
            }
            for endpoint, arm_regions in specification.items()
        }
    return resolved


def build_safe_mpd_config(base, batch_size):
    """Apply the memory-safe A5 switches while retaining production validation."""
    config = deepcopy(base)
    config["n_trajectory_samples"] = int(batch_size)
    config["runtime_top_k_valid_trajectories"] = min(
        int(config.get("runtime_top_k_valid_trajectories", batch_size)),
        int(batch_size),
    )
    collision = config.setdefault("collision_optimization", {})
    collision["pair_streaming"] = {"enabled": True, "pair_chunk_size": 1024}
    collision["reduced_guide_geometry"] = {
        "enabled": True,
        "profile": "foam_pika_100",
    }
    dense = config.setdefault("dense_validation", {})
    dense["chunking"] = {
        "enabled": True,
        "candidate_chunk_size": min(8, int(batch_size)),
        "time_chunk_size": 32,
        "self_pair_chunk_size": 1024,
    }
    broad_phase = (
        config.setdefault("gradient_pruning", {})
        .setdefault("spatial", {})
        .setdefault("link_broad_phase", {})
    )
    broad_phase.update(
        enabled=True,
        full_scan=True,
        scan_geometry="parent_bounds",
        environment_margin=0.20,
        self_margin=0.10,
    )
    return config


def _actual_pose(generator, q, arm):
    pose = generator._pose(q, arm)
    return np.c_[pose.rotation, pose.translation]


def audit_ik_reachability(
    generator,
    atomic_regions,
    atomic_metadata,
    training_regions,
    attempts,
    attempt_timeout_s,
    restarts=10,
    checkpoint=None,
    previous=None,
):
    """Test fixed random poses with multiple IK seeds; failure is not proof of unreachability."""
    original_tries = generator.config.get("ik_tries", 30)
    original_sample_pose = generator._sample_pose
    report = dict(previous or {})
    try:
        generator.config["ik_tries"] = restarts
        for arm in ARMS:
            for alias, region in atomic_regions.items():
                key = f"{arm}:{alias}"
                if key in report:
                    continue
                records = []
                for attempt in range(attempts):
                    generator.deadline = time.perf_counter() + attempt_timeout_s
                    q_reference = generator._random_valid_state()
                    target_position, target_rotation = original_sample_pose(alias)
                    training_region = training_regions.get(
                        atomic_metadata[alias]["base_region"]
                    )
                    in_training_bounds = bool(
                        training_region is not None
                        and _translation_contains(training_region, target_position)
                    )
                    expected_arm_region = atomic_metadata[alias]["base_region"].startswith(
                        f"{arm}_"
                    )
                    record = {
                        "attempt": attempt,
                        "target_position": target_position.tolist(),
                        "target_rotation": target_rotation.tolist(),
                        "in_same_named_training_bounds": in_training_bounds,
                        "same_side_as_training_tasks": expected_arm_region,
                        "reference_valid": q_reference is not None,
                        "ik_solved": False,
                        "collision_valid": False,
                    }
                    if q_reference is not None:
                        before_hits = int(generator.stats.get("target_pose_hits", 0))
                        generator._sample_pose = (
                            lambda _name, p=target_position, r=target_rotation: (p.copy(), r.copy())
                        )
                        q_solution = generator._target_state(q_reference, arm, alias)
                        record["ik_solved"] = int(
                            generator.stats.get("target_pose_hits", 0)
                        ) > before_hits
                        record["collision_valid"] = q_solution is not None
                        if q_solution is not None:
                            record["q"] = q_solution.tolist()
                            record["achieved_position"] = _actual_pose(
                                generator, q_solution, arm
                            )[:, 3].tolist()
                    records.append(record)
                solved = sum(item["ik_solved"] for item in records)
                collision_valid = sum(item["collision_valid"] for item in records)
                outside = [
                    item for item in records if not item["in_same_named_training_bounds"]
                ]
                report[key] = {
                    "arm": arm,
                    "region": alias,
                    "base_region": atomic_metadata[alias]["base_region"],
                    "bounds": atomic_metadata[alias]["translation"],
                    "attempts": len(records),
                    "ik_solved": solved,
                    "ik_rate": solved / len(records) if records else 0.0,
                    "collision_valid": collision_valid,
                    "collision_valid_rate": collision_valid / len(records) if records else 0.0,
                    "outside_training_attempts": len(outside),
                    "outside_training_collision_valid": sum(
                        item["collision_valid"] for item in outside
                    ),
                    "records": records,
                }
                print(f"IK {key}: {solved}/{attempts} solved, {collision_valid}/{attempts} collision-valid", flush=True)
                if checkpoint is not None:
                    _write_json(checkpoint, {"results": report, "complete": False})
    finally:
        generator.config["ik_tries"] = original_tries
        generator._sample_pose = original_sample_pose
    return report


def _sample_mapping(generator, q_reference, mapping):
    if hasattr(generator, "endpoint_pool"):
        q = np.asarray(q_reference, dtype=float).copy()
        for index, arm in enumerate(ARMS):
            pool = generator.endpoint_pool.get(f"{arm}:{mapping[arm]}", [])
            if not pool:
                return None
            chosen = pool[int(generator.rng.integers(len(pool)))]
            q[index * 7:(index + 1) * 7] = chosen[index * 7:(index + 1) * 7]
        return q if generator.valid(q) else None
    q = np.asarray(q_reference, dtype=float).copy()
    for arm in ARMS:
        q = generator._target_state(q, arm, mapping[arm])
        if q is None:
            return None
    return q if generator.valid(q) else None


def _request_for_pair(generator, case_name, pair_index, regions, q_start, q_goal):
    goal_poses = [_actual_pose(generator, q_goal, arm) for arm in ARMS]
    diffusion_seed = int(generator.rng.integers(0, 2**32, dtype=np.uint64))
    request = _base_request(
        request_id=f"workspace-generalization-{case_name}-{pair_index:03d}",
        seed=diffusion_seed,
        q_start=q_start,
        q_goal=q_goal,
        source={
            "type": "workspace_generalization_regions",
            "sampling": "system_entropy",
            "case": case_name,
            "pair_index": pair_index,
            "start": regions["start"],
            "goal": regions["goal"],
        },
    )
    request["left_goal_pose"] = _pose_from_matrix(goal_poses[0])
    request["right_goal_pose"] = _pose_from_matrix(goal_poses[1])
    BimanualRequest.from_dict(request)
    return request


def generate_pairs(
    generator,
    resolved_cases,
    output_dir,
    pairs_per_case,
    maximum_attempts,
    attempt_timeout_s,
):
    manifest_path = output_dir / "pair-manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {
        "schema": "marvin_workspace_generalization_pairs/v1", "pairs": []}
    manifest["complete"] = False
    minimum_delta = float(generator.config.get("min_active_joint_delta", 0.08))
    for case_name, regions in resolved_cases.items():
        accepted = sum(item["case"] == case_name for item in manifest["pairs"])
        seen = set()
        for item in manifest["pairs"]:
            if item["case"] == case_name:
                saved = _load_json(output_dir / item["request"])
                seen.add(tuple(saved["q_start"] + saved["q_goal"]))
        attempts = 0
        missing = sorted({f"{arm}:{alias}" for mapping in regions.values()
                          for arm, alias in mapping.items()
                          if hasattr(generator, "endpoint_pool") and not generator.endpoint_pool.get(f"{arm}:{alias}")})
        if missing:
            manifest.setdefault("shortfalls", {})[case_name] = {
                "requested": pairs_per_case, "accepted": 0,
                "reason": "empty_endpoint_pool", "missing": missing,
            }
            continue
        while accepted < pairs_per_case and attempts < maximum_attempts:
            attempts += 1
            generator.deadline = time.perf_counter() + attempt_timeout_s
            reference = generator._random_valid_state()
            if reference is None:
                continue
            q_start = _sample_mapping(generator, reference, regions["start"])
            if q_start is None:
                continue
            q_goal = _sample_mapping(generator, q_start, regions["goal"])
            if q_goal is None:
                continue
            if any(
                np.linalg.norm(q_goal[arm_slice] - q_start[arm_slice]) < minimum_delta
                for arm_slice in (slice(0, 7), slice(7, 14))
            ):
                continue
            signature = tuple(q_start.tolist() + q_goal.tolist())
            if signature in seen:
                continue
            seen.add(signature)
            request = _request_for_pair(
                generator, case_name, accepted, regions, q_start, q_goal
            )
            relative = Path("pairs") / case_name / f"pair-{accepted:03d}" / "request.json"
            request_path = output_dir / relative
            _write_json(request_path, request)
            manifest["pairs"].append(
                {
                    "case": case_name,
                    "pair_index": accepted,
                    "request": relative.as_posix(),
                    "start": regions["start"],
                    "goal": regions["goal"],
                }
            )
            accepted += 1
            _write_json(output_dir / "pair-manifest.json", manifest)
        print(f"pair generation {case_name}: {accepted}/{pairs_per_case} ({attempts} attempts)", flush=True)
        if accepted < pairs_per_case:
            manifest.setdefault("shortfalls", {})[case_name] = {
                "requested": pairs_per_case,
                "accepted": accepted,
                "attempts": attempts,
            }
        else:
            manifest.get("shortfalls", {}).pop(case_name, None)
    manifest["complete"] = True
    _write_json(output_dir / "pair-manifest.json", manifest)
    return manifest


def generate_stage(args, output_dir):
    from scripts.generate_data.generate_marvin_warehouse_bimanual import (
        MarvinWarehouseGenerator,
        validate_config,
    )

    snapshot = output_dir / "configs" / "regions-snapshot.yaml"
    region_config = yaml.safe_load((snapshot if args.resume and snapshot.exists() else args.regions).read_text(encoding="utf-8"))
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(yaml.safe_dump(region_config, sort_keys=False), encoding="utf-8")
    validate_config(region_config)
    training_config = yaml.safe_load(args.training_regions.read_text(encoding="utf-8"))
    atomic_regions, atomic_metadata = expand_atomic_translation_regions(
        region_config["placement_regions"]
    )
    resolved_cases = resolve_cases(args.cases, atomic_metadata)
    generator = MarvinWarehouseGenerator(
        region_config, None, progress_label="workspace-generalization"
    )
    generator.regions.update(atomic_regions)
    try:
        reachability = audit_ik_reachability(
            generator,
            atomic_regions,
            atomic_metadata,
            training_config.get("placement_regions", {}),
            args.ik_attempts,
            args.endpoint_timeout_s,
            restarts=args.ik_restarts,
            checkpoint=output_dir / "ik-progress.json",
            previous=_load_json(output_dir / "ik-progress.json")["results"] if args.resume and (output_dir / "ik-progress.json").exists() else None,
        )
        _write_json(
            output_dir / "ik-reachability.json",
            {
                "schema": "marvin_workspace_ik_reachability/v1",
                "sampling": "system_entropy",
                "ik_restarts_per_pose": args.ik_restarts,
                "regions_file": str(args.regions),
                "training_regions_file": str(args.training_regions),
                "regions": atomic_metadata,
                "results": reachability,
            },
        )
        generator.endpoint_pool = {
            key: [record["q"] for record in value["records"] if record["collision_valid"]]
            for key, value in reachability.items()
        }
        manifest = generate_pairs(
            generator,
            resolved_cases,
            output_dir,
            args.pairs_per_case,
            args.max_pair_attempts,
            args.endpoint_timeout_s,
        )
    finally:
        generator.close()
    return manifest


def _counter_delta(after, before):
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in set(after) | set(before)
        if after.get(key, 0) != before.get(key, 0)
    }


def run_rrt_once(generator, request, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    before = Counter(generator.stats)
    started = time.perf_counter()
    path = generator.plan_once(request["q_start"], request["q_goal"], "dual_independent")
    raw_wall = time.perf_counter() - started
    after_raw = Counter(generator.stats)
    result = {
        "schema": "marvin_workspace_rrt_result/v1",
        "request_id": request["request_id"],
        "raw_wall_seconds": raw_wall,
        "raw_success": path is not None,
        "spline_success": False,
        "stats": _counter_delta(after_raw, before),
    }
    if path is None:
        result["status"] = (
            "no_exact_solution"
            if after_raw.get("rrt_no_exact_solution", 0)
            > before.get("rrt_no_exact_solution", 0)
            else "raw_path_rejected"
        )
        _write_json(output_dir / "result.json", result)
        return result

    result["raw_joint_path_length"] = float(
        np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    )
    spline_started = time.perf_counter()
    spline = generator.validated_spline(path)
    result["spline_wall_seconds"] = time.perf_counter() - spline_started
    if spline is None:
        result["status"] = "spline_rejected"
        np.savez_compressed(output_dir / "trajectory.npz", raw_positions=path)
        _write_json(output_dir / "result.json", result)
        return result

    knots, control_points, degree = spline
    spline_positions = BSpline(knots, control_points.T, degree)(
        np.linspace(0.0, 1.0, 128)
    )
    result.update(
        status="success",
        spline_success=True,
        total_wall_seconds=time.perf_counter() - started,
        spline_joint_path_length=float(
            np.linalg.norm(np.diff(spline_positions, axis=0), axis=1).sum()
        ),
        final_joint_error_rad=float(
            np.max(np.abs(spline_positions[-1] - np.asarray(request["q_goal"])))
        ),
    )
    np.savez_compressed(
        output_dir / "trajectory.npz",
        raw_positions=path,
        spline_positions=spline_positions,
        bspline_knots=knots,
        bspline_control_points=control_points,
        bspline_degree=np.asarray(degree),
    )
    _write_json(output_dir / "result.json", result)
    return result


def run_rrt_stage(args, output_dir, manifest):
    from scripts.generate_data.generate_marvin_warehouse_bimanual import (
        MarvinWarehouseGenerator,
        validate_config,
    )

    snapshot = output_dir / "configs" / "regions-snapshot.yaml"
    config = yaml.safe_load((snapshot if snapshot.exists() else args.regions).read_text(encoding="utf-8"))
    validate_config(config)
    config["planner_allowed_time"] = args.rrt_timeout_s
    generator = MarvinWarehouseGenerator(config, None, progress_label="workspace-rrt")
    try:
        for item in manifest["pairs"]:
            request = _load_json(output_dir / item["request"])
            destination = output_dir / "rrt" / item["case"] / f"pair-{item['pair_index']:03d}"
            if args.resume and (destination / "result.json").is_file():
                continue
            result = run_rrt_once(generator, request, destination)
            print(f"RRT {item['case']}[{item['pair_index']}]: {result['status']}", flush=True)
    finally:
        generator.close()


def _classify_mpd(returncode, result, stdout, stderr):
    text = "\n".join((stdout, stderr, str(result.get("error", {})))).lower()
    if "out of memory" in text:
        return "cuda_oom"
    if result.get("status"):
        return str(result["status"])
    return "process_error" if returncode else "missing_result"


def run_mpd_once(args, request_path, config_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(INFERENCE),
        "--request",
        str(request_path),
        "--config",
        str(config_path),
        "--output-dir",
        str(destination),
        "--device",
        args.device,
        "--sim-backend",
        "none",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=args.mpd_timeout_s,
        )
        timed_out = False
    except subprocess.TimeoutExpired as error:
        completed = error
        timed_out = True
    wall_seconds = time.perf_counter() - started
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    (destination / "stdout.log").write_text(stdout, encoding="utf-8")
    (destination / "stderr.log").write_text(stderr, encoding="utf-8")
    result_path = destination / "result.json"
    result = _load_json(result_path) if result_path.is_file() else {}
    benchmark = {
        "schema": "marvin_workspace_mpd_result/v1",
        "status": "timeout"
        if timed_out
        else _classify_mpd(completed.returncode, result, stdout, stderr),
        "returncode": None if timed_out else completed.returncode,
        "wall_seconds": wall_seconds,
        "command": command,
        "result_status": result.get("status"),
        "validation": result.get("validation"),
        "timing": result.get("timing"),
        "candidates": result.get("candidates"),
        "cuda_memory": result.get("cuda_memory"),
        "error": result.get("error"),
        "joint_path_length": float(np.linalg.norm(np.diff(np.asarray(result["positions"]), axis=0), axis=1).sum()) if result.get("positions") else None,
    }
    _write_json(destination / "benchmark-result.json", benchmark)
    return benchmark


def run_mpd_stage(args, output_dir, manifest):
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: run in a GPU-accessible environment; no CPU fallback or batch change is performed")
    base = yaml.safe_load(args.mpd_config.read_text(encoding="utf-8"))
    config = build_safe_mpd_config(base, args.batch_size) if args.safe_mpd else base
    config_path = output_dir / "configs" / "mpd-benchmark.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and yaml.safe_load(config_path.read_text()) != config:
        raise ValueError("MPD config differs from saved run; use a new output directory")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for item in manifest["pairs"]:
        request_path = output_dir / item["request"]
        destination = output_dir / "mpd" / item["case"] / f"pair-{item['pair_index']:03d}"
        if args.resume and (destination / "benchmark-result.json").is_file():
            continue
        result = run_mpd_once(args, request_path, config_path, destination)
        print(f"MPD {item['case']}[{item['pair_index']}]: {result['status']}", flush=True)


def _median(values):
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return statistics.median(finite) if finite else None


def summarize(output_dir, manifest):
    rows = []
    for case_name in sorted({item["case"] for item in manifest["pairs"]}):
        pairs = [item for item in manifest["pairs"] if item["case"] == case_name]
        rrt = []
        mpd = []
        paired = Counter()
        for item in pairs:
            pair_name = f"pair-{item['pair_index']:03d}"
            rrt_path = output_dir / "rrt" / case_name / pair_name / "result.json"
            mpd_path = output_dir / "mpd" / case_name / pair_name / "benchmark-result.json"
            if rrt_path.is_file():
                rrt.append(_load_json(rrt_path))
            if mpd_path.is_file():
                mpd.append(_load_json(mpd_path))
            if rrt_path.is_file() and mpd_path.is_file():
                r = _load_json(rrt_path)
                m = _load_json(mpd_path)
                if m["status"] in ("success", "no_valid_trajectory"):
                    paired[f"mpd_{m['status'] == 'success'}_rrt_{bool(r.get('raw_success'))}"] += 1
        successful_mpd = [item for item in mpd if item["status"] == "success"]
        successful_rrt = [item for item in rrt if item.get("spline_success")]
        rows.append(
            {
                "case": case_name,
                "pairs": len(pairs),
                "paired_outcomes_raw_rrt": dict(paired),
                "mpd_status_counts": dict(Counter(item["status"] for item in mpd)),
                "mpd_median_joint_path_length": _median(item.get("joint_path_length") for item in successful_mpd),
                "mpd_runs": len(mpd),
                "mpd_success": len(successful_mpd),
                "mpd_success_rate": len(successful_mpd) / sum(item["status"] in ("success", "no_valid_trajectory") for item in mpd) if any(item["status"] in ("success", "no_valid_trajectory") for item in mpd) else None,
                "mpd_median_wall_s": _median(item.get("wall_seconds") for item in mpd),
                "mpd_median_core_s": _median(
                    item.get("timing", {}).get("inference_total_s")
                    for item in successful_mpd
                ),
                "mpd_median_environment_clearance_m": _median(
                    item.get("validation", {}).get("minimum_environment_clearance_m")
                    for item in successful_mpd
                ),
                "mpd_median_interarm_clearance_m": _median(
                    item.get("validation", {}).get("minimum_interarm_clearance_m")
                    for item in successful_mpd
                ),
                "rrt_runs": len(rrt),
                "rrt_raw_success": sum(item.get("raw_success", False) for item in rrt),
                "rrt_spline_success": len(successful_rrt),
                "rrt_spline_success_rate": len(successful_rrt) / len(rrt) if rrt else None,
                "rrt_median_total_wall_s": _median(
                    item.get("total_wall_seconds") for item in successful_rrt
                ),
                "rrt_median_joint_path_length": _median(
                    item.get("spline_joint_path_length") for item in successful_rrt
                ),
            }
        )
    summary = {
        "schema": "marvin_workspace_generalization_summary/v1",
        "sampling_shortfalls": manifest.get("shortfalls", {}),
        "rows": rows,
    }
    _write_json(output_dir / "summary.json", summary)
    if rows:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "rrt", "mpd", "all"), default="generate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--training-regions", type=Path, default=DEFAULT_TRAINING_REGIONS)
    parser.add_argument("--mpd-config", type=Path, default=DEFAULT_MPD_CONFIG)
    parser.add_argument("--cases", nargs="+", choices=tuple(BUILTIN_CASES), default=tuple(BUILTIN_CASES))
    parser.add_argument("--ik-attempts", type=int, default=100)
    parser.add_argument("--ik-restarts", type=int, default=10)
    parser.add_argument("--pairs-per-case", type=int, default=20)
    parser.add_argument("--max-pair-attempts", type=int, default=1000)
    parser.add_argument("--endpoint-timeout-s", type=float, default=60.0)
    parser.add_argument("--rrt-timeout-s", type=float, default=10.0)
    parser.add_argument("--mpd-timeout-s", type=float, default=900.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    safe = parser.add_mutually_exclusive_group()
    safe.add_argument("--safe-mpd", dest="safe_mpd", action="store_true")
    safe.add_argument("--raw-mpd-config", dest="safe_mpd", action="store_false")
    parser.set_defaults(safe_mpd=True)
    args = parser.parse_args(argv)
    for name in ("regions", "training_regions", "mpd_config"):
        value = getattr(args, name).expanduser().resolve()
        if not value.is_file():
            parser.error(f"--{name.replace('_', '-')} is not a file: {value}")
        setattr(args, name, value)
    if min(args.ik_attempts, args.ik_restarts, args.pairs_per_case, args.max_pair_attempts, args.batch_size) < 1:
        parser.error("attempt, pair, and batch counts must be positive")
    if min(args.endpoint_timeout_s, args.rrt_timeout_s, args.mpd_timeout_s) <= 0:
        parser.error("timeouts must be positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main(argv=None):
    args = _parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "pair-manifest.json"
    if args.stage in {"generate", "all"}:
        if manifest_path.exists() and not args.resume:
            raise SystemExit(f"benchmark already exists; use --resume or another output: {output_dir}")
        existing = _load_json(manifest_path) if args.resume and manifest_path.is_file() else None
        manifest = existing if existing and existing.get("complete") else generate_stage(args, output_dir)
    else:
        if not manifest_path.is_file():
            raise SystemExit(f"generate stage is required first: {manifest_path}")
        manifest = _load_json(manifest_path)
    if args.stage in {"rrt", "all"}:
        run_rrt_stage(args, output_dir, manifest)
    if args.stage in {"mpd", "all"}:
        run_mpd_stage(args, output_dir, manifest)
    summary = summarize(output_dir, manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if manifest.get("shortfalls") else 0


if __name__ == "__main__":
    raise SystemExit(main())
