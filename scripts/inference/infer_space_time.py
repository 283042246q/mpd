#!/usr/bin/env python3
"""Run Phase-5 space-time MPD directly, without a socket or ROS.

The model, planning task and validators are loaded once.  One immutable request
and world snapshot can then be evaluated repeatedly for latency/reproducibility
measurements.  Each run exports the same neutral JSON/NPZ contract used by the
resident runtime, including candidate-specific schema-v3 timing.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from mpd.utils.patches import numpy_monkey_patch

numpy_monkey_patch()

import numpy as np

from scripts.runtime.infer_once import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SEED,
    RuntimeContractError,
    _atomic_write_json,
    _atomic_write_npz,
    _load_request,
)
from scripts.runtime.space_time_runtime_engine import SpaceTimeMpdRuntimeEngine
from scripts.runtime.timing_contract import TIMING_MODES


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_WORLD_VERSION = 1
DEFAULT_WORLD_STAMP_UNIX_NS = 1
DEFAULT_WORLD_VALID_UNTIL_UNIX_NS = 2**63 - 1


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _static_world() -> dict[str, Any]:
    return {
        "world_version": DEFAULT_WORLD_VERSION,
        "frame_id": "fr3_link0",
        "stamp_unix_ns": DEFAULT_WORLD_STAMP_UNIX_NS,
        "valid_until_unix_ns": DEFAULT_WORLD_VALID_UNTIL_UNIX_NS,
        "objects": [],
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def _nested(mapping: dict[str, Any], *keys: str, default=None):
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _run_summary(
    payload: dict[str, Any],
    *,
    index: int,
    seed: int,
    elapsed_sec: float,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "index": index,
        "seed": seed,
        "status": payload.get("status"),
        "elapsed_sec": float(elapsed_sec),
        "duration_s": _nested(payload, "trajectory", "duration_s"),
        "minimum_environment_clearance_m": _nested(
            payload, "trajectory", "minimum_environment_clearance_m"
        ),
        "minimum_self_clearance_m": _nested(
            payload, "trajectory", "minimum_self_clearance_m"
        ),
        "path_length": _nested(
            payload, "best_trajectory_diagnostics", "path_length"
        ),
        "generated_candidates": _nested(payload, "candidates", "generated"),
        "dense_checked_candidates": _nested(
            payload, "candidates", "dense_checked"
        ),
        "dense_complete": _nested(payload, "candidates", "dense_complete"),
        "valid_candidates": _nested(payload, "candidates", "valid"),
        "guide_sec": _nested(payload, "timing", "guide_sec"),
        "dense_validation_sec": _nested(
            payload, "timing", "dense_validation_sec"
        ),
        "result_path": (run_dir / "result.json").as_posix(),
        "trajectory_path": (run_dir / "trajectory.npz").as_posix(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference-only Phase-5 space-time MPD directly in one process. "
            "No Unix socket, ROS, simulator, or robot command is used."
        )
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument(
        "--world",
        type=Path,
        default=None,
        help=(
            "Dynamic-world snapshot JSON. When omitted, request.dynamic_world is "
            "used if present; otherwise only the configured static scene is used."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--timing-mode",
        choices=tuple(sorted(TIMING_MODES)),
        default="phase5_joint",
    )
    parser.add_argument("--repeats", type=_positive_integer, default=1)
    parser.add_argument(
        "--seed",
        type=_nonnegative_integer,
        default=None,
        help="Override request seed; the request seed is used when omitted.",
    )
    parser.add_argument(
        "--seed-step",
        type=_nonnegative_integer,
        default=0,
        help="Add this value to the seed after each repeat; zero gives exact repeats.",
    )
    parser.add_argument(
        "--trajectory-start-unix-ns",
        type=_nonnegative_integer,
        default=None,
        help="Defaults to the world snapshot stamp (or 1 for a zero stamp).",
    )
    parser.add_argument("--max-dynamic-objects", type=_positive_integer, default=16)
    parser.add_argument("--covariance-sigma", type=float, default=3.0)
    parser.add_argument("--process-acceleration-std", type=float, default=0.01)
    parser.add_argument("--timing-control-points", type=_positive_integer, default=8)
    parser.add_argument("--timing-degree", type=_positive_integer, default=3)
    parser.add_argument("--u-min", type=float, default=0.05)
    parser.add_argument("--duration-min", type=float, default=6.0)
    parser.add_argument("--duration-max", type=float, default=14.0)
    parser.add_argument("--nominal-duration", type=float, default=10.0)
    parser.add_argument("--timing-learning-rate", type=float, default=0.08)
    parser.add_argument(
        "--static-spatial-pruning",
        dest="static_spatial_pruning",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-static-spatial-pruning",
        dest="static_spatial_pruning",
        action="store_false",
    )
    parser.add_argument(
        "--dynamic-space-time-pruning",
        action="store_true",
        default=False,
        help="Reserved fail-closed switch; candidate-specific pruning is not implemented.",
    )
    return parser


def run(
    args: argparse.Namespace,
    *,
    engine_factory: Callable[..., Any] = SpaceTimeMpdRuntimeEngine,
) -> Path:
    request_path = args.request.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_request, request_sha256 = _load_request(request_path)
    embedded_world = raw_request.get("dynamic_world")
    if args.world is not None and embedded_world is not None:
        raise ValueError(
            "dynamic world is ambiguous: use either --world or request.dynamic_world"
        )
    if args.world is not None:
        world = _load_json_object(args.world)
        world_source = "--world"
    elif embedded_world is not None:
        if not isinstance(embedded_world, dict):
            raise ValueError("request.dynamic_world must be a JSON object")
        world = deepcopy(embedded_world)
        world_source = "request.dynamic_world"
    else:
        world = _static_world()
        world_source = "static"
    world_version = world.get("world_version")
    stamp_unix_ns = world.get("stamp_unix_ns")
    valid_until_unix_ns = world.get("valid_until_unix_ns")
    if isinstance(world_version, bool) or not isinstance(world_version, int):
        raise ValueError("world_version must be an integer")
    if isinstance(stamp_unix_ns, bool) or not isinstance(stamp_unix_ns, int):
        raise ValueError("stamp_unix_ns must be an integer")
    if isinstance(valid_until_unix_ns, bool) or not isinstance(
        valid_until_unix_ns, int
    ):
        raise ValueError("valid_until_unix_ns must be an integer")
    trajectory_start_unix_ns = (
        max(stamp_unix_ns, 1)
        if args.trajectory_start_unix_ns is None
        else args.trajectory_start_unix_ns
    )
    if trajectory_start_unix_ns < stamp_unix_ns:
        raise ValueError("trajectory start predates the world snapshot")
    if trajectory_start_unix_ns + int(args.duration_max * 1e9) > valid_until_unix_ns:
        raise ValueError(
            "world validity must cover trajectory start plus Phase-5 duration-max"
        )

    request_seed = raw_request.get("seed", DEFAULT_SEED)
    base_seed = request_seed if args.seed is None else args.seed
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("seed must be an integer")
    final_seed = base_seed + (args.repeats - 1) * args.seed_step
    if base_seed < 0 or final_seed >= 2**32:
        raise ValueError("all generated seeds must lie in [0, 2**32)")

    settings = {
        "num_timing_control_points": args.timing_control_points,
        "timing_degree": args.timing_degree,
        "u_min": args.u_min,
        "duration_min": args.duration_min,
        "duration_max": args.duration_max,
        "nominal_duration": args.nominal_duration,
        "timing_learning_rate": args.timing_learning_rate,
    }
    engine = engine_factory(
        config_path=config_path,
        runtime_output_root=output_dir,
        device_text=args.device,
        timing_mode=args.timing_mode,
        space_time_settings=settings,
        max_dynamic_objects=args.max_dynamic_objects,
        covariance_sigma=args.covariance_sigma,
        process_acceleration_std_m_s2=args.process_acceleration_std,
        static_spatial_pruning_enabled=args.static_spatial_pruning,
        dynamic_space_time_pruning_enabled=args.dynamic_space_time_pruning,
    )
    loaded_world_version = engine.update_world(world)
    if loaded_world_version != world_version:
        raise ValueError(
            f"engine loaded world version {loaded_world_version}, expected {world_version}"
        )

    runs = []
    for index in range(args.repeats):
        seed = base_seed + index * args.seed_step
        request = deepcopy(raw_request)
        request["seed"] = seed
        request["_dynamic_world_version"] = world_version
        request["_trajectory_start_unix_ns"] = trajectory_start_unix_ns
        run_dir = output_dir / f"run-{index:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = run_dir / "trajectory.npz"
        trajectory_path.unlink(missing_ok=True)

        started = time.perf_counter()
        artifacts = engine.plan(request)
        elapsed_sec = time.perf_counter() - started
        artifacts.result_payload["standalone_space_time"] = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "run_index": index,
            "repeat_count": args.repeats,
            "seed_step": args.seed_step,
            "request_path": request_path.as_posix(),
            "request_sha256": request_sha256,
            "world_path": (
                None if args.world is None else args.world.expanduser().resolve().as_posix()
            ),
            "world_source": world_source,
            "socket_used": False,
            "ros_used": False,
        }
        _atomic_write_npz(
            trajectory_path,
            compressed=False,
            **artifacts.trajectory_arrays,
        )
        _atomic_write_json(run_dir / "result.json", artifacts.result_payload)
        _atomic_write_json(
            run_dir / "request.json",
            {
                "request": {key: value for key, value in request.items() if not key.startswith("_")},
                "world_version": world_version,
                "trajectory_start_unix_ns": trajectory_start_unix_ns,
            },
        )
        runs.append(
            _run_summary(
                artifacts.result_payload,
                index=index,
                seed=seed,
                elapsed_sec=elapsed_sec,
                run_dir=run_dir,
            )
        )

    summary = {
        "schema": "mpd_phase5_standalone_inference",
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "success",
        "timing_mode": args.timing_mode,
        "request_path": request_path.as_posix(),
        "request_sha256": request_sha256,
        "config_path": config_path.as_posix(),
        "world_path": (
            None if args.world is None else args.world.expanduser().resolve().as_posix()
        ),
        "world_source": world_source,
        "world_version": world_version,
        "dynamic_object_count": len(world.get("objects", [])),
        "trajectory_start_unix_ns": trajectory_start_unix_ns,
        "repeats": args.repeats,
        "seed": base_seed,
        "seed_step": args.seed_step,
        "engine": engine.health(),
        "runs": runs,
        "aggregate": {
            "elapsed_sec": _distribution([run["elapsed_sec"] for run in runs]),
            "duration_s": _distribution(
                [float(run["duration_s"]) for run in runs if run["duration_s"] is not None]
            ),
            "minimum_environment_clearance_m": _distribution(
                [
                    float(run["minimum_environment_clearance_m"])
                    for run in runs
                    if run["minimum_environment_clearance_m"] is not None
                ]
            ),
            "valid_candidates": _distribution(
                [
                    float(run["valid_candidates"])
                    for run in runs
                    if run["valid_candidates"] is not None
                ]
            ),
        },
        "created_unix_time": time.time(),
    }
    summary_path = output_dir / "summary.json"
    _atomic_write_json(summary_path, summary)
    return summary_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary_path = run(args)
        print(summary_path.as_posix())
        return 0
    except RuntimeContractError as error:
        failure = {
            "schema": "mpd_phase5_standalone_inference",
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": error.status,
            "error": {"type": type(error).__name__, "message": str(error)},
            "created_unix_time": time.time(),
        }
        _atomic_write_json(output_dir / "summary.json", failure)
        print(str(error), file=sys.stderr)
        return error.exit_code
    except Exception as error:
        failure = {
            "schema": "mpd_phase5_standalone_inference",
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "inference_error",
            "error": {"type": type(error).__name__, "message": str(error)},
            "created_unix_time": time.time(),
        }
        _atomic_write_json(output_dir / "summary.json", failure)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
