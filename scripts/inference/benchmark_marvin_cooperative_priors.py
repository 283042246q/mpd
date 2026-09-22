#!/usr/bin/env python3
"""Benchmark cooperative priors and closure projection on frozen region tasks.

The benchmark is paired by construction: endpoint requests are sampled once,
written to disk, and reused by all four prior/projection combinations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import yaml

from mpd.bimanual.cooperative_sources import request_from_regions
from mpd.bimanual.runtime_contract import BimanualRequest
from scripts.inference.benchmark_marvin_collision_optimizations import execute_case


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "scripts/inference/cfgs/" "config_EnvWarehouse-RobotMarvinBimanual-cooperative-independent-prior.yaml"
)
DEFAULT_SCENARIOS = ROOT / "scripts/inference/cfgs/marvin_cooperative_prior_benchmark.yaml"
DEFAULT_REGIONS = (
    ROOT / "scripts/inference/cfgs/start_goal_regions/" "EnvWarehouse-RobotMarvinBimanual-cooperative-regions.yaml"
)
DIFFICULTIES = ("easy", "medium", "hard")
CASES = {
    "direct_project__projection_on": {
        "prior_mode": "direct_project",
        "closed_chain_projection": True,
    },
    "direct_project__projection_off": {
        "prior_mode": "direct_project",
        "closed_chain_projection": False,
    },
    "reference_residual__projection_on": {
        "prior_mode": "reference_residual",
        "closed_chain_projection": True,
    },
    "reference_residual__projection_off": {
        "prior_mode": "reference_residual",
        "closed_chain_projection": False,
    },
}


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object: {path}")
    return value


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _immutable_json(path: Path, value) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError(f"Existing benchmark definition differs; use a new output directory: {path}")
        return
    _atomic_json(path, value)


def _immutable_text(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Existing benchmark definition differs; use a new output directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_config_path(config_path: Path, value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def load_scenarios(
    scenario_path: Path,
    regions_path: Path,
    *,
    names=None,
    difficulties=None,
) -> tuple[dict, dict]:
    payload = _read_yaml(scenario_path)
    if payload.get("schema") != "marvin_cooperative_prior_benchmark/v1":
        raise ValueError("scenario schema must be marvin_cooperative_prior_benchmark/v1")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("cooperative benchmark must define scenarios")
    unknown = set(names or ()) - set(scenarios)
    if unknown:
        raise ValueError(f"Unknown cooperative scenarios: {sorted(unknown)}")

    regions = _read_yaml(regions_path)
    if regions.get("schema") != "marvin_bimanual_cooperative_regions/v1":
        raise ValueError("regions file is not a cooperative region catalog")
    region_names = set(regions.get("object_regions", {}))
    matrix_pairs = {(item.get("start"), item.get("goal")) for item in regions.get("sampling_matrix", [])}
    selected = {}
    for name, spec in scenarios.items():
        if names and name not in names:
            continue
        difficulty = spec.get("difficulty")
        if difficulties and difficulty not in difficulties:
            continue
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"{name}: difficulty must be one of {DIFFICULTIES}")
        pair = (spec.get("start_region"), spec.get("goal_region"))
        if not set(pair) <= region_names:
            raise ValueError(f"{name}: references an unknown object region")
        if pair not in matrix_pairs:
            raise ValueError(f"{name}: region pair is absent from the production matrix")
        selected[name] = spec
    if not selected:
        raise ValueError("scenario filters selected nothing")
    return payload, selected


def build_cases(base: dict, selected=None, n_trajectory_samples=None) -> dict:
    names = list(selected or CASES)
    unknown = set(names) - set(CASES)
    if unknown:
        raise ValueError(f"Unknown cooperative benchmark cases: {sorted(unknown)}")
    cases = {}
    for name in names:
        config = deepcopy(base)
        options = config.setdefault("cooperative_inference", {})
        options.update(CASES[name])
        if n_trajectory_samples is not None:
            config["n_trajectory_samples"] = int(n_trajectory_samples)
            config["runtime_top_k_valid_trajectories"] = min(
                int(config.get("runtime_top_k_valid_trajectories", n_trajectory_samples)),
                int(n_trajectory_samples),
            )
        cases[name] = config
    return cases


def _effective_region_text(regions: dict, spec: dict) -> str:
    effective = deepcopy(regions)
    effective["sampling_matrix"] = [
        {
            "start": spec["start_region"],
            "goal": spec["goal_region"],
            "weight": 1.0,
        }
    ]
    return yaml.safe_dump(effective, sort_keys=False)


def _task_metrics(request: dict) -> dict:
    scene = request["scene"]
    start = np.asarray(scene["object_start_state_xyz_yaw"], dtype=float)
    goal = np.asarray(scene["object_goal_state_xyz_yaw"], dtype=float)
    delta = goal - start
    yaw = abs(math.atan2(math.sin(delta[3]), math.cos(delta[3])))
    return {
        "object_translation_m": float(np.linalg.norm(delta[:3])),
        "object_longitudinal_m": float(abs(delta[0])),
        "object_lateral_m": float(abs(delta[1])),
        "object_vertical_m": float(abs(delta[2])),
        "object_yaw_rad": float(yaw),
    }


def prepare_tasks(
    output: Path,
    scenarios: dict,
    regions_path: Path,
    generation_config: Path,
    *,
    tasks_per_scenario: int,
    max_endpoint_attempts: int,
    endpoint_seed: int,
) -> dict:
    task_root = output / "tasks"
    regions = _read_yaml(regions_path)
    settings = {
        "regions_path": str(regions_path),
        "regions_sha256": _sha256(regions_path),
        "generation_config": str(generation_config),
        "generation_config_sha256": _sha256(generation_config),
        "scenarios": scenarios,
        "tasks_per_scenario": int(tasks_per_scenario),
        "max_endpoint_attempts": int(max_endpoint_attempts),
        "endpoint_seed": int(endpoint_seed),
        "paired_design": True,
    }
    _immutable_json(task_root / "settings.json", settings)
    report_path = task_root / "tasks.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("complete"):
            return report
    else:
        report = {
            "schema": "marvin_cooperative_prior_tasks/v1",
            "complete": False,
            "tasks": [],
            "scenarios": {},
        }

    for scenario_index, (scenario_name, spec) in enumerate(scenarios.items()):
        effective_path = task_root / "effective-regions" / f"{scenario_name}.yaml"
        _immutable_text(effective_path, _effective_region_text(regions, spec))
        state = report["scenarios"].setdefault(
            scenario_name,
            {"attempts": 0, "accepted": 0, "errors": {}},
        )
        existing = [item for item in report["tasks"] if item["scenario"] == scenario_name]
        state["accepted"] = len(existing)
        limit = int(tasks_per_scenario) * int(max_endpoint_attempts)
        while state["accepted"] < tasks_per_scenario and state["attempts"] < limit:
            task_index = state["accepted"]
            attempt = state["attempts"]
            state["attempts"] += 1
            seed = int(endpoint_seed) + scenario_index * 100_000 + attempt
            request_id = f"coop-prior-{scenario_name}-{task_index:03d}"
            try:
                request = request_from_regions(
                    effective_path,
                    generator_config_path=generation_config,
                    request_id=request_id,
                    seed=seed,
                )
                BimanualRequest.from_dict(request)
                request["scene"]["start_goal_source"].update(
                    benchmark_scenario=scenario_name,
                    difficulty=spec["difficulty"],
                    transition=spec["transition"],
                    endpoint_seed=seed,
                )
                request_path = task_root / "requests" / scenario_name / f"task-{task_index:03d}.json"
                _immutable_json(request_path, request)
                report["tasks"].append(
                    {
                        "scenario": scenario_name,
                        "difficulty": spec["difficulty"],
                        "transition": spec["transition"],
                        "description": spec["description"],
                        "start_region": spec["start_region"],
                        "goal_region": spec["goal_region"],
                        "task_index": task_index,
                        "endpoint_seed": seed,
                        "request_path": str(request_path.resolve()),
                        **_task_metrics(request),
                    }
                )
                state["accepted"] += 1
            except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
                key = type(error).__name__
                state["errors"][key] = state["errors"].get(key, 0) + 1
                state["last_error"] = str(error)
            state["shortfall"] = tasks_per_scenario - state["accepted"]
            _atomic_json(report_path, report)
        print(
            f"prepare {scenario_name}: {state['accepted']}/{tasks_per_scenario}, "
            f"attempts={state['attempts']}/{limit}",
            flush=True,
        )

    report["complete"] = True
    report["sampling_shortfalls"] = {name: state.get("shortfall", 0) for name, state in report["scenarios"].items()}
    _atomic_json(report_path, report)
    return report


def _resolve_model_run(config: dict) -> tuple[Path, Path, dict, str]:
    configured = Path(os.path.expandvars(str(config["model_dir_ddpm_bspline"]))).expanduser()
    configured = configured.resolve()
    if (configured / "args.yaml").is_file():
        run = configured
    else:
        runs = [item.parent for item in configured.glob("*/args.yaml")]
        if len(runs) != 1:
            raise ValueError(f"Model directory is ambiguous: {configured}")
        run = runs[0]
    train = _read_yaml(run / "args.yaml")
    configured_checkpoint = config.get("checkpoint")
    if configured_checkpoint:
        checkpoint_name = str(configured_checkpoint)
        if Path(checkpoint_name).name != checkpoint_name:
            raise ValueError("checkpoint must be a filename under model_dir/checkpoints")
        selection = "configured"
    else:
        prefix = "ema_" if train.get("use_ema", False) else ""
        expression = re.compile(rf"{re.escape(prefix)}model__iter_(\d+)\.pth")
        numbered = []
        for candidate in (run / "checkpoints").glob(f"{prefix}model__iter_*.pth"):
            match = expression.fullmatch(candidate.name)
            if match:
                numbered.append((int(match.group(1)), candidate.name))
        if numbered:
            checkpoint_name = max(numbered)[1]
            selection = "latest_numbered"
        else:
            checkpoint_name = prefix + "model_current.pth"
            selection = "current_fallback"
    checkpoint = run / "checkpoints" / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return run, checkpoint, train, selection


def _verify_model_fingerprint(manifest: dict) -> None:
    expected = manifest["model"]
    checkpoint = Path(expected["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual = _sha256(checkpoint)
    if actual != expected["checkpoint_sha256"]:
        raise RuntimeError(
            "Pinned diffusion checkpoint changed after benchmark preparation: "
            f"{checkpoint}. Prepare a new output directory."
        )


def materialize_experiment(
    output: Path,
    config_path: Path,
    base: dict,
    task_report: dict,
    scenario_path: Path,
    regions_path: Path,
    args,
) -> dict:
    base = deepcopy(base)
    for key in (
        "cooperative_generation_config",
        "cooperative_regions_path",
        "cooperative_seed_file_path",
    ):
        if base.get(key):
            base[key] = str(_resolve_config_path(config_path, base[key]))
    run, checkpoint, train, checkpoint_selection = _resolve_model_run(base)
    # Every generated case names one immutable, numbered checkpoint. This
    # prevents an active training job from changing model_current between the
    # paired trials.
    base["model_dir_ddpm_bspline"] = str(run)
    base["checkpoint"] = checkpoint.name
    cases = build_cases(base, args.cases, args.n_trajectory_samples)
    case_entries = []
    for name, config in cases.items():
        path = output / "configs" / f"{name}.yaml"
        _immutable_text(path, yaml.safe_dump(config, sort_keys=False))
        case_entries.append(
            {
                "name": name,
                **CASES[name],
                "config_path": str(path.resolve()),
                "config_sha256": _sha256(path),
            }
        )
    stat = checkpoint.stat()
    manifest = {
        "schema": "marvin_cooperative_prior_experiment/v1",
        "base_config": str(config_path),
        "base_config_sha256": _sha256(config_path),
        "scenario_config": str(scenario_path),
        "scenario_config_sha256": _sha256(scenario_path),
        "regions": str(regions_path),
        "regions_sha256": _sha256(regions_path),
        "tasks_file": str((output / "tasks/tasks.json").resolve()),
        "task_count": len(task_report["tasks"]),
        "sampling_shortfalls": task_report.get("sampling_shortfalls", {}),
        "cases": case_entries,
        "seeds": list(args.seeds),
        "device": args.device,
        "backend": args.backend,
        "timeout_s": float(args.timeout_s),
        "gpu_poll_interval": float(args.gpu_poll_interval),
        "n_trajectory_samples": int(args.n_trajectory_samples or base["n_trajectory_samples"]),
        "model": {
            "run_dir": str(run),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_size": stat.st_size,
            "checkpoint_mtime_ns": stat.st_mtime_ns,
            "checkpoint_selection": checkpoint_selection,
            "network_variant": train.get("bimanual_network_variant"),
            "dataset_subdir": train.get("dataset_subdir"),
            "use_ema": bool(train.get("use_ema", False)),
        },
        "paired_design": {
            "same_request_across_cases": True,
            "same_inference_seed_across_cases": True,
        },
    }
    _immutable_json(output / "manifest.json", manifest)
    return manifest


def _failure_reason(row: dict) -> str:
    error = row.get("result_error") or {}
    if error.get("type"):
        return str(error["type"])
    return str(row.get("status", "unknown"))


def run_experiment(output: Path, manifest: dict, tasks: dict) -> None:
    _verify_model_fingerprint(manifest)
    for task in tasks["tasks"]:
        original = json.loads(Path(task["request_path"]).read_text(encoding="utf-8"))
        for seed in manifest["seeds"]:
            request = deepcopy(original)
            request["seed"] = int(seed)
            request["request_id"] = f"{original['request_id']}-seed-{seed}"
            for case in manifest["cases"]:
                artifact = (
                    output
                    / "runs"
                    / case["name"]
                    / task["difficulty"]
                    / task["scenario"]
                    / f"task-{task['task_index']:03d}"
                    / f"seed-{seed}"
                )
                report_path = artifact / "benchmark-result.json"
                if report_path.is_file():
                    row = json.loads(report_path.read_text(encoding="utf-8"))
                else:
                    request_path = artifact / "input-request.json"
                    _immutable_json(request_path, request)
                    run_args = SimpleNamespace(
                        request=request_path,
                        start_goal_source="regions",
                        sample_index=0,
                        seed=int(seed),
                        device=manifest["device"],
                        timeout_s=manifest["timeout_s"],
                        gpu_poll_interval=manifest["gpu_poll_interval"],
                        backend=manifest["backend"],
                    )
                    row = execute_case(Path(case["config_path"]), artifact, run_args)
                    row.update(
                        {
                            "case": case["name"],
                            "prior_mode": case["prior_mode"],
                            "closed_chain_projection": case["closed_chain_projection"],
                            "scenario": task["scenario"],
                            "difficulty": task["difficulty"],
                            "transition": task["transition"],
                            "task_index": task["task_index"],
                            "endpoint_seed": task["endpoint_seed"],
                            "inference_seed": int(seed),
                            "task_metrics": {key: task[key] for key in task if key.startswith("object_")},
                        }
                    )
                    row["failure_reason"] = _failure_reason(row)
                    _atomic_json(report_path, row)
                print(
                    f"{task['difficulty']} {task['scenario']} task={task['task_index']} "
                    f"seed={seed} {case['name']}: {row['status']}",
                    flush=True,
                )
                summarize_experiment(output, manifest)


def _collect_rows(output: Path) -> list[dict]:
    if not (output / "runs").is_dir():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "runs").glob("**/benchmark-result.json"))
    ]


def _get_number(row: dict, *path):
    value = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _mean(rows, *path):
    values = [value for row in rows if (value := _get_number(row, *path)) is not None]
    return float(np.mean(values)) if values else None


def _aggregate(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, items in sorted(groups.items()):
        wall = [float(item["wall_seconds"]) for item in items]
        successes = [item for item in items if item["status"] == "success"]
        entry = dict(zip(fields, key))
        entry.update(
            {
                "runs": len(items),
                "successes": len(successes),
                "success_rate": len(successes) / len(items),
                "status_counts": dict(Counter(item["status"] for item in items)),
                "failure_reason_counts": dict(
                    Counter(
                        item.get("failure_reason", _failure_reason(item))
                        for item in items
                        if item["status"] != "success"
                    )
                ),
                "mean_wall_s": float(np.mean(wall)),
                "median_wall_s": float(np.median(wall)),
                "p95_wall_s": float(np.percentile(wall, 95)),
                "mean_inference_s": _mean(items, "timing", "inference_total_s"),
                "mean_generator_s": _mean(items, "timing", "generator_s"),
                "mean_guide_s": _mean(items, "timing", "guide_s"),
                "mean_dense_validation_s": _mean(items, "timing", "dense_validation_s"),
                "mean_reference_build_s": _mean(items, "cooperative_inference", "reference_build_s"),
                "mean_projection_s": _mean(items, "cooperative_inference", "projection_s"),
                "mean_valid_candidates": _mean(items, "candidates", "valid"),
                "mean_closure_translation_m": _mean(
                    successes,
                    "validation",
                    "max_closure_translation_error_m",
                ),
                "mean_closure_rotation_rad": _mean(
                    successes,
                    "validation",
                    "max_closure_rotation_error_rad",
                ),
            }
        )
        result.append(entry)
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                    for key, value in row.items()
                }
            )


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Marvin cooperative prior benchmark",
        "",
        "| difficulty | case | success | mean wall (s) | mean inference (s) | valid candidates |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["by_difficulty_case"]:
        success = f"{row['successes']}/{row['runs']} ({100 * row['success_rate']:.1f}%)"
        inference = f"{row['mean_inference_s']:.3f}" if row["mean_inference_s"] is not None else "n/a"
        valid = f"{row['mean_valid_candidates']:.2f}" if row["mean_valid_candidates"] is not None else "n/a"
        lines.append(
            f"| {row['difficulty']} | {row['case']} | {success} | "
            f"{row['mean_wall_s']:.3f} | {inference} | {valid} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_experiment(output: Path, manifest: dict) -> dict:
    rows = _collect_rows(output)
    summary = {
        "schema": "marvin_cooperative_prior_summary/v1",
        "expected_runs": manifest["task_count"] * len(manifest["seeds"]) * len(manifest["cases"]),
        "completed_runs": len(rows),
        "by_case": _aggregate(rows, ("case",)),
        "by_difficulty_case": _aggregate(rows, ("difficulty", "case")),
        "by_scenario_case": _aggregate(rows, ("scenario", "case")),
        "rows": rows,
    }
    _atomic_json(output / "summary.json", summary)
    _write_csv(output / "summary-by-difficulty-case.csv", summary["by_difficulty_case"])
    _write_csv(output / "summary-by-scenario-case.csv", summary["by_scenario_case"])
    _write_markdown(output / "summary.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "all", "summarize"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--difficulties", nargs="+", choices=DIFFICULTIES)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--tasks-per-scenario", type=int, default=1)
    parser.add_argument("--max-endpoint-attempts", type=int, default=20)
    parser.add_argument("--endpoint-seed", type=int, default=47000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[12345])
    parser.add_argument("--n-trajectory-samples", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--gpu-poll-interval", type=float, default=0.2)
    parser.add_argument("--backend", choices=("mpd", "contract_stub"), default="mpd")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.tasks_per_scenario < 1
        or args.max_endpoint_attempts < 1
        or args.timeout_s <= 0
        or args.gpu_poll_interval <= 0
        or (args.n_trajectory_samples is not None and args.n_trajectory_samples < 1)
    ):
        parser.error("task counts, attempts, candidates, timeout and polling must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("inference seeds must be unique uint32 values")
    output = args.output_dir.expanduser().resolve()

    if args.stage == "summarize":
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        summarize_experiment(output, manifest)
        print(output / "summary.json")
        return 0

    config_path = args.config.expanduser().resolve()
    scenario_path = args.scenario_config.expanduser().resolve()
    regions_path = args.regions.expanduser().resolve()
    base = _read_yaml(config_path)
    _, scenarios = load_scenarios(
        scenario_path,
        regions_path,
        names=args.scenarios,
        difficulties=args.difficulties,
    )
    generation_config = _resolve_config_path(config_path, base["cooperative_generation_config"])

    if args.stage in ("prepare", "all"):
        tasks = prepare_tasks(
            output,
            scenarios,
            regions_path,
            generation_config,
            tasks_per_scenario=args.tasks_per_scenario,
            max_endpoint_attempts=args.max_endpoint_attempts,
            endpoint_seed=args.endpoint_seed,
        )
        manifest = materialize_experiment(
            output,
            config_path,
            base,
            tasks,
            scenario_path,
            regions_path,
            args,
        )
    else:
        tasks = json.loads((output / "tasks/tasks.json").read_text(encoding="utf-8"))
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    if args.stage == "prepare":
        print(output / "manifest.json")
        return 0
    if not tasks.get("complete"):
        raise RuntimeError("task preparation is incomplete")
    if manifest["backend"] == "mpd" and manifest["device"].startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; use --stage prepare or --backend contract_stub")
    run_experiment(output, manifest, tasks)
    summarize_experiment(output, manifest)
    print(output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
