#!/usr/bin/env python3
"""Benchmark Marvin checkpoints on fixed ID, OOD, and shared-workspace tasks.

The benchmark has two deliberately separate phases:

* ``prepare`` samples collision-valid endpoints once and freezes request JSON.
* ``run`` reuses those exact requests for every model/checkpoint/seed.

This prevents endpoint-sampling differences from contaminating model comparisons.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from mpd.bimanual.runtime_contract import BimanualRequest
from mpd.bimanual.start_goal_sources import (
    StartGoalSamplingError,
    request_from_dataset,
    request_from_regions,
)
from scripts.inference.benchmark_marvin_collision_optimizations import (
    DEFAULT_CONFIG,
    ROOT,
    execute_case,
)
from scripts.inference.benchmark_marvin_inference_quality import (
    a5_defaults,
    select_checkpoints,
)


DEFAULT_SCENARIOS = ROOT / "scripts/inference/cfgs/marvin_region_model_benchmark.yaml"
DEFAULT_MODELS = ROOT / "scripts/inference/cfgs/marvin_reversed_model_registry.yaml"
DEFAULT_REGIONS = (
    ROOT
    / "scripts/inference/cfgs/start_goal_regions/"
    "EnvWarehouse-RobotMarvinBimanual-regions-matrix-generalization.yaml"
)
DIFFICULTIES = ("easy", "medium", "hard", "extreme")


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object: {path}")
    return value


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _immutable_json(path: Path, value) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing experiment differs; use a new output directory: {path}")
        return
    _atomic_json(path, value)


def _resolve_root_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scenarios(path: Path, names=None, difficulties=None) -> dict:
    payload = _read_yaml(path)
    if payload.get("schema") != "marvin_region_model_benchmark/v1":
        raise ValueError("scenario schema must be marvin_region_model_benchmark/v1")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("scenario catalog must contain scenarios")
    unknown = set(names or ()) - set(scenarios)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
    selected = {}
    for name, spec in scenarios.items():
        if names and name not in names:
            continue
        if difficulties and spec.get("difficulty") not in difficulties:
            continue
        if spec.get("difficulty") not in DIFFICULTIES:
            raise ValueError(f"{name}: invalid difficulty")
        if not isinstance(spec.get("support"), str) or not isinstance(spec.get("transition"), str):
            raise ValueError(f"{name}: support and transition must be strings")
        source = spec.get("source", "regions")
        if source not in ("dataset", "regions"):
            raise ValueError(f"{name}: source must be dataset or regions")
        if source == "regions":
            for endpoint in ("start", "goal"):
                mapping = spec.get(endpoint)
                if set(mapping or {}) != {"left", "right"}:
                    raise ValueError(f"{name}.{endpoint} must define left and right")
        selected[name] = spec
    if not selected:
        raise ValueError("Scenario filters selected nothing")
    return selected


def validate_scenario_regions(scenarios: dict, regions_path: Path) -> None:
    catalog = _read_yaml(regions_path)
    available = {"random"}
    for key in ("placement_regions", "placement_ood_regions", "named_random_regions"):
        available.update(catalog.get(key, {}))
    for scenario, spec in scenarios.items():
        if spec.get("source", "regions") != "regions":
            continue
        for endpoint in ("start", "goal"):
            for arm, name in spec[endpoint].items():
                if name not in available:
                    raise ValueError(f"{scenario}.{endpoint}.{arm}: unknown region {name!r}")
                if name != "random" and not name.startswith(arm + "_"):
                    raise ValueError(
                        f"{scenario}.{endpoint}.{arm}: region belongs to the other arm: {name!r}"
                    )


def _effective_regions(source: Path, destination: Path, attempts: int, timeout_s: float) -> None:
    config = _read_yaml(source)
    config["max_sampling_attempts"] = int(attempts)
    config["sampling_timeout_seconds"] = float(timeout_s)
    text = yaml.safe_dump(config, sort_keys=False)
    if destination.is_file() and destination.read_text(encoding="utf-8") != text:
        raise ValueError(f"Effective region config changed; use a new output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_text(text, encoding="utf-8")


def _request_metadata(request: dict, scenario: str, spec: dict, task_index: int) -> dict:
    goal_xyz = {}
    for arm in ("left", "right"):
        pose = request.get(f"{arm}_goal_pose", {})
        values = pose.get("pose_xyzw", [])
        goal_xyz[arm] = values[:3] if len(values) == 7 else None
    return {
        "scenario": scenario,
        "task_index": task_index,
        "difficulty": spec["difficulty"],
        "support": spec["support"],
        "transition": spec["transition"],
        "workspace_overlap": bool(spec.get("workspace_overlap", False)),
        "expected_sampling": spec.get("expected_sampling", "normal"),
        "source": spec.get("source", "regions"),
        "start_regions": spec.get("start"),
        "goal_regions": spec.get("goal"),
        "goal_xyz": goal_xyz,
    }


def prepare_tasks(
    output: Path,
    scenarios: dict,
    regions_path: Path,
    dataset_config: dict,
    tasks_per_scenario: int,
    max_task_attempts: int,
    endpoint_attempts: int,
    endpoint_timeout_s: float,
) -> dict:
    """Materialize frozen runtime requests, preserving partial progress."""
    task_root = output / "tasks"
    effective_regions = task_root / "effective-regions.yaml"
    _effective_regions(regions_path, effective_regions, endpoint_attempts, endpoint_timeout_s)
    settings = {
        "scenarios": scenarios,
        "regions_path": str(regions_path.resolve()),
        "dataset": dataset_config,
        "tasks_per_scenario": tasks_per_scenario,
        "max_task_attempts": max_task_attempts,
        "endpoint_attempts_per_call": endpoint_attempts,
        "endpoint_timeout_s_per_call": endpoint_timeout_s,
        "sampling": "system_entropy_for_regions; indexed_for_dataset",
    }
    _immutable_json(task_root / "settings.json", settings)
    report_path = task_root / "tasks.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {
        "schema": "marvin_region_benchmark_tasks/v1",
        "complete": False,
        "tasks": [],
        "scenarios": {},
    }
    if report.get("complete"):
        return report

    for scenario_name, spec in scenarios.items():
        state = report["scenarios"].setdefault(
            scenario_name, {"attempts": 0, "accepted": 0, "errors": {}},
        )
        existing = [item for item in report["tasks"] if item["scenario"] == scenario_name]
        state["accepted"] = len(existing)
        source = spec.get("source", "regions")
        while state["accepted"] < tasks_per_scenario and state["attempts"] < max_task_attempts:
            local_index = state["accepted"]
            state["attempts"] += 1
            request_id = f"region-benchmark-{scenario_name}-{local_index:03d}"
            try:
                if source == "dataset":
                    indices = spec.get("sample_indices", [])
                    if local_index >= len(indices):
                        raise IndexError("scenario has fewer sample_indices than requested tasks")
                    request = request_from_dataset(
                        dataset_config,
                        request_id=request_id,
                        seed=0,
                        sample_index=int(indices[local_index]),
                    )
                else:
                    overrides = {
                        f"{endpoint}.{arm}": spec[endpoint][arm]
                        for endpoint in ("start", "goal")
                        for arm in ("left", "right")
                    }
                    request = request_from_regions(
                        effective_regions,
                        request_id=request_id,
                        seed=0,
                        sample_index=-1,
                        region_overrides=overrides,
                    )
                source_info = request["scene"]["start_goal_source"]
                source_info.update({
                    "benchmark_scenario": scenario_name,
                    "difficulty": spec["difficulty"],
                    "support": spec["support"],
                    "transition": spec["transition"],
                    "workspace_overlap": bool(spec.get("workspace_overlap", False)),
                })
                BimanualRequest.from_dict(request)
                global_index = len(report["tasks"])
                request_path = task_root / "requests" / f"task-{global_index:04d}.json"
                _immutable_json(request_path, request)
                metadata = _request_metadata(request, scenario_name, spec, local_index)
                metadata["request_path"] = str(request_path.resolve())
                report["tasks"].append(metadata)
                state["accepted"] += 1
            except (StartGoalSamplingError, IndexError, ValueError) as error:
                key = type(error).__name__
                state["errors"][key] = state["errors"].get(key, 0) + 1
                state["last_error"] = str(error)
                if source == "dataset":
                    state["attempts"] = max_task_attempts
            state["shortfall"] = tasks_per_scenario - state["accepted"]
            _atomic_json(report_path, report)
            print(
                f"prepare {scenario_name}: {state['accepted']}/{tasks_per_scenario}, "
                f"attempts={state['attempts']}/{max_task_attempts}",
                flush=True,
            )
    report["complete"] = True
    _atomic_json(report_path, report)
    return report


def _resolve_selection(run_dir: Path, selection: dict) -> list[dict]:
    kind = selection.get("type")
    if kind == "explicit":
        result = []
        for item in selection.get("checkpoints", []):
            result.append({
                "checkpoint": str(item["file"]),
                "step": int(item["step"]),
                "val_loss": float(item["val_loss"]),
            })
        if not result:
            raise ValueError(f"No explicit checkpoints configured for {run_dir}")
        return result
    if kind == "top_validation":
        _, result = select_checkpoints(run_dir, int(selection.get("top_k", 2)))
        return result
    raise ValueError(f"Unknown checkpoint selection type: {kind}")


def load_model_cases(registry_path: Path, base_config: Path, variants, strict=False):
    registry = _read_yaml(registry_path)
    if registry.get("schema") != "marvin_bimanual_model_registry/v1":
        raise ValueError("model registry schema must be marvin_bimanual_model_registry/v1")
    catalog = registry.get("models", {})
    unknown = set(variants) - set(catalog)
    if unknown:
        raise ValueError(f"Unknown model variants: {sorted(unknown)}")
    base = _read_yaml(base_config)
    for key in ("start_goal_states_path", "start_goal_regions_path"):
        if base.get(key):
            path = Path(os.path.expandvars(str(base[key]))).expanduser()
            base[key] = str(path.resolve() if path.is_absolute() else (base_config.parent / path).resolve())
    cases, skipped = [], []
    for variant in variants:
        spec = catalog[variant]
        if not spec.get("enabled", False):
            skipped.append({"variant": variant, "reason": "disabled", "note": spec.get("note")})
            continue
        run_dir = _resolve_root_path(spec["run_dir"])
        try:
            train = _read_yaml(run_dir / "args.yaml")
            selected = _resolve_selection(run_dir, spec["selection"])
            actual_variant = train.get("bimanual_network_variant")
            if actual_variant != spec.get("network_variant", variant):
                raise ValueError(f"training variant is {actual_variant!r}")
            if train.get("dataset_subdir") != registry["dataset_subdir"]:
                raise ValueError("training dataset differs from registry dataset")
            for item in selected:
                checkpoint = run_dir / "checkpoints" / item["checkpoint"]
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                step_label = (
                    f"{item['step'] // 1000}k" if item["step"] >= 1000 and item["step"] % 1000 == 0
                    else str(item["step"])
                )
                config = a5_defaults(base)
                config["runtime"]["network_variant"] = variant
                config["model_dir_ddpm_bspline"] = str(run_dir)
                config["checkpoint"] = item["checkpoint"]
                config["dataset_subdir"] = registry["dataset_subdir"]
                config["dataset_file_merged"] = registry.get("dataset_file_merged", "dataset_merged.hdf5")
                stat = checkpoint.stat()
                cases.append({
                    "id": f"{variant}-{step_label}",
                    "variant": variant,
                    "step": item["step"],
                    "val_loss": item["val_loss"],
                    "run_dir": str(run_dir),
                    "checkpoint": str(checkpoint),
                    "fingerprint": {
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": _sha256(checkpoint),
                    },
                    "config": config,
                })
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            if strict:
                raise
            skipped.append({"variant": variant, "reason": type(error).__name__, "detail": str(error)})
    return registry, cases, skipped


def materialize_experiment(output: Path, task_report: dict, cases: list, skipped: list, args) -> dict:
    config_root = output / "configs"
    case_manifest = []
    for case in cases:
        config = deepcopy(case["config"])
        config["n_trajectory_samples"] = int(args.candidates_per_batch)
        config["runtime_top_k_valid_trajectories"] = min(
            int(config.get("runtime_top_k_valid_trajectories", args.candidates_per_batch)),
            int(args.candidates_per_batch),
        )
        config_path = config_root / f"{case['id']}.yaml"
        text = yaml.safe_dump(config, sort_keys=False)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if config_path.is_file() and config_path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Model config changed; use a new output directory: {config_path}")
        if not config_path.exists():
            config_path.write_text(text, encoding="utf-8")
        public = {key: value for key, value in case.items() if key != "config"}
        public["config_path"] = str(config_path.resolve())
        case_manifest.append(public)
    manifest = {
        "schema": "marvin_region_model_experiment/v1",
        "tasks_file": str((output / "tasks/tasks.json").resolve()),
        "task_count": len(task_report["tasks"]),
        "sampling_shortfalls": {
            name: state.get("shortfall", 0) for name, state in task_report["scenarios"].items()
        },
        "models": case_manifest,
        "skipped_models": skipped,
        "seeds": list(args.seeds),
        "candidate_batches": int(args.candidate_batches),
        "candidates_per_batch": int(args.candidates_per_batch),
        "early_stop_on_success": True,
        "device": args.device,
        "timeout_s": float(args.timeout_s),
    }
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        _atomic_json(manifest_path, manifest)
        return manifest

    # A completed benchmark can be extended with additional checkpoints while
    # preserving the frozen tasks and inference budget.  Existing checkpoint
    # definitions remain immutable: an ID collision with different content is
    # rejected instead of silently mixing experiments.
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    extensible = {"models", "skipped_models"}
    for key, value in manifest.items():
        if key not in extensible and existing.get(key) != value:
            raise ValueError(
                f"Existing experiment differs at {key!r}; use a new output directory: {manifest_path}"
            )
    merged_models = list(existing.get("models", []))
    by_id = {model["id"]: model for model in merged_models}
    for model in manifest["models"]:
        previous = by_id.get(model["id"])
        if previous is not None and previous != model:
            raise ValueError(
                f"Checkpoint definition changed for {model['id']}; use a new output directory"
            )
        if previous is None:
            merged_models.append(model)
            by_id[model["id"]] = model
    merged_skipped = list(existing.get("skipped_models", []))
    for item in manifest["skipped_models"]:
        if item not in merged_skipped:
            merged_skipped.append(item)
    existing["models"] = merged_models
    existing["skipped_models"] = merged_skipped
    _atomic_json(manifest_path, existing)
    return existing


def resume_counts(output: Path, manifest: dict, task_report: dict) -> dict:
    """Count reusable reports and the maximum number of subprocesses still needed."""
    existing, pending = 0, 0
    for model in manifest["models"]:
        for task in task_report["tasks"]:
            for seed in manifest["seeds"]:
                group = (
                    output / "runs" / model["id"] / task["scenario"]
                    / f"task-{task['task_index']:03d}" / f"seed-{seed}"
                )
                for batch in range(manifest["candidate_batches"]):
                    report = group / f"batch-{batch:02d}/benchmark-result.json"
                    if report.is_file():
                        existing += 1
                        if json.loads(report.read_text(encoding="utf-8")).get("status") == "success":
                            break
                    else:
                        pending += 1
    return {"existing_batch_reports": existing, "pending_batch_runs_max": pending}


def run_experiment(output: Path, manifest: dict, task_report: dict, args) -> None:
    for model in manifest["models"]:
        config_path = Path(model["config_path"])
        for task in task_report["tasks"]:
            original = json.loads(Path(task["request_path"]).read_text(encoding="utf-8"))
            for seed in manifest["seeds"]:
                group = (
                    output / "runs" / model["id"] / task["scenario"]
                    / f"task-{task['task_index']:03d}" / f"seed-{seed}"
                )
                for batch in range(manifest["candidate_batches"]):
                    artifact = group / f"batch-{batch:02d}"
                    report_path = artifact / "benchmark-result.json"
                    if report_path.is_file():
                        row = json.loads(report_path.read_text(encoding="utf-8"))
                        reused = True
                    else:
                        reused = False
                        request = deepcopy(original)
                        request["seed"] = int(seed) + batch
                        request_path = artifact / "input-request.json"
                        _immutable_json(request_path, request)
                        run_args = SimpleNamespace(
                            request=request_path,
                            start_goal_source="dataset",
                            sample_index=0,
                            seed=int(seed) + batch,
                            device=args.device,
                            timeout_s=args.timeout_s,
                            gpu_poll_interval=args.gpu_poll_interval,
                        )
                        row = execute_case(config_path, artifact, run_args)
                        row.update({
                            "model": model["id"],
                            "variant": model["variant"],
                            "checkpoint_step": model["step"],
                            "scenario": task["scenario"],
                            "task_index": task["task_index"],
                            "difficulty": task["difficulty"],
                            "support": task["support"],
                            "transition": task["transition"],
                            "workspace_overlap": task["workspace_overlap"],
                            "seed": seed,
                            "batch": batch,
                            "max_batches": manifest["candidate_batches"],
                        })
                        _immutable_json(report_path, row)
                    if not reused:
                        print(
                            f"{model['id']} {task['scenario']} task={task['task_index']} "
                            f"seed={seed} batch={batch}: {row['status']}",
                            flush=True,
                        )
                    summarize_experiment(output, manifest, task_report)
                    if row["status"] == "success":
                        break


def _collect_rows(output: Path, manifest: dict, task_report: dict) -> list[dict]:
    rows = []
    for model in manifest["models"]:
        for task in task_report["tasks"]:
            for seed in manifest["seeds"]:
                group = (
                    output / "runs" / model["id"] / task["scenario"]
                    / f"task-{task['task_index']:03d}" / f"seed-{seed}"
                )
                group_rows = []
                for batch in range(manifest["candidate_batches"]):
                    path = group / f"batch-{batch:02d}/benchmark-result.json"
                    if path.is_file():
                        row = json.loads(path.read_text(encoding="utf-8"))
                        rows.append(row)
                        group_rows.append(row)
                        if row["status"] == "success":
                            break
                    elif any(row["status"] == "success" for row in group_rows):
                        break
    return rows


def _group_trials(rows: list[dict], max_batches: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["scenario"], row["task_index"], row["seed"])].append(row)
    trials = []
    for (model, scenario, task_index, seed), items in grouped.items():
        items.sort(key=lambda item: item["batch"])
        success = any(item["status"] == "success" for item in items)
        complete = success or len(items) >= max_batches
        first = items[0]
        trials.append({
            "model": model,
            "variant": first["variant"],
            "checkpoint_step": first["checkpoint_step"],
            "scenario": scenario,
            "task_index": task_index,
            "seed": seed,
            "difficulty": first["difficulty"],
            "support": first["support"],
            "transition": first["transition"],
            "workspace_overlap": first["workspace_overlap"],
            "complete": complete,
            "success": success,
            "batches_attempted": len(items),
            "wall_seconds": sum(float(item.get("wall_seconds", 0.0)) for item in items),
            "peak_device_used_mib": max(
                (item.get("gpu_memory") or {}).get("peak_device_used_mib", 0) for item in items
            ),
            "valid_candidates": sum((item.get("candidates") or {}).get("valid", 0) for item in items),
            "statuses": dict(Counter(item["status"] for item in items)),
        })
    return trials


def _aggregate(trials: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for trial in trials:
        groups[tuple(trial[field] for field in fields)].append(trial)
    result = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        complete = [item for item in items if item["complete"]]
        successes = sum(item["success"] for item in complete)
        evaluable = [
            item for item in complete
            if item["success"] or set(item["statuses"]) == {"no_valid_trajectory"}
        ]
        planning_successes = sum(item["success"] for item in evaluable)
        row = dict(zip(fields, key))
        row.update({
            "trials_seen": len(items),
            "trials_complete": len(complete),
            "successes": successes,
            "planning_evaluable_trials": len(evaluable),
            "infrastructure_or_contract_trials": len(complete) - len(evaluable),
            "success_rate": planning_successes / len(evaluable) if evaluable else None,
            "planning_success_rate": planning_successes / len(evaluable) if evaluable else None,
            "end_to_end_success_rate": successes / len(complete) if complete else None,
            "mean_wall_seconds": float(np.mean([item["wall_seconds"] for item in complete])) if complete else None,
            "max_peak_device_used_mib": max((item["peak_device_used_mib"] for item in items), default=0),
            "mean_valid_candidates": float(np.mean([item["valid_candidates"] for item in complete])) if complete else None,
            "status_counts": dict(Counter(
                status for item in items for status, count in item["statuses"].items() for _ in range(count)
            )),
        })
        result.append(row)
    return result


def summarize_experiment(output: Path, manifest: dict, task_report: dict) -> dict:
    rows = _collect_rows(output, manifest, task_report)
    trials = _group_trials(rows, manifest["candidate_batches"])
    summary = {
        "schema": "marvin_region_model_summary/v1",
        "sampling": task_report["scenarios"],
        "raw_batch_runs": len(rows),
        "trial_groups": len(trials),
        "by_model": _aggregate(trials, ("model",)),
        "by_model_difficulty": _aggregate(trials, ("model", "difficulty")),
        "by_model_support": _aggregate(trials, ("model", "support")),
        "by_model_transition": _aggregate(trials, ("model", "transition")),
        "by_model_scenario": _aggregate(trials, ("model", "scenario")),
        "shared_workspace": _aggregate(
            [trial for trial in trials if trial["workspace_overlap"]], ("model", "scenario")
        ),
        "trials": trials,
    }
    _atomic_json(output / "summary.json", summary)
    csv_rows = summary["by_model_scenario"]
    csv_path = output / "summary-by-model-scenario.csv"
    if csv_rows:
        fields = [field for field in csv_rows[0] if field != "status_counts"] + ["status_counts"]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow({**row, "status_counts": json.dumps(row["status_counts"], sort_keys=True)})
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "all", "summarize"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--model-registry", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--difficulties", nargs="+", choices=DIFFICULTIES)
    parser.add_argument("--variants", nargs="+", choices=tuple("ABCD"), default=list("ABCD"))
    parser.add_argument("--strict-models", action="store_true")
    parser.add_argument("--tasks-per-scenario", type=int, default=1)
    parser.add_argument("--max-task-attempts", type=int, default=30)
    parser.add_argument("--endpoint-attempts", type=int, default=30)
    parser.add_argument("--endpoint-timeout-s", type=float, default=15.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[12345])
    parser.add_argument("--candidate-batches", type=int, default=1)
    parser.add_argument("--candidates-per-batch", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--gpu-poll-interval", type=float, default=0.2)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    positive = (
        args.tasks_per_scenario,
        args.max_task_attempts,
        args.endpoint_attempts,
        args.endpoint_timeout_s,
        args.candidate_batches,
        args.candidates_per_batch,
        args.timeout_s,
        args.gpu_poll_interval,
    )
    if any(value <= 0 for value in positive):
        parser.error("counts, attempts, polling intervals, and timeouts must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(
        seed < 0 or seed + args.candidate_batches >= 2**32 for seed in args.seeds
    ):
        parser.error("seeds must be unique and leave room for every candidate batch")
    output = args.output_dir.expanduser().resolve()
    scenarios = load_scenarios(args.scenario_config.resolve(), args.scenarios, args.difficulties)
    validate_scenario_regions(scenarios, args.regions.resolve())
    registry, cases, skipped = load_model_cases(
        args.model_registry.resolve(), args.config.resolve(), args.variants, args.strict_models
    )
    if args.stage in ("run", "all") and not cases:
        parser.error("No enabled and available model checkpoints selected")
    dataset_config = {
        "dataset_subdir": registry["dataset_subdir"],
        "dataset_file_merged": registry.get("dataset_file_merged", "dataset_merged.hdf5"),
    }
    task_path = output / "tasks/tasks.json"
    if args.stage in ("prepare", "all"):
        task_report = prepare_tasks(
            output,
            scenarios,
            args.regions.resolve(),
            dataset_config,
            args.tasks_per_scenario,
            args.max_task_attempts,
            args.endpoint_attempts,
            args.endpoint_timeout_s,
        )
    else:
        if not task_path.is_file():
            parser.error(f"prepare tasks first: {task_path}")
        task_report = json.loads(task_path.read_text(encoding="utf-8"))
    manifest = materialize_experiment(output, task_report, cases, skipped, args)
    resume = resume_counts(output, manifest, task_report)
    print(json.dumps({
        "tasks": len(task_report["tasks"]),
        "models": [item["id"] for item in manifest["models"]],
        "skipped_models": skipped,
        "maximum_trial_groups": (
            len(task_report["tasks"]) * len(manifest["models"]) * len(args.seeds)
        ),
        "maximum_subprocess_runs": (
            len(task_report["tasks"]) * len(manifest["models"])
            * len(args.seeds) * args.candidate_batches
        ),
        **resume,
        "sampling_shortfalls": manifest["sampling_shortfalls"],
    }, indent=2), flush=True)
    if args.stage == "prepare":
        return 0
    if args.stage in ("run", "all"):
        if args.device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                parser.error("CUDA unavailable in this interpreter; activate mpd-splines-public")
        run_experiment(output, manifest, task_report, args)
    summarize_experiment(output, manifest, task_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
