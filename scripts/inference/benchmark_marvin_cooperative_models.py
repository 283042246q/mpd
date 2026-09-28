#!/usr/bin/env python3
"""Compare nine independent checkpoints on identical cooperative region tasks.

Prepare freezes paired-IK endpoints; run reuses each request and diffusion seed
for every checkpoint. All models use the same reference-residual prior, closure
projection, candidate budget, cost weights and cooperative validator.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace

import yaml

from mpd.bimanual.checkpoint_contract import validate_checkpoint_args
from scripts.inference.benchmark_marvin_collision_optimizations import execute_case
from scripts.inference.benchmark_marvin_cooperative_priors import (
    DEFAULT_CONFIG,
    DEFAULT_REGIONS,
    DEFAULT_SCENARIOS,
    _aggregate,
    _atomic_json,
    _immutable_json,
    _immutable_text,
    _read_yaml,
    _resolve_config_path,
    _sha256,
    _write_csv,
    load_scenarios,
    prepare_tasks,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "scripts/inference/cfgs/marvin_reversed_model_registry.yaml"
DEFAULT_MODELS = (
    "D-600k",
    "D-665k",
    "D-305k",
    "A-835k",
    "A-660k",
    "B-955k",
    "C-835k",
    "C-715k",
    "B-715k",
)


def _model_spec(model_id: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"([ABCD])-(\d+)k", model_id)
    if match is None:
        raise ValueError(f"Model must be a variant and numbered checkpoint: {model_id}")
    variant, thousands = match.groups()
    step = int(thousands) * 1000
    return variant, step, f"ema_model__iter_{step}.pth"


def resolve_models(registry_path: Path, names: list[str]) -> list[dict]:
    registry = _read_yaml(registry_path)
    if registry.get("schema") != "marvin_bimanual_model_registry/v1":
        raise ValueError("Invalid Marvin model registry schema")
    if not names or len(set(names)) != len(names):
        raise ValueError("Model IDs must be nonempty and unique")
    models = []
    for model_id in names:
        variant, step, filename = _model_spec(model_id)
        spec = registry.get("models", {}).get(variant)
        if not isinstance(spec, dict) or not spec.get("enabled"):
            raise ValueError(f"{model_id}: variant is not enabled in the registry")
        run_dir = Path(spec["run_dir"]).expanduser()
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        run_dir = run_dir.resolve()
        train = validate_checkpoint_args(
            _read_yaml(run_dir / "args.yaml"),
            expected_dataset_subdir=registry["dataset_subdir"],
            expected_variant=variant,
        )
        if not train.get("use_ema"):
            raise ValueError(f"{model_id}: requested checkpoint is EMA but run is not")
        checkpoint = run_dir / "checkpoints" / filename
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        models.append(
            {
                "id": model_id,
                "variant": variant,
                "step": step,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
                "dataset_subdir": train["dataset_subdir"],
            }
        )
    return models


def _effective_generation_config(output: Path, original: Path, regions_path: Path) -> Path:
    """Make object planner bounds cover the selected endpoint cells."""

    config = _read_yaml(original)
    regions = _read_yaml(regions_path)["object_regions"]
    bounds = config["object_planning"]["bounds"]
    lateral_offset = float(config["object_planning"].get("lateral_offset", 0.0))
    for index, axis in enumerate(("x", "y", "z", "yaw_deg")):
        limits = [region[axis] for region in regions.values()]
        low = min(pair[0] for pair in limits)
        high = max(pair[1] for pair in limits)
        if axis == "yaw_deg":
            low, high = math.radians(low), math.radians(high)
        if axis == "y":
            low -= lateral_offset
            high += lateral_offset
        bounds[index][0] = min(float(bounds[index][0]), low)
        bounds[index][1] = max(float(bounds[index][1]), high)
    effective = output / "configs" / "cooperative-generation-effective.yaml"
    _immutable_text(effective, yaml.safe_dump(config, sort_keys=False))
    return effective


def _materialize(
    output: Path,
    config_path: Path,
    base: dict,
    scenario_path: Path,
    regions_path: Path,
    registry_path: Path,
    generation_config: Path,
    tasks: dict,
    args,
) -> dict:
    models = resolve_models(registry_path, args.models)
    base = deepcopy(base)
    for key in (
        "cooperative_generation_config",
        "cooperative_regions_path",
        "cooperative_seed_file_path",
    ):
        if base.get(key):
            base[key] = str(_resolve_config_path(config_path, base[key]))
    base["cooperative_generation_config"] = str(generation_config.resolve())
    base.setdefault("cooperative_inference", {}).update(prior_mode="reference_residual", closed_chain_projection=True)
    if args.n_trajectory_samples is not None:
        base["n_trajectory_samples"] = args.n_trajectory_samples
        base["runtime_top_k_valid_trajectories"] = min(
            int(base.get("runtime_top_k_valid_trajectories", args.n_trajectory_samples)),
            args.n_trajectory_samples,
        )
    for model in models:
        config = deepcopy(base)
        config["model_dir_ddpm_bspline"] = model["run_dir"]
        config["checkpoint"] = Path(model["checkpoint"]).name
        config["runtime"]["network_variant"] = model["variant"]
        config["dataset_subdir"] = model["dataset_subdir"]
        config_path_out = output / "configs" / f"{model['id']}.yaml"
        _immutable_text(config_path_out, yaml.safe_dump(config, sort_keys=False))
        model["config_path"] = str(config_path_out.resolve())
        model["config_sha256"] = _sha256(config_path_out)
    request_hashes = {
        str(Path(task["request_path"]).resolve()): _sha256(Path(task["request_path"])) for task in tasks["tasks"]
    }
    manifest = {
        "schema": "marvin_cooperative_model_experiment/v1",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "scenarios": str(scenario_path),
        "scenarios_sha256": _sha256(scenario_path),
        "regions": str(regions_path),
        "regions_sha256": _sha256(regions_path),
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "generation_config": str(generation_config.resolve()),
        "generation_config_sha256": _sha256(generation_config),
        "tasks_file": str((output / "tasks/tasks.json").resolve()),
        "task_count": len(tasks["tasks"]),
        "sampling_shortfalls": tasks.get("sampling_shortfalls", {}),
        "request_sha256": request_hashes,
        "models": models,
        "seeds": list(args.seeds),
        "prior_mode": "reference_residual",
        "closed_chain_projection": True,
        "n_trajectory_samples": int(base["n_trajectory_samples"]),
        "device": args.device,
        "backend": args.backend,
        "timeout_s": float(args.timeout_s),
        "gpu_poll_interval": float(args.gpu_poll_interval),
    }
    _immutable_json(output / "manifest.json", manifest)
    return manifest


def _verify_fingerprints(manifest: dict, tasks: dict) -> None:
    generation_config = Path(manifest["generation_config"])
    if not generation_config.is_file() or _sha256(generation_config) != manifest["generation_config_sha256"]:
        raise RuntimeError(f"Pinned cooperative generation config changed: {generation_config}")
    for model in manifest["models"]:
        for path_key, hash_key in (
            ("checkpoint", "checkpoint_sha256"),
            ("config_path", "config_sha256"),
        ):
            path = Path(model[path_key])
            if not path.is_file() or _sha256(path) != model[hash_key]:
                raise RuntimeError(f"{model['id']}: pinned {path_key} changed: {path}")
    for task in tasks["tasks"]:
        path = Path(task["request_path"])
        if not path.is_file() or _sha256(path) != manifest["request_sha256"][str(path.resolve())]:
            raise RuntimeError(f"Frozen request changed: {path}")


def _read_trial_report(path: Path):
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    required = ("status", "model", "scenario", "task_index", "inference_seed", "wall_seconds")
    return row if isinstance(row, dict) and all(key in row for key in required) else None


def _preserve_incomplete(path: Path) -> None:
    """Keep interrupted artifacts for inspection before replacing them."""

    if path.is_file():
        path.rename(path.with_name(f"{path.name}.incomplete-{time.time_ns()}"))


def _ensure_trial_request(path: Path, request: dict) -> None:
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            _preserve_incomplete(path)
        else:
            if existing != request:
                raise ValueError(f"Existing trial request differs from frozen task: {path}")
            return
    _atomic_json(path, request)


def _run(output: Path, manifest: dict, tasks: dict) -> None:
    _verify_fingerprints(manifest, tasks)
    for task in tasks["tasks"]:
        original = json.loads(Path(task["request_path"]).read_text(encoding="utf-8"))
        for seed in manifest["seeds"]:
            request = deepcopy(original)
            request["seed"] = int(seed)
            request["request_id"] = f"{original['request_id']}-seed-{seed}"
            for model in manifest["models"]:
                artifact = (
                    output
                    / "runs"
                    / model["id"]
                    / task["difficulty"]
                    / task["scenario"]
                    / f"task-{task['task_index']:03d}"
                    / f"seed-{seed}"
                )
                report_path = artifact / "benchmark-result.json"
                completed = _read_trial_report(report_path)
                if completed is not None:
                    if (
                        completed["model"] != model["id"]
                        or completed["scenario"] != task["scenario"]
                        or completed["task_index"] != task["task_index"]
                        or completed["inference_seed"] != int(seed)
                    ):
                        raise ValueError(f"Existing trial result belongs to a different task: {report_path}")
                    continue
                _preserve_incomplete(report_path)
                request_path = artifact / "input-request.json"
                _ensure_trial_request(request_path, request)
                # An interrupted attempt may have left a result.json behind.
                # Keep it, but never let execute_case classify it as this run.
                _preserve_incomplete(artifact / "result.json")
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
                row = execute_case(Path(model["config_path"]), artifact, run_args)
                row.update(
                    model=model["id"],
                    variant=model["variant"],
                    checkpoint_step=model["step"],
                    scenario=task["scenario"],
                    difficulty=task["difficulty"],
                    task_index=task["task_index"],
                    endpoint_seed=task["endpoint_seed"],
                    inference_seed=int(seed),
                    task_metrics={key: value for key, value in task.items() if key.startswith("object_")},
                )
                error = row.get("result_error") or {}
                row["failure_reason"] = error.get("type") or row["status"]
                _atomic_json(report_path, row)
                print(
                    f"{task['scenario']} task={task['task_index']} seed={seed} " f"{model['id']}: {row['status']}",
                    flush=True,
                )


def _collect_rows(output: Path, manifest: dict, tasks: dict) -> list[dict]:
    rows = []
    for task in tasks["tasks"]:
        for seed in manifest["seeds"]:
            for model in manifest["models"]:
                path = (
                    output
                    / "runs"
                    / model["id"]
                    / task["difficulty"]
                    / task["scenario"]
                    / f"task-{task['task_index']:03d}"
                    / f"seed-{seed}"
                    / "benchmark-result.json"
                )
                row = _read_trial_report(path)
                if row is not None:
                    rows.append(row)
    return rows


def _add_failure_denominators(aggregates: list[dict]) -> None:
    for row in aggregates:
        errors = row["status_counts"]
        evaluable = row["successes"] + errors.get("no_valid_trajectory", 0)
        row["planning_evaluable_runs"] = evaluable
        row["reference_or_runtime_failures"] = row["runs"] - evaluable
        row["planning_success_rate"] = row["successes"] / evaluable if evaluable else None


def _summarize(output: Path, manifest: dict, tasks: dict) -> dict:
    rows = _collect_rows(output, manifest, tasks)
    by_model = _aggregate(rows, ("model",))
    by_difficulty = _aggregate(rows, ("difficulty", "model"))
    by_scenario = _aggregate(rows, ("scenario", "model"))
    for group in (by_model, by_difficulty, by_scenario):
        _add_failure_denominators(group)
    summary = {
        "schema": "marvin_cooperative_model_summary/v1",
        "expected_runs": manifest["task_count"] * len(manifest["seeds"]) * len(manifest["models"]),
        "completed_runs": len(rows),
        "by_model": by_model,
        "by_difficulty_model": by_difficulty,
        "by_scenario_model": by_scenario,
    }
    _atomic_json(output / "summary.json", summary)
    _write_csv(output / "summary-by-model.csv", by_model)
    _write_csv(output / "summary-by-difficulty-model.csv", by_difficulty)
    _write_csv(output / "summary-by-scenario-model.csv", by_scenario)
    lines = [
        "# Cooperative checkpoint benchmark",
        "",
        f"Completed {len(rows)}/{summary['expected_runs']} runs; reference_residual + projection on.",
        "",
        "| model | end-to-end success | planning success* | mean wall (s) | mean valid candidates |",
        "|---|---:|---:|---:|---:|",
    ]
    by_name = {row["model"]: row for row in by_model}
    for model in manifest["models"]:
        row = by_name.get(model["id"])
        if row is None:
            lines.append(f"| {model['id']} | pending | pending | pending | pending |")
            continue
        conditional = row["planning_success_rate"]
        conditional_text = "n/a" if conditional is None else f"{conditional:.1%}"
        valid = row["mean_valid_candidates"]
        valid_text = "n/a" if valid is None else f"{valid:.2f}"
        lines.append(
            f"| {model['id']} | {row['successes']}/{row['runs']} ({row['success_rate']:.1%}) "
            f"| {conditional_text} | {row['mean_wall_s']:.2f} | {valid_text} |"
        )
    lines.extend(["", "*Planning success excludes reference construction and runtime faults from the denominator."])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "all", "summarize"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--model-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--difficulties", nargs="+", choices=("easy", "medium", "hard"))
    parser.add_argument("--tasks-per-scenario", type=int, default=5)
    parser.add_argument("--max-endpoint-attempts", type=int, default=20)
    parser.add_argument("--endpoint-seed", type=int, default=47000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1101, 1102, 1103])
    parser.add_argument("--n-trajectory-samples", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--gpu-poll-interval", type=float, default=0.2)
    parser.add_argument("--backend", choices=("mpd", "contract_stub"), default="mpd")
    args = parser.parse_args(argv)
    if (
        args.tasks_per_scenario < 1
        or args.max_endpoint_attempts < 1
        or args.timeout_s <= 0
        or args.gpu_poll_interval <= 0
        or (args.n_trajectory_samples is not None and args.n_trajectory_samples < 1)
    ):
        parser.error("Task counts, attempts, candidates, timeout and polling must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Inference seeds must be unique uint32 values")
    output = args.output_dir.expanduser().resolve()
    if args.stage in ("run", "summarize"):
        tasks = json.loads((output / "tasks/tasks.json").read_text(encoding="utf-8"))
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    else:
        config_path = args.config.expanduser().resolve()
        scenario_path = args.scenario_config.expanduser().resolve()
        regions_path = args.regions.expanduser().resolve()
        registry_path = args.model_registry.expanduser().resolve()
        base = _read_yaml(config_path)
        _, scenarios = load_scenarios(scenario_path, regions_path, names=args.scenarios, difficulties=args.difficulties)
        original_generation_config = _resolve_config_path(config_path, base["cooperative_generation_config"])
        generation_config = _effective_generation_config(output, original_generation_config, regions_path)
        tasks = prepare_tasks(
            output,
            scenarios,
            regions_path,
            generation_config,
            tasks_per_scenario=args.tasks_per_scenario,
            max_endpoint_attempts=args.max_endpoint_attempts,
            endpoint_seed=args.endpoint_seed,
        )
        shortfalls = {name: value for name, value in tasks.get("sampling_shortfalls", {}).items() if value}
        if shortfalls:
            raise RuntimeError(
                f"Cooperative endpoint sampling missed task quotas: {shortfalls}. "
                "Inspect tasks/tasks.json and use a new output directory for a revised experiment."
            )
        manifest = _materialize(
            output, config_path, base, scenario_path, regions_path, registry_path, generation_config, tasks, args
        )
    if args.stage == "prepare":
        print(output / "manifest.json")
        return 0
    if args.stage != "summarize":
        if manifest["backend"] == "mpd" and str(manifest["device"]).startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                parser.error("CUDA unavailable; use --stage prepare or --backend contract_stub")
        _run(output, manifest, tasks)
    _summarize(output, manifest, tasks)
    print(output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
