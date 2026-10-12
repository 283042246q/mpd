#!/usr/bin/env python3
"""Run the eight cooperative benchmark scenarios sequentially with D-600k.

Each endpoint task gets the benchmark's three inference seeds. If all fail,
resample endpoints within the same scenario before trying the next three.
Only a completed Isaac Lab check and nonempty video advance to the next scenario.
Re-running with the same output directory preserves prior attempts and resumes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

from mpd.bimanual.cooperative_sources import request_from_regions
from mpd.bimanual.runtime_contract import BimanualRequest
from scripts.inference import benchmark_marvin_cooperative_models as benchmark
from scripts.inference.benchmark_marvin_collision_optimizations import execute_case
from scripts.inference.benchmark_marvin_cooperative_priors import (
    _atomic_json,
    _read_yaml,
    _resolve_config_path,
    _task_metrics,
    load_scenarios,
    prepare_tasks,
)


MODEL = "D-600k"
REPLAY = benchmark.ROOT / "scripts/inference/replay_marvin_cooperative_isaaclab.py"


def _attempt_dir(output: Path, task: dict, attempt: int, seed: int) -> Path:
    return (
        output / "runs" / MODEL / task["difficulty"] / task["scenario"]
        / f"task-{task['task_index']:03d}" / f"attempt-{attempt:03d}-seed-{seed}"
    )


def _read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _resampled_task(output: Path, manifest: dict, original: dict, index: int, endpoint_attempts: int) -> dict:
    """Freeze a new endpoint request without modifying the original manifest."""

    assert index > 0
    scenario = original["scenario"]
    request_path = output / "tasks" / "requests" / scenario / f"task-{index:03d}.json"
    metadata_path = output / "tasks" / "resampled" / scenario / f"task-{index:03d}.json"
    metadata = _read_json(metadata_path)
    if metadata is not None:
        if (metadata.get("scenario") != scenario or metadata.get("task_index") != index
                or metadata.get("request_path") != str(request_path.resolve())
                or not request_path.is_file()
                or metadata.get("request_sha256") != benchmark._sha256(request_path)):
            raise RuntimeError(f"Frozen resampled task changed: {metadata_path}")
        return metadata
    if metadata_path.exists():
        raise RuntimeError(f"Incomplete resampled task metadata: {metadata_path}")
    request = _read_json(request_path)
    if request is None and request_path.exists():
        raise RuntimeError(f"Incomplete resampled request: {request_path}")
    if request is None:
        region_path = output / "tasks" / "effective-regions" / f"{scenario}.yaml"
        if not region_path.is_file():
            raise FileNotFoundError(region_path)
        errors = []
        for offset in range(endpoint_attempts):
            endpoint_seed = int(original["endpoint_seed"]) + 1 + (index - 1) * endpoint_attempts + offset
            if endpoint_seed >= 2**32:
                raise ValueError("Endpoint seed exceeds uint32")
            try:
                request = request_from_regions(
                    region_path, generator_config_path=Path(manifest["generation_config"]),
                    request_id=f"coop-prior-{scenario}-{index:03d}", seed=endpoint_seed,
                )
                BimanualRequest.from_dict(request)
                request["scene"]["start_goal_source"].update(
                    benchmark_scenario=scenario, difficulty=original["difficulty"],
                    transition=original["transition"], endpoint_seed=endpoint_seed,
                )
                _atomic_json(request_path, request)
                break
            except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
                errors.append(f"{endpoint_seed}: {type(error).__name__}: {error}")
        else:
            raise RuntimeError(f"{scenario} task-{index:03d}: endpoint sampling failed after "
                               f"{endpoint_attempts} attempts; last errors: {errors[-3:]}")
    if request.get("request_id") != f"coop-prior-{scenario}-{index:03d}":
        raise ValueError(f"Resampled request ID differs: {request_path}")
    BimanualRequest.from_dict(request)
    source = request["scene"]["start_goal_source"]
    if source.get("benchmark_scenario") != scenario:
        raise ValueError(f"Resampled request scenario differs: {request_path}")
    metadata = {**original, "task_index": index, "endpoint_seed": int(source["endpoint_seed"]),
                "request_path": str(request_path.resolve()), "request_sha256": benchmark._sha256(request_path),
                **_task_metrics(request)}
    _atomic_json(metadata_path, metadata)
    return metadata


def _recorded(artifact: Path) -> bool:
    run = _read_json(artifact / "isaaclab-run.json")
    video = artifact / "isaaclab-replay.mp4"
    return bool(run and run.get("status") == "completed" and video.is_file() and video.stat().st_size > 0)


def _completed_marker(scenario_root: Path):
    for endpoint_root in sorted(scenario_root.glob("task-*")):
        marker = _read_json(endpoint_root / "task-complete.json")
        if marker and marker.get("artifact") and _recorded(Path(marker["artifact"])):
            return marker
    return None


def _retryable_failure(report: dict) -> bool:
    if report["status"] in ("no_valid_trajectory", "timeout"):
        return True
    error = report.get("result_error") or {}
    return (report["status"] == "fault"
            and error.get("message") == "could not build a continuous paired-IK reference")


def _replay(artifact: Path, args) -> str:
    command = [
        sys.executable, str(REPLAY), "--result", str(artifact),
        "--isaaclab-conda-env", args.isaaclab_conda_env,
        "--isaaclab-device", args.isaaclab_device,
        "--isaaclab-timeout-s", str(args.isaaclab_timeout_s),
        "--isaaclab-headless", "--isaaclab-replay", "--isaaclab-capture",
    ]
    with (artifact / "isaaclab-launch-stdout.log").open("w", encoding="utf-8") as stdout, (
        artifact / "isaaclab-launch-stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=benchmark.ROOT, stdout=stdout, stderr=stderr, check=False)
    if _recorded(artifact) and completed.returncode == 0:
        return "completed"
    run = _read_json(artifact / "isaaclab-run.json") or {}
    if completed.returncode == 7 and run.get("status") == "safety_validation_failed":
        return "safety_validation_failed"
    raise RuntimeError(
        f"Isaac Lab recording failed for {artifact} (exit={completed.returncode}, "
        f"status={run.get('status')!r}); inspect isaaclab-launch-stderr.log and isaaclab-replay.log"
    )


def _summary(output: Path, tasks: dict) -> dict:
    rows = []
    for task in tasks["tasks"]:
        scenario_root = output / "runs" / MODEL / task["difficulty"] / task["scenario"]
        attempts = []
        for endpoint_root in sorted(scenario_root.glob("task-*")):
            for artifact in sorted(endpoint_root.glob("attempt-*-seed-*")):
                report = _read_json(artifact / "benchmark-result.json")
                if report:
                    attempts.append({
                        "artifact": str(artifact), "endpoint_task_index": report["task_index"],
                        "seed": report["inference_seed"], "mpd_status": report["status"],
                        "isaaclab_status": (_read_json(artifact / "isaaclab-run.json") or {}).get("status"),
                    })
        complete = _completed_marker(scenario_root)
        rows.append({"scenario": task["scenario"], "difficulty": task["difficulty"],
                     "complete": bool(complete), "attempts": attempts,
                     "video": complete["video"] if complete else None,
                     "completed_endpoint_task_index": complete.get("endpoint_task_index") if complete else None})
    value = {"schema": "marvin_cooperative_d600k_videos/v2", "completed_tasks": sum(r["complete"] for r in rows),
             "total_tasks": len(rows), "tasks": rows}
    _atomic_json(output / "summary.json", value)
    return value


def _run(output: Path, manifest: dict, tasks: dict, args) -> dict:
    benchmark._verify_fingerprints(manifest, tasks)
    model = manifest["models"][0]
    assert model["id"] == MODEL
    if len(manifest["seeds"]) != 3:
        raise ValueError("This runner requires exactly three benchmark inference seeds")
    settings = _read_json(output / "tasks" / "settings.json") or {}
    endpoint_attempts = int(settings.get("max_endpoint_attempts", args.max_endpoint_attempts))
    for original_task in tasks["tasks"]:
        scenario_root = output / "runs" / MODEL / original_task["difficulty"] / original_task["scenario"]
        if _completed_marker(scenario_root):
            print(f"skip completed {original_task['scenario']}", flush=True)
            continue
        # An older runner may already have planned a success beyond attempt 2.
        # Reuse and record it before sampling new endpoints.
        for artifact in sorted(scenario_root.glob("task-*/attempt-*-seed-*")):
            report = _read_json(artifact / "benchmark-result.json")
            if not report or report.get("status") != "success":
                continue
            previous_isaac = _read_json(artifact / "isaaclab-run.json") or {}
            if previous_isaac.get("status") == "safety_validation_failed":
                continue
            status = "completed" if _recorded(artifact) else _replay(artifact, args)
            if status == "completed":
                endpoint_root = artifact.parent
                _atomic_json(endpoint_root / "task-complete.json",
                             {"artifact": str(artifact), "video": str(artifact / "isaaclab-replay.mp4"),
                              "endpoint_task_index": report["task_index"],
                              "attempt_index": report.get("attempt_index"),
                              "inference_seed": report["inference_seed"]})
                _summary(output, tasks)
                break
        if _completed_marker(scenario_root):
            continue
        task_index = 0
        while True:
            if args.max_endpoint_tasks_per_scenario and task_index >= args.max_endpoint_tasks_per_scenario:
                _summary(output, tasks)
                raise RuntimeError(f"{original_task['scenario']}: endpoint task limit reached; not advancing")
            task = (original_task if task_index == 0 else
                    _resampled_task(output, manifest, original_task, task_index, endpoint_attempts))
            original_request = _read_json(Path(task["request_path"]))
            if original_request is None:
                raise RuntimeError(f"Missing or invalid frozen task request: {task['request_path']}")
            for attempt, seed in enumerate(manifest["seeds"]):
                artifact = _attempt_dir(output, task, attempt, seed)
                artifact.mkdir(parents=True, exist_ok=True)
                request = deepcopy(original_request)
                request["seed"] = int(seed)
                request["request_id"] = f"{original_request['request_id']}-seed-{seed}"
                request_path = artifact / "input-request.json"
                benchmark._ensure_trial_request(request_path, request)
                report_path = artifact / "benchmark-result.json"
                report = benchmark._read_trial_report(report_path)
                if report is not None and (report["model"], report["scenario"], report["inference_seed"],
                                           report["task_index"]) != (MODEL, task["scenario"], seed, task_index):
                    raise ValueError(f"Existing attempt belongs to another task: {report_path}")
                if report is None:
                    benchmark._preserve_incomplete(report_path)
                    benchmark._preserve_incomplete(artifact / "result.json")
                    for name in ("isaaclab-run.json", "isaaclab-replay.mp4", "isaaclab-replay.png"):
                        benchmark._preserve_incomplete(artifact / name)
                    run_args = SimpleNamespace(request=request_path, start_goal_source="regions", sample_index=0,
                                               seed=seed, device=manifest["device"], timeout_s=manifest["timeout_s"],
                                               gpu_poll_interval=manifest["gpu_poll_interval"], backend="mpd")
                    report = execute_case(Path(model["config_path"]), artifact, run_args)
                    report.update(model=MODEL, variant="D", checkpoint_step=600_000,
                                  scenario=task["scenario"], difficulty=task["difficulty"],
                                  task_index=task_index, endpoint_seed=task["endpoint_seed"],
                                  inference_seed=seed, attempt_index=attempt)
                    _atomic_json(report_path, report)
                print(f"{task['scenario']} task={task_index} seed={seed}: MPD {report['status']}", flush=True)
                if report["status"] == "success":
                    try:
                        prior_isaac = _read_json(artifact / "isaaclab-run.json") or {}
                        status = ("safety_validation_failed" if prior_isaac.get("status") == "safety_validation_failed"
                                  else "completed" if _recorded(artifact) else _replay(artifact, args))
                    except Exception:
                        _summary(output, tasks)
                        raise
                    print(f"{task['scenario']} task={task_index} seed={seed}: Isaac Lab {status}", flush=True)
                    if status == "completed":
                        video = artifact / "isaaclab-replay.mp4"
                        endpoint_root = artifact.parent
                        _atomic_json(endpoint_root / "task-complete.json",
                                     {"artifact": str(artifact), "video": str(video),
                                      "endpoint_task_index": task_index,
                                      "attempt_index": attempt, "inference_seed": seed})
                        _summary(output, tasks)
                        break
                elif not _retryable_failure(report):
                    _summary(output, tasks)
                    raise RuntimeError(
                        f"{task['scenario']}: non-retryable MPD failure {report['status']!r} in {artifact}; not advancing"
                    )
                _summary(output, tasks)
            else:
                print(f"{task['scenario']} task={task_index}: three seeds failed; resampling endpoints", flush=True)
                task_index += 1
                continue
            _summary(output, tasks)
            break
    return _summary(output, tasks)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=benchmark.DEFAULT_CONFIG)
    parser.add_argument("--scenario-config", type=Path, default=benchmark.DEFAULT_SCENARIOS)
    parser.add_argument("--regions", type=Path, default=benchmark.DEFAULT_REGIONS)
    parser.add_argument("--model-registry", type=Path, default=benchmark.DEFAULT_REGISTRY)
    parser.add_argument("--max-endpoint-attempts", type=int, default=20)
    parser.add_argument("--endpoint-seed", type=int, default=47000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1101, 1102, 1103])
    parser.add_argument("--max-endpoint-tasks-per-scenario", type=int, default=0,
                        help="0 keeps resampling until success; a positive limit stops without advancing")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--gpu-poll-interval", type=float, default=0.2)
    parser.add_argument("--isaaclab-conda-env", default="env_isaaclab")
    parser.add_argument("--isaaclab-device", default="cuda:0")
    parser.add_argument("--isaaclab-timeout-s", type=int, default=900)
    args = parser.parse_args(argv)
    if (args.max_endpoint_attempts < 1 or args.max_endpoint_tasks_per_scenario < 0 or args.timeout_s <= 0
            or args.gpu_poll_interval <= 0 or args.isaaclab_timeout_s < 1 or not args.seeds
            or len(args.seeds) != 3 or len(set(args.seeds)) != 3
            or any(not 0 <= s < 2**32 for s in args.seeds)):
        parser.error("Invalid attempts, seeds, timeout or polling interval")
    output = args.output_dir.expanduser().resolve()
    if args.stage == "run":
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        tasks = json.loads((output / "tasks/tasks.json").read_text(encoding="utf-8"))
    else:
        config_path = args.config.expanduser().resolve()
        scenario_path = args.scenario_config.expanduser().resolve()
        regions_path = args.regions.expanduser().resolve()
        registry_path = args.model_registry.expanduser().resolve()
        base = _read_yaml(config_path)
        _, scenarios = load_scenarios(scenario_path, regions_path)
        if len(scenarios) != 8:
            raise ValueError(f"Expected eight benchmark scenarios, found {len(scenarios)}")
        generation = benchmark._effective_generation_config(
            output, _resolve_config_path(config_path, base["cooperative_generation_config"]), regions_path
        )
        tasks = prepare_tasks(output, scenarios, regions_path, generation, tasks_per_scenario=1,
                              max_endpoint_attempts=args.max_endpoint_attempts, endpoint_seed=args.endpoint_seed)
        if len(tasks["tasks"]) != 8 or any(tasks.get("sampling_shortfalls", {}).values()):
            raise RuntimeError(f"Could not freeze all eight tasks: {tasks.get('sampling_shortfalls')}")
        manifest_args = SimpleNamespace(models=[MODEL], n_trajectory_samples=None, seeds=args.seeds,
                                        device=args.device, backend="mpd", timeout_s=args.timeout_s,
                                        gpu_poll_interval=args.gpu_poll_interval)
        manifest = benchmark._materialize(output, config_path, base, scenario_path, regions_path,
                                          registry_path, generation, tasks, manifest_args)
    if (len(tasks["tasks"]) != 8 or len(manifest["models"]) != 1 or manifest["models"][0]["id"] != MODEL
            or manifest.get("prior_mode") != "reference_residual"
            or manifest.get("closed_chain_projection") is not True or manifest.get("backend") != "mpd"):
        raise ValueError("Output directory is not an eight-task D-600k run")
    if args.stage == "prepare":
        print(output / "manifest.json")
        return 0
    if str(manifest["device"]).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --stage prepare or a CPU device")
    summary = _run(output, manifest, tasks, args)
    print(output / "summary.json")
    return 0 if summary["completed_tasks"] == 8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
