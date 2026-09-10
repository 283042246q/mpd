#!/usr/bin/env python3
"""Generate and execute Marvin collision-optimization inference ablations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ROOT / "scripts/inference/inference_marvin_bimanual.py"
DEFAULT_CONFIG = (
    ROOT
    / "scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml"
)
CASES = {
    "A0": (False, False, False, False),
    "A1": (True, False, False, False),
    "A2": (True, True, False, False),
    "A3": (True, True, True, False),
    "A4": (True, True, False, True),
    "A5": (True, True, True, True),
}


def build_case_config(base, case_id, batch_size, args):
    pair_streaming, validator_chunking, parent_bounds, foam = CASES[case_id]
    config = deepcopy(base)
    config["n_trajectory_samples"] = int(batch_size)
    config["runtime_top_k_valid_trajectories"] = min(
        int(config.get("runtime_top_k_valid_trajectories", batch_size)), batch_size
    )
    collision = config.setdefault("collision_optimization", {})
    collision["pair_streaming"] = {
        "enabled": pair_streaming,
        "pair_chunk_size": int(args.pair_chunk_size),
    }
    collision["reduced_guide_geometry"] = {
        "enabled": foam,
        "profile": "foam_pika_100",
    }
    dense = config.setdefault("dense_validation", {})
    dense["chunking"] = {
        "enabled": validator_chunking,
        "candidate_chunk_size": int(args.validator_candidate_chunk_size),
        "time_chunk_size": int(args.validator_time_chunk_size),
        "self_pair_chunk_size": int(args.validator_pair_chunk_size),
    }
    spatial = config.setdefault("gradient_pruning", {}).setdefault("spatial", {})
    broad_phase = spatial.setdefault("link_broad_phase", {})
    broad_phase.update(
        enabled=parent_bounds,
        full_scan=True,
        scan_geometry="parent_bounds",
        environment_margin=float(args.parent_environment_margin),
        self_margin=float(args.parent_self_margin),
    )
    return config


def _query_device_memory_mib(device_index):
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        used, free, total = (
            int(value.strip()) for value in completed.stdout.splitlines()[0].split(",")
        )
        return {"used_mib": used, "free_mib": free, "total_mib": total}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _monitor_gpu(stop, samples, device_index, interval):
    while not stop.is_set():
        sample = _query_device_memory_mib(device_index)
        if sample is not None:
            sample["monotonic_s"] = time.monotonic()
            samples.append(sample)
        stop.wait(interval)


def _classify(returncode, payload, stdout, stderr):
    message = "\n".join(
        str(value)
        for value in (
            payload.get("error", {}).get("message") if isinstance(payload, dict) else "",
            stdout,
            stderr,
        )
    )
    if "out of memory" in message.lower() or "cuda_oom" in message.lower():
        return "cuda_oom"
    if isinstance(payload, dict) and payload.get("status") == "success":
        return "success"
    if isinstance(payload, dict) and payload.get("status"):
        return str(payload["status"])
    return "process_error" if returncode else "missing_result"


def execute_case(config_path, artifact_dir, args):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = artifact_dir / "stdout.log"
    stderr_path = artifact_dir / "stderr.log"
    command = [
        sys.executable,
        str(INFERENCE),
        "--config",
        str(config_path),
        "--start-goal-source",
        args.start_goal_source,
        "--sample-index",
        str(args.sample_index),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(artifact_dir),
        "--device",
        args.device,
        "--sim-backend",
        "none",
    ]
    device_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
    samples = []
    stop = threading.Event()
    monitor = threading.Thread(
        target=_monitor_gpu,
        args=(stop, samples, device_index, args.gpu_poll_interval),
        daemon=True,
    )
    started = time.perf_counter()
    timed_out = False
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=dict(os.environ),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        monitor.start()
        try:
            returncode = process.wait(timeout=args.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
        finally:
            stop.set()
            monitor.join(timeout=10)
    elapsed = time.perf_counter() - started
    stdout_text = stdout_path.read_text(errors="replace")
    stderr_text = stderr_path.read_text(errors="replace")
    result_path = artifact_dir / "result.json"
    payload = json.loads(result_path.read_text()) if result_path.is_file() else {}
    status = "timeout" if timed_out else _classify(
        returncode, payload, stdout_text, stderr_text
    )
    memory = {}
    if samples:
        memory = {
            "first_used_mib": samples[0]["used_mib"],
            "peak_device_used_mib": max(sample["used_mib"] for sample in samples),
            "minimum_device_free_mib": min(sample["free_mib"] for sample in samples),
            "total_mib": samples[0]["total_mib"],
            "samples": len(samples),
        }
    return {
        "status": status,
        "returncode": returncode,
        "wall_seconds": elapsed,
        "gpu_memory": memory,
        "result_status": payload.get("status"),
        "result_error": payload.get("error"),
        "timing": payload.get("timing"),
        "collision_geometry": payload.get("collision_geometry"),
        "validation": payload.get("validation"),
        "candidates": payload.get("candidates"),
        "command": command,
        "artifact_dir": str(artifact_dir),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--capacity-batches",
        nargs="*",
        type=int,
        default=(16, 8),
        help="Also run A0 and A5 at these candidate counts.",
    )
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=tuple(CASES))
    parser.add_argument("--start-goal-source", choices=("dataset", "states_file", "regions"), default="dataset")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--pair-chunk-size", type=int, default=1024)
    parser.add_argument("--validator-candidate-chunk-size", type=int, default=8)
    parser.add_argument("--validator-time-chunk-size", type=int, default=32)
    parser.add_argument("--validator-pair-chunk-size", type=int, default=1024)
    parser.add_argument("--parent-environment-margin", type=float, default=0.20)
    parser.add_argument("--parent-self-margin", type=float, default=0.10)
    parser.add_argument("--gpu-poll-interval", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument(
        "--execute", action="store_true", help="Execute generated inference cases."
    )
    args = parser.parse_args()
    positive = (
        args.batch_size,
        *args.capacity_batches,
        args.pair_chunk_size,
        args.validator_candidate_chunk_size,
        args.validator_time_chunk_size,
        args.validator_pair_chunk_size,
        args.timeout_s,
    )
    if min(positive) < 1 or args.gpu_poll_interval <= 0:
        raise SystemExit("batch/chunk/timeout values must be positive")

    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    configs_dir = output_dir / "configs"
    artifacts_dir = output_dir / "artifacts"
    configs_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(config_path.read_text())
    matrix = [(case_id, args.batch_size) for case_id in args.cases]
    matrix.extend(
        (case_id, batch)
        for batch in args.capacity_batches
        for case_id in ("A0", "A5")
        if case_id in args.cases and batch != args.batch_size
    )
    report = {
        "schema": "marvin_collision_optimization_ablation/v1",
        "base_config": str(config_path),
        "device": args.device,
        "sample_index": args.sample_index,
        "seed": args.seed,
        "cases": {},
    }
    for case_id, batch_size in matrix:
        run_id = f"{case_id}-b{batch_size}"
        config = build_case_config(base, case_id, batch_size, args)
        generated_config = configs_dir / f"{run_id}.yaml"
        generated_config.write_text(yaml.safe_dump(config, sort_keys=False))
        entry = {
            "case": case_id,
            "batch_size": batch_size,
            "switches": dict(
                zip(
                    ("pair_streaming", "validator_chunking", "parent_bounds", "foam_guide"),
                    CASES[case_id],
                )
            ),
            "config": str(generated_config),
            "status": "generated",
        }
        if args.execute:
            entry.update(
                execute_case(generated_config, artifacts_dir / run_id, args)
            )
        report["cases"][run_id] = entry
        report_path = output_dir / "ablation-report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"{run_id}: {entry['status']}", flush=True)
    print(output_dir / "ablation-report.json")


if __name__ == "__main__":
    main()
