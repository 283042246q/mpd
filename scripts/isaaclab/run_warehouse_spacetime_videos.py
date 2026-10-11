"""Generate three validated Warehouse cases per motion mode and render a suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import torch

from scripts.isaaclab.scene_payload import export_isaaclab_scene_payload
from scripts.isaaclab.subprocess_utils import _clean_isaaclab_subprocess_env
from scripts.isaaclab.warehouse_spacetime_payload import validate_suite
from scripts.spacetime_data.generate_warehouse_spacetime import atomic_json
from scripts.spacetime_data.warehouse_world import MOTION_MODES, WarehouseCollision


ROOT = Path(__file__).resolve().parents[2]


def export_cases(dataset, mode, count, destination, robot):
    """Export accepted q(t)/world(t) data, retaining the actual SI time grids."""
    index = json.loads((dataset / "index.json").read_text())
    entries = sorted([e for rows in index["splits"].values() for e in rows], key=lambda e: e["attempt_id"])
    if len(entries) < count:
        raise ValueError("not enough accepted samples: " + mode)
    cases, signatures = [], set()
    for variant, entry in enumerate(entries[:count], 1):
        case_id = "%s-%02d" % (mode, variant)
        path = destination / (case_id + ".npz")
        with h5py.File(dataset / entry["file"], "r") as shard:
            sample = shard[entry["group"]]
            world = json.loads(sample.attrs["world_metadata_json"])
            if any(value != mode for value in world["modes"]):
                raise ValueError("mixed motion modes in a single-mode demo")
            values = {"execution_time": np.asarray(sample["execution/time"]),
                      "q": np.asarray(sample["execution/q"]), "dq": np.asarray(sample["execution/dq"]),
                      "forecast_time": np.asarray(sample["world/times"]),
                      "obstacle_positions": np.asarray(sample["world/positions"]),
                      "shape": np.asarray(sample["world/shape"]), "size": np.asarray(sample["world/size"]),
                      "event_times": np.asarray(sample["world/event_times"])}
            metadata = {"case_id": case_id, "mode": mode, "variant": variant, "dataset": str(dataset),
                        "entry": entry, "seed": int(sample.attrs["seed"]),
                        "source_row": int(sample.attrs["source_row"]), "backend": str(sample.attrs["backend"]),
                        "quality": json.loads(sample.attrs["quality_json"]), "world": world,
                        "robot_replay": "kinematic; no physics execution claim"}
        signature = hashlib.sha256(values["obstacle_positions"].tobytes() + values["q"].tobytes()).hexdigest()
        if signature in signatures:
            raise ValueError("duplicate variants in mode " + mode)
        signatures.add(signature)
        with torch.no_grad():
            values["ee_positions"] = robot.get_EE_pose(torch.as_tensor(values["q"], dtype=torch.float32))[:, :3, 3].cpu().numpy()
            values["ee_path"] = values["ee_positions"][::8]
        values["metadata_json"] = np.asarray(json.dumps(metadata))
        temporary = path.with_suffix(".npz.inprogress")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
        os.replace(temporary, path)
        cases.append({"case_id": case_id, "mode": mode, "variant": variant, "payload": str(path),
                      "signature": signature, "seed": metadata["seed"], "source_row": metadata["source_row"],
                      "duration_s": float(values["execution_time"][-1])})
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MOTION_MODES, default=list(MOTION_MODES))
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--backend", choices=("torch-forest", "ompl-strrtstar"), default="torch-forest")
    parser.add_argument("--planning-device", default="cuda:0")
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--solve-seconds", type=float, default=3.)
    parser.add_argument("--max-attempts-per-mode", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/eric/IsaacLab"))
    parser.add_argument("--isaac-prefix", type=Path, default=Path("/home/eric/anaconda3/envs/env_isaaclab"))
    parser.add_argument("--robot-usd")
    parser.add_argument("--resume", action="store_true")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true")
    stage.add_argument("--render-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.variants, args.width, args.height, args.cpu_workers, args.fps, args.solve_seconds) <= 0:
        parser.error("counts, resolution and budgets must be positive")
    if len(set(args.modes)) != len(args.modes):
        parser.error("mode list must be unique")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    manifest_path = output / "suite.json"
    if not args.render_only:
        cases = []
        (output / "payloads").mkdir(exist_ok=True)
        (output / "logs").mkdir(exist_ok=True)
        model = WarehouseCollision("cpu")
        scene = export_isaaclab_scene_payload(model.environment, include_boxes=True)
        if scene["unsupported_obstacles"]:
            raise ValueError("cannot render complete Warehouse geometry")
        atomic_json(output / "static_scene.json", scene)
        for mode in args.modes:
            mode_index = MOTION_MODES.index(mode)
            dataset = output / "datasets" / mode
            command = [sys.executable, "-m", "scripts.spacetime_data.generate_warehouse_spacetime",
                       "--output-root", str(dataset), "--backend", args.backend,
                       "--device", args.planning_device, "--num-trajectories", str(args.variants),
                       "--query-batch-size", str(args.variants), "--cpu-workers", str(args.cpu_workers),
                       "--solve-seconds", str(args.solve_seconds), "--max-iterations", "256",
                       "--max-attempts", str(args.max_attempts_per_mode), "--motion-modes", mode,
                       "--seed", str(args.seed + mode_index * 100), "--start-index", str(2 + mode_index * 40),
                       "--obstacles", "3", "--require-tau-r"]
            if args.resume:
                command.append("--resume")
            print("[warehouse-suite] generating " + mode, flush=True)
            with (output / "logs" / (mode + "-generation.log")).open("a") as log:
                subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True)
            cases.extend(export_cases(dataset, mode, args.variants, output / "payloads", model.robot))
            atomic_json(manifest_path, {"schema": "warehouse_spacetime_video_suite_v1", "static_scene": str(output / "static_scene.json"),
                                       "expected_cases": len(args.modes) * args.variants, "cases": cases,
                                       "modes": args.modes, "variants_per_mode": args.variants})
            print("[warehouse-suite] prepared %s: %d variants" % (mode, args.variants), flush=True)
    if args.prepare_only:
        return 0
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    validate_suite(json.loads(manifest_path.read_text()))
    env = _clean_isaaclab_subprocess_env()
    env["CONDA_PREFIX"] = str(args.isaac_prefix)
    command = [str(args.isaaclab_root / "isaaclab.sh"), "-p", str(ROOT / "scripts/isaaclab/replay_warehouse_spacetime.py"),
               "--suite", str(manifest_path), "--output_dir", str(output / "videos"),
               "--video_fps", str(args.fps), "--width", str(args.width), "--height", str(args.height),
               "--headless", "--enable_cameras"]
    if args.robot_usd:
        command.extend(["--robot_usd", args.robot_usd])
    if args.resume:
        command.append("--resume")
    (output / "logs").mkdir(exist_ok=True)
    print("[warehouse-suite] rendering frozen suite in IsaacLab", flush=True)
    with (output / "logs/isaaclab-render.log").open("a") as log:
        subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([str(args.isaac_prefix / "bin/python"), str(ROOT / "scripts/isaaclab/finalize_warehouse_spacetime_videos.py"),
                    "--suite", str(manifest_path), "--output_dir", str(output / "videos")], cwd=str(ROOT), env=env, check=True)
    print("[warehouse-suite] complete: " + str(output / "videos/report.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
