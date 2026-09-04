#!/usr/bin/env python3
"""Create an auditable Marvin 14D trajectory dataset scaffold."""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import numpy as np
import yaml

JOINT_NAMES = tuple([f"Joint{i}_L" for i in range(1, 8)] + [f"Joint{i}_R" for i in range(1, 8)])


def _generate_path(rng: np.random.Generator, mode: str) -> np.ndarray:
    start = rng.uniform(-0.25, 0.25, 14)
    goal = start + rng.uniform(-0.45, 0.45, 14)
    if mode == "left_only":
        goal[7:] = start[7:]
    elif mode == "right_only":
        goal[:7] = start[:7]
    return np.linspace(start, goal, 64, dtype=np.float32)


def generate(config: dict) -> Path:
    if config.get("robot_model") != "marvin_bimanual":
        raise ValueError("robot_model must be marvin_bimanual")
    mode = "cooperative_rigid" if config.get("task_family") == "cooperative" else config.get("task_mode", "dual_independent")
    names = tuple(config.get("joint_names", JOINT_NAMES))
    if names != JOINT_NAMES:
        raise ValueError("joint_names must use the canonical left-then-right order")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required to generate Marvin datasets") from error
    rng = np.random.default_rng(int(config.get("seed", 0)))
    paths = np.stack([_generate_path(rng, mode) for _ in range(int(config.get("num_trajectories", 1)))])
    with h5py.File(output / "dataset_merged.hdf5", "w") as handle:
        handle.create_dataset("sol_path", data=paths, compression="gzip")
        handle.create_dataset("q_start", data=paths[:, 0])
        handle.create_dataset("q_goal", data=paths[:, -1])
        handle.create_dataset("task_id", data=np.arange(len(paths), dtype=np.int64))
        handle.create_dataset("task_mode", data=np.asarray([mode.encode()] * len(paths)))
        active = np.ones((len(paths), 14), dtype=np.bool_)
        if mode == "left_only":
            active[:, 7:] = False
        elif mode == "right_only":
            active[:, :7] = False
        handle.create_dataset("active_joint_mask", data=active)
        handle.create_dataset("min_clearance", data=np.full(len(paths), np.inf, dtype=np.float32))
        handle.create_dataset("closure_error", data=np.zeros((len(paths), 2), dtype=np.float32))
        handle.create_dataset("object_start_pose", data=np.zeros((len(paths), 7), dtype=np.float32))
        handle.create_dataset("object_goal_pose", data=np.zeros((len(paths), 7), dtype=np.float32))
        grasp_left = np.eye(4, dtype=np.float32)
        grasp_right = np.eye(4, dtype=np.float32)
        grasp_left[1, 3] = 0.12
        grasp_right[1, 3] = -0.12
        handle.create_dataset("T_object_left_grasp", data=np.broadcast_to(grasp_left, (len(paths), 4, 4)))
        handle.create_dataset("T_object_right_grasp", data=np.broadcast_to(grasp_right, (len(paths), 4, 4)))
        handle.create_dataset("scene_id", data=np.asarray([b"synthetic_table"] * len(paths)))
        handle.create_dataset("scene_hash", data=np.asarray([b"synthetic_table_v1"] * len(paths)))
        handle.create_dataset("generator_version", data=np.asarray([b"marvin_bimanual_generator_v1"] * len(paths)))
    manifest = {"schema": "marvin_bimanual_dataset/v1", "robot_model": "marvin_bimanual", "joint_names": list(names), "task_mode": mode, "num_trajectories": len(paths), "dataset_sha256": hashlib.sha256((output / "dataset_merged.hdf5").read_bytes()).hexdigest()}
    (output / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (output / "args.yaml").write_text(yaml.safe_dump({"env_id": "EnvMarvinTable", "robot_id": "RobotMarvinBimanual", "task_mode": mode, "joint_names": list(names)}, sort_keys=False))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    print(generate(yaml.safe_load(args.config.read_text()) or {}))


if __name__ == "__main__":
    main()
