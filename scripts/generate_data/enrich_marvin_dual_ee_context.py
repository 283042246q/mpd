#!/usr/bin/env python3
"""Atomically add dual Pika TCP goals and activity masks to Marvin shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
import torch
import yaml

from scripts.generate_data.generate_marvin_warehouse_bimanual import (
    ARM_SLICES,
    EE_GOAL_LINKS,
    EE_GOAL_SCHEMA,
    active_arms,
    file_sha256,
)
from torch_robotics.robots import RobotMarvinBimanual


def _atomic_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _expected_masks(task_modes):
    ee_mask = np.zeros((len(task_modes), 2), dtype=bool)
    joint_mask = np.zeros((len(task_modes), 14), dtype=bool)
    for index, mode in enumerate(task_modes):
        for arm in active_arms(mode):
            slot = 0 if arm == "left" else 1
            ee_mask[index, slot] = True
            joint_mask[index, ARM_SLICES[arm]] = True
    return ee_mask, joint_mask


def _dual_fk(robot, q_goal):
    q = torch.as_tensor(q_goal, dtype=robot.q_pos_min.dtype, device=robot.q_pos_min.device)
    with torch.no_grad():
        pose = torch.stack((robot.fk_left(q), robot.fk_right(q)), dim=1)
    pose = pose.detach().cpu().numpy().astype(np.float32, copy=False)
    if pose.shape != (len(q_goal), 2, 3, 4) or not np.isfinite(pose).all():
        raise ValueError(f"invalid TorchKin output shape/content: {pose.shape}")
    rotations = pose[..., :3]
    if not np.allclose(np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=2e-4):
        raise ValueError("TorchKin returned a non-orthonormal rotation")
    return pose


def enrich_shard(shard: Path, robot: RobotMarvinBimanual, dry_run=False) -> str:
    dataset_path = shard / "dataset_merged.hdf5"
    manifest_path = shard / "manifest.yaml"
    if not dataset_path.is_file() or not manifest_path.is_file() or dataset_path.stat().st_size == 0:
        return "incomplete"
    try:
        manifest = yaml.safe_load(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return "incomplete"
    if not isinstance(manifest, dict) or manifest.get("dataset_sha256") != file_sha256(dataset_path):
        return "hash_mismatch"

    with h5py.File(dataset_path, "r") as data:
        n = len(data["q_goal"])
        if n == 0 or data["q_goal"].shape != (n, 14) or data["bspline_params_cc"].shape[:2] != (n, 14):
            return "incomplete"
        already_enriched = (
            "ee_goal_pose" in data
            and "active_ee_mask" in data
            and data["ee_goal_pose"].shape == (n, 2, 3, 4)
            and data["active_ee_mask"].shape == (n, 2)
            and data.attrs.get("ee_goal_schema") == EE_GOAL_SCHEMA
        )
        if already_enriched:
            return "already_enriched"
        if "ee_goal_pose" in data or "active_ee_mask" in data:
            return "partial_fields"
        q_goal = data["q_goal"][:]
        if not np.allclose(q_goal, data["bspline_params_cc"][:, :, -1], atol=1e-7):
            raise ValueError(f"{shard}: q_goal differs from the validated spline endpoint")
        task_modes = data["task_mode"].asstr()[:]
        ee_mask, joint_mask = _expected_masks(task_modes)
        if not np.array_equal(joint_mask, data["active_joint_mask"][:]):
            raise ValueError(f"{shard}: task_mode and active_joint_mask disagree")

    pose = _dual_fk(robot, q_goal)
    if dry_run:
        return "would_enrich"

    temporary = shard / f".dataset_merged.ee-context-{os.getpid()}.hdf5"
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        shutil.copy2(dataset_path, temporary)
        with h5py.File(temporary, "r+") as data:
            data.create_dataset("ee_goal_pose", data=pose, compression="gzip")
            data.create_dataset("active_ee_mask", data=ee_mask)
            data.attrs["ee_goal_schema"] = EE_GOAL_SCHEMA
            data.attrs["ee_goal_links"] = np.asarray(EE_GOAL_LINKS, dtype=object)
            data.flush()
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, dataset_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    manifest.update(
        schema="marvin_bimanual_warehouse_dataset/v3",
        ee_goal_schema=EE_GOAL_SCHEMA,
        ee_goal_links=list(EE_GOAL_LINKS),
        dataset_sha256=file_sha256(dataset_path),
    )
    _atomic_text(manifest_path, yaml.safe_dump(manifest, sort_keys=False))
    _atomic_text(shard / "generation_summary.json", json.dumps(manifest, indent=2))
    return "enriched"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    shard_root = args.root / "shards"
    shards = sorted(path for path in shard_root.iterdir() if path.is_dir() and path.name.isdigit())
    if args.limit is not None:
        shards = shards[: args.limit]
    robot = RobotMarvinBimanual(tensor_args={"device": "cpu", "dtype": torch.float32})
    counts = {}
    for index, shard in enumerate(shards, 1):
        status = enrich_shard(shard, robot, dry_run=args.dry_run)
        counts[status] = counts.get(status, 0) + 1
        if status not in {"enriched", "already_enriched", "would_enrich"}:
            print(f"[{shard.name}] skipped: {status}", flush=True)
        elif index % 10 == 0 or index == len(shards):
            print(f"processed {index}/{len(shards)}: {counts}", flush=True)
    print(json.dumps({"root": str(args.root), "shards_seen": len(shards), **counts}, indent=2))
    return 1 if any(counts.get(key, 0) for key in ("hash_mismatch", "partial_fields")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
