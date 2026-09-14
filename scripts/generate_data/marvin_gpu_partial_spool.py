"""Durable per-task staging for incomplete Marvin GPU shards."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import yaml


SPOOL_SCHEMA = "marvin_gpu_partial_task/v1"


def _config_sha256(config):
    payload = yaml.safe_dump(dict(config), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_yaml(path, value):
    path = Path(path)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    with temporary.open("x") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


class PartialTaskSpool:
    def __init__(self, root, config):
        self.root = Path(root) / ".inflight"
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_sha256 = _config_sha256(config)

    def _directory(self, start):
        return self.root / f"{int(start):09d}"

    def prepare_shard(self, start, size):
        directory = self._directory(start)
        directory.mkdir(parents=True, exist_ok=True)
        contract_path = directory / "contract.yaml"
        expected = {
            "schema": SPOOL_SCHEMA,
            "config_sha256": self.config_sha256,
            "start": int(start),
            "size": int(size),
        }
        if contract_path.exists():
            saved = yaml.safe_load(contract_path.read_text())
            if saved != expected:
                raise ValueError(f"partial spool contract differs: {directory}")
        else:
            _atomic_yaml(contract_path, expected)
        return directory

    def save_task(self, task, shard_size=10):
        if not task.accepted:
            raise ValueError(f"task {task.task_id} is not accepted")
        directory = self.prepare_shard(task.shard_start, shard_size)
        path = directory / f"task-{task.task_id:09d}.npz"
        temporary = directory / f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
        metadata = dict(task.accepted_metadata)
        knots, coefficients, degree = metadata.pop("bspline")
        strings = {
            key: metadata.pop(key)
            for key in (
                "task_id",
                "task_mode",
                "direction",
                "source_region_left",
                "source_region_right",
                "goal_region_left",
                "goal_region_right",
            )
        }
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                path=np.asarray(task.accepted_path),
                q_start=np.asarray(metadata["q_start"]),
                q_goal=np.asarray(metadata["q_goal"]),
                planning_time=np.asarray(metadata["planning_time"]),
                ee_goal_pose=np.asarray(metadata["ee_goal_pose"]),
                joint_path_length=np.asarray(metadata["joint_path_length"]),
                bspline_tt=np.asarray(knots),
                bspline_cc=np.asarray(coefficients),
                bspline_k=np.asarray(degree),
                strings=np.frombuffer(json.dumps(strings).encode(), dtype=np.uint8),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(directory)

    def save_stats(self, shard):
        directory = self.prepare_shard(shard.start, shard.size)
        _atomic_yaml(directory / "stats.yaml", dict(shard.stats))

    def restore(self, shard):
        directory = self.prepare_shard(shard.start, shard.size)
        stats_path = directory / "stats.yaml"
        if stats_path.exists():
            shard.stats = Counter(yaml.safe_load(stats_path.read_text()) or {})
        restored = []
        for path in sorted(directory.glob("task-*.npz")):
            with np.load(path, allow_pickle=False) as saved:
                strings = json.loads(saved["strings"].tobytes().decode())
                task_id = int(strings["task_id"])
                if task_id not in shard.tasks:
                    raise ValueError(f"spooled task {task_id} is outside shard {shard.start}")
                task = shard.tasks[task_id]
                definition = task.definition
                expected = {
                    "task_id": definition.task_id,
                    "task_mode": definition.mode,
                    "direction": definition.direction,
                    "source_region_left": definition.source["left"],
                    "source_region_right": definition.source["right"],
                    "goal_region_left": definition.goal["left"],
                    "goal_region_right": definition.goal["right"],
                }
                if strings != expected:
                    raise ValueError(f"spooled task contract differs: {path}")
                trajectory = np.asarray(saved["path"])
                metadata = {
                    **strings,
                    "q_start": np.asarray(saved["q_start"]),
                    "q_goal": np.asarray(saved["q_goal"]),
                    "planning_time": float(saved["planning_time"]),
                    "ee_goal_pose": np.asarray(saved["ee_goal_pose"]),
                    "joint_path_length": float(saved["joint_path_length"]),
                    "bspline": (
                        np.asarray(saved["bspline_tt"]),
                        np.asarray(saved["bspline_cc"]),
                        int(saved["bspline_k"]),
                    ),
                }
            if trajectory.ndim != 2 or trajectory.shape[1] != 14:
                raise ValueError(f"invalid spooled trajectory shape: {path}")
            if not np.isfinite(trajectory).all():
                raise ValueError(f"nonfinite spooled trajectory: {path}")
            task.accept(trajectory, metadata)
            restored.append(task_id)
        shard.stats["accepted"] = len(restored)
        for mode in ("dual_independent", "left_only", "right_only"):
            shard.stats[f"accepted/{mode}"] = sum(
                shard.tasks[task_id].definition.mode == mode for task_id in restored
            )
        return restored

    def clear_shard(self, start):
        directory = self._directory(start)
        if directory.exists():
            shutil.rmtree(directory)
            _fsync_directory(self.root)
