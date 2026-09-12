"""Build strict Marvin runtime requests from offline inference sources."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from mpd.bimanual.runtime_contract import JOINT_NAMES, SCHEMA
from mpd.paths import DATASET_BASE_DIR


SOURCE_NAMES = ("dataset", "states_file", "regions")


class StartGoalSamplingError(RuntimeError):
    """No collision-valid start/goal pair could be sampled from regions."""


def resolve_source_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config_path).expanduser().resolve().parent / path).resolve()


def _base_request(*, request_id: str, seed: int, q_start, q_goal, source: dict):
    return {
        "schema": SCHEMA,
        "request_id": request_id,
        "task_mode": "dual_independent",
        "runtime_mode": "snapshot_no_time",
        "robot_model": "marvin_bimanual",
        "planning_frame": "world",
        "scene_id": "EnvWarehouseMarvinBimanual",
        "scene_version": "marvin_warehouse_v2",
        "joint_names": list(JOINT_NAMES),
        "q_start": np.asarray(q_start, dtype=np.float64).tolist(),
        "q_goal": np.asarray(q_goal, dtype=np.float64).tolist(),
        "q_velocity_start": [0.0] * 14,
        "q_acceleration_start": [0.0] * 14,
        "world_version": 0,
        "deadline_unix_ns": 0,
        "seed": int(seed),
        "scene": {"start_goal_source": source},
    }


def _pose_from_matrix(matrix) -> dict[str, Any]:
    from scipy.spatial.transform import Rotation

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 4):
        raise ValueError(f"EE pose matrix must have shape [3,4], got {matrix.shape}")
    quaternion = Rotation.from_matrix(matrix[:, :3]).as_quat()
    return {
        "frame_id": "world",
        "pose_xyzw": [*matrix[:, 3].tolist(), *quaternion.tolist()],
    }


def request_from_dataset(
    config: dict,
    *,
    request_id: str,
    seed: int,
    sample_index: int,
) -> dict:
    import h5py

    dataset_path = (
        Path(DATASET_BASE_DIR)
        / str(config["dataset_subdir"])
        / str(config.get("dataset_file_merged", "dataset_merged.hdf5"))
    )
    with h5py.File(dataset_path, "r") as dataset:
        required = ("q_start", "q_goal", "ee_goal_pose", "task_mode")
        missing = [key for key in required if key not in dataset]
        if missing:
            raise ValueError(f"dataset is missing fields: {missing}")
        modes = np.asarray(dataset["task_mode"][:]).astype("U")
        eligible = np.flatnonzero(modes == "dual_independent")
        if not eligible.size:
            raise ValueError("dataset contains no dual_independent samples")
        if sample_index < 0:
            row = int(eligible[int(seed) % len(eligible)])
        else:
            if sample_index >= len(eligible):
                raise IndexError(f"sample_index {sample_index} exceeds {len(eligible)} eligible rows")
            row = int(eligible[sample_index])
        request = _base_request(
            request_id=request_id,
            seed=seed,
            q_start=dataset["q_start"][row],
            q_goal=dataset["q_goal"][row],
            source={"type": "dataset", "path": str(dataset_path), "row": row},
        )
        goals = np.asarray(dataset["ee_goal_pose"][row])
        request["left_goal_pose"] = _pose_from_matrix(goals[0])
        request["right_goal_pose"] = _pose_from_matrix(goals[1])
        return request


def request_from_states_file(
    path: Path,
    *,
    request_id: str,
    seed: int,
    sample_index: int,
) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "marvin_bimanual_states/v1":
        raise ValueError("states file schema must be marvin_bimanual_states/v1")
    if payload.get("planning_frame", "world") != "world":
        raise ValueError("states file planning_frame must be world")
    if tuple(payload.get("joint_names", JOINT_NAMES)) != JOINT_NAMES:
        raise ValueError("states file joint_names must use canonical left-then-right order")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("states file samples must be a non-empty list")
    index = int(seed) % len(samples) if sample_index < 0 else sample_index
    if not 0 <= index < len(samples):
        raise IndexError(f"sample_index {index} exceeds {len(samples)} states")
    sample = samples[index]
    if not isinstance(sample, dict):
        raise ValueError("each states sample must be an object")
    request = _base_request(
        request_id=request_id,
        seed=seed,
        q_start=sample.get("q_start"),
        q_goal=sample.get("q_goal"),
        source={
            "type": "states_file",
            "path": str(Path(path).resolve()),
            "index": index,
            "name": sample.get("name"),
        },
    )
    for key in ("left_goal_pose", "right_goal_pose"):
        if sample.get(key) is not None:
            request[key] = sample[key]
    return request


def _choose_region(value, rng, *, field):
    choices = [value] if isinstance(value, str) else value
    if not isinstance(choices, list) or not choices or not all(isinstance(item, str) and item for item in choices):
        raise ValueError(f"{field} must be a region name or non-empty name list")
    return choices[int(rng.integers(len(choices)))]


def request_from_regions(
    path: Path,
    *,
    request_id: str,
    seed: int,
    sample_index: int,
) -> dict:
    from scripts.generate_data.generate_marvin_warehouse_bimanual import (
        MarvinWarehouseGenerator,
    )

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != "marvin_bimanual_regions/v1":
        raise ValueError("regions file schema must be marvin_bimanual_regions/v1")
    selection = config.get("inference_selection", {})
    start_selection = selection.get("start", {"left": "random", "right": "random"})
    goal_selection = selection.get("goal")
    if not isinstance(start_selection, dict):
        raise ValueError("inference_selection.start must define left and right selections")
    if not isinstance(goal_selection, dict):
        raise ValueError("inference_selection.goal must define left and right regions")
    # Region endpoints are intentionally fresh samples on every invocation.
    # ``seed`` remains part of the runtime request for MPD's diffusion sampler,
    # while ``sample_index`` only addresses dataset/states-file entries.
    generator = MarvinWarehouseGenerator(config, None, progress_label="inference-regions")
    generator.deadline = time.perf_counter() + float(config.get("sampling_timeout_seconds", 60.0))
    try:
        attempts = int(config.get("max_sampling_attempts", 100))
        for attempt in range(attempts):
            q_start = generator._random_valid_state()
            if q_start is None:
                continue
            start_regions = {
                arm: _choose_region(start_selection.get(arm, "random"), generator.rng, field=f"start.{arm}")
                for arm in ("left", "right")
            }
            named_start = {arm: name for arm, name in start_regions.items() if name != "random"}
            for arm, region_name in named_start.items():
                q_start = generator._target_state(q_start, arm, region_name)
                if q_start is None:
                    break
            if q_start is None or not generator.valid(q_start):
                continue
            goal_regions = {
                arm: _choose_region(goal_selection.get(arm), generator.rng, field=f"goal.{arm}")
                for arm in ("left", "right")
            }
            q_goal = generator._sample_endpoint(q_start, "dual_independent", goal_regions)
            if q_goal is None:
                continue
            minimum_delta = float(config.get("min_active_joint_delta", 0.08))
            if any(
                np.linalg.norm(q_goal[arm_slice] - q_start[arm_slice]) < minimum_delta
                for arm_slice in (slice(0, 7), slice(7, 14))
            ):
                continue
            goal_matrices = []
            for arm in ("left", "right"):
                pose = generator._pose(q_goal, arm)
                goal_matrices.append(np.c_[pose.rotation, pose.translation])
            request = _base_request(
                request_id=request_id,
                seed=seed,
                q_start=q_start,
                q_goal=q_goal,
                source={
                    "type": "regions",
                    "path": str(Path(path).resolve()),
                    "sampling": "system_entropy",
                    "attempt": attempt + 1,
                    "start": start_regions,
                    "goal": goal_regions,
                },
            )
            request["left_goal_pose"] = _pose_from_matrix(goal_matrices[0])
            request["right_goal_pose"] = _pose_from_matrix(goal_matrices[1])
            return request
        raise StartGoalSamplingError(f"could not sample a valid dual-arm region pair in {attempts} attempts")
    finally:
        generator.close()


def request_from_config_source(
    config_path: Path,
    *,
    source: str | None,
    source_path: Path | None,
    sample_index: int,
    seed: int,
    request_id: str,
) -> dict:
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = str(source or config.get("start_goal_source", "dataset")).lower()
    if selected == "auto":
        selected = "dataset"
    if selected not in SOURCE_NAMES:
        raise ValueError(f"start_goal_source must be one of {SOURCE_NAMES} or auto")
    if selected == "dataset":
        return request_from_dataset(config, request_id=request_id, seed=seed, sample_index=sample_index)
    configured_key = "start_goal_states_path" if selected == "states_file" else "start_goal_regions_path"
    configured_path = source_path or config.get(configured_key)
    if not configured_path:
        raise ValueError(f"{configured_key} is required for {selected}")
    path = (
        Path(source_path).expanduser().resolve()
        if source_path is not None
        else resolve_source_path(config_path, configured_path)
    )
    if selected == "states_file":
        return request_from_states_file(path, request_id=request_id, seed=seed, sample_index=sample_index)
    return request_from_regions(path, request_id=request_id, seed=seed, sample_index=sample_index)
