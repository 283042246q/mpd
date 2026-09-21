"""Cooperative warehouse endpoint sources for the independent-prior runtime."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mpd.bimanual.runtime_contract import JOINT_NAMES, SCHEMA


REGION_SCHEMA = "marvin_bimanual_cooperative_regions/v1"
SEED_SCHEMA = "marvin_bimanual_cooperative_seeds/v1"
SOURCE_NAMES = ("regions", "seed_file")


def _pose(transform) -> dict[str, Any]:
    from scipy.spatial.transform import Rotation

    transform = np.asarray(transform, dtype=np.float64)
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return {
        "frame_id": "world",
        "pose_xyzw": [*transform[:3, 3].tolist(), *quaternion.tolist()],
    }


def _request(
    *,
    request_id,
    seed,
    q_start,
    q_goal,
    object_start,
    object_goal,
    base_rotation,
    object_to_left,
    object_to_right,
    source,
    grasp_profile="default_box",
):
    from scripts.generate_data.generate_marvin_warehouse_cooperative import (
        object_transform,
    )

    start_transform = object_transform(object_start, base_rotation)
    goal_transform = object_transform(object_goal, base_rotation)
    return {
        "schema": SCHEMA,
        "request_id": str(request_id),
        "task_mode": "cooperative_rigid",
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
        "left_goal_pose": _pose(goal_transform @ object_to_left),
        "right_goal_pose": _pose(goal_transform @ object_to_right),
        "object_goal_pose": _pose(goal_transform),
        "grasp_profile": grasp_profile,
        "world_version": 0,
        "deadline_unix_ns": 0,
        "seed": int(seed),
        "scene": {
            "start_goal_source": source,
            "object_start_state_xyz_yaw": np.asarray(object_start).tolist(),
            "object_goal_state_xyz_yaw": np.asarray(object_goal).tolist(),
        },
    }


def _load_generator_config(path: Path):
    from scripts.generate_data.generate_marvin_warehouse_cooperative import (
        validate_config,
    )

    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def request_from_regions(
    path: Path,
    *,
    generator_config_path: Path,
    request_id: str,
    seed: int,
):
    """Sample a deterministic cooperative endpoint pair from an explicit matrix."""

    from scripts.generate_data.generate_marvin_warehouse_cooperative import (
        MarvinWarehouseCooperativeGenerator,
        validate_config,
    )

    region_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(region_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != REGION_SCHEMA:
        raise ValueError(f"cooperative regions schema must be {REGION_SCHEMA}")
    regions = payload.get("object_regions")
    matrix = payload.get("sampling_matrix")
    if not isinstance(regions, dict) or not regions:
        raise ValueError("cooperative object_regions must be a non-empty mapping")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("cooperative sampling_matrix must be a non-empty list")
    weights = np.asarray([float(item.get("weight", 1.0)) for item in matrix])
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("cooperative sampling_matrix weights must be finite and positive")
    rng = np.random.default_rng(int(seed))
    matrix_index = int(rng.choice(len(matrix), p=weights / weights.sum()))
    pair = matrix[matrix_index]
    start_name, goal_name = pair.get("start"), pair.get("goal")
    if start_name not in regions or goal_name not in regions:
        raise ValueError("sampling_matrix references an unknown cooperative region")

    config = deepcopy(_load_generator_config(generator_config_path))
    config["object_regions"] = {
        "start": deepcopy(regions[start_name]),
        "goal": deepcopy(regions[goal_name]),
    }
    # Endpoint/source selection should remain bounded independently from the
    # later MPD deadline.
    source_config = payload.get("sampling", {})
    config["dataset"]["context_sampling_tries"] = int(
        source_config.get("context_sampling_tries", 30)
    )
    validate_config(config)
    generator = MarvinWarehouseCooperativeGenerator(config, int(seed))
    try:
        context = generator.sample_context()
        if context is None:
            raise RuntimeError(
                f"no paired endpoint IK for cooperative matrix row {matrix_index} "
                f"({start_name}->{goal_name})"
            )
        object_start, object_goal, q_start, q_goal = context
        source = {
            "type": "regions",
            "path": str(region_path),
            "sampling": "deterministic_seed",
            "matrix_index": matrix_index,
            "start_region": start_name,
            "goal_region": goal_name,
        }
        return _request(
            request_id=request_id,
            seed=seed,
            q_start=q_start,
            q_goal=q_goal,
            object_start=object_start,
            object_goal=object_goal,
            base_rotation=generator.base_rotation,
            object_to_left=generator.object_to_left,
            object_to_right=generator.object_to_right,
            source=source,
            grasp_profile=config["grasp"]["profile"],
        )
    finally:
        generator.close()


def request_from_seed_file(
    path: Path,
    *,
    generator_config_path: Path,
    request_id: str,
    seed: int,
    sample_index: int,
):
    """Load fixed, previously collision-audited cooperative endpoints."""

    from scripts.generate_data.generate_marvin_warehouse_cooperative import (
        GRASP_PROFILES,
        _transform,
    )

    seed_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SEED_SCHEMA:
        raise ValueError(f"cooperative seed schema must be {SEED_SCHEMA}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("cooperative seed file must contain samples")
    index = int(seed) % len(samples) if sample_index < 0 else int(sample_index)
    if not 0 <= index < len(samples):
        raise IndexError(f"sample_index {index} exceeds {len(samples)} cooperative seeds")
    sample = samples[index]
    object_start = np.asarray(sample.get("object_start_xyz_yaw"), dtype=float)
    object_goal = np.asarray(sample.get("object_goal_xyz_yaw"), dtype=float)
    q_start = np.asarray(sample.get("q_start"), dtype=float)
    q_goal = np.asarray(sample.get("q_goal"), dtype=float)
    if object_start.shape != (4,) or object_goal.shape != (4,):
        raise ValueError("fixed cooperative object endpoints must be xyz_yaw vectors")
    if q_start.shape != (14,) or q_goal.shape != (14,):
        raise ValueError("fixed cooperative joint endpoints must contain 14 values")

    config = _load_generator_config(generator_config_path)
    profile_name = config["grasp"]["profile"]
    profiles = yaml.safe_load(GRASP_PROFILES.read_text(encoding="utf-8"))
    profile = profiles[profile_name]
    source = {
        "type": "seed_file",
        "path": str(seed_path),
        "index": index,
        "name": sample.get("name"),
    }
    return _request(
        request_id=request_id,
        seed=seed,
        q_start=q_start,
        q_goal=q_goal,
        object_start=object_start,
        object_goal=object_goal,
        base_rotation=np.asarray(config["object_planning"]["base_rotation"]),
        object_to_left=_transform(
            profile["left_grasp"]["xyz"], profile["left_grasp"]["rpy"]
        ),
        object_to_right=_transform(
            profile["right_grasp"]["xyz"], profile["right_grasp"]["rpy"]
        ),
        source=source,
        grasp_profile=profile_name,
    )


def request_from_config_source(
    config_path: Path,
    *,
    source: str | None,
    source_path: Path | None,
    sample_index: int,
    seed: int,
    request_id: str,
):
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = str(source or config.get("start_goal_source", "regions")).lower()
    if selected not in SOURCE_NAMES:
        raise ValueError(f"cooperative start_goal_source must be one of {SOURCE_NAMES}")
    generator_config = config.get("cooperative_generation_config")
    if not generator_config:
        raise ValueError("cooperative_generation_config is required")
    generator_config_path = (config_path.parent / generator_config).resolve()
    configured_key = (
        "cooperative_regions_path" if selected == "regions" else "cooperative_seed_file_path"
    )
    selected_path = source_path or config.get(configured_key)
    if not selected_path:
        raise ValueError(f"{configured_key} is required")
    resolved = (
        Path(source_path).expanduser().resolve()
        if source_path is not None
        else (config_path.parent / selected_path).resolve()
    )
    if selected == "regions":
        return request_from_regions(
            resolved,
            generator_config_path=generator_config_path,
            request_id=request_id,
            seed=seed,
        )
    return request_from_seed_file(
        resolved,
        generator_config_path=generator_config_path,
        request_id=request_id,
        seed=seed,
        sample_index=sample_index,
    )
