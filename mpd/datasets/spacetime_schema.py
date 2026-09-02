"""Canonical data contracts for Space-Time MPD training data.

This module is intentionally independent from the legacy trajectory dataset
loader.  It provides the immutable robot identity and field names shared by
F1/F2/F3 and JointDual datasets without changing the existing spatial MPD
training path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
from pathlib import Path
import shutil
from typing import Dict, Iterable, Mapping, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import yaml


SCHEMA_VERSION = "spacetime_mpd_v1"
TIMING_REPRESENTATION = "dt_ds_softplus_v1"


class SpatialSource(IntEnum):
    """Values stored in ``/source/spatial``."""

    UNKNOWN = 0
    RRT_CONNECT = 1
    DIFFUSION = 2
    REPAIRED = 3


class TimingSource(IntEnum):
    """Values stored in ``/source/timing``."""

    TOPPRA = 1
    DURATION_SCALED = 2
    LIMIT_SCALED_TOPPRA = 3
    LOCAL_SLOWDOWN = 4
    NEAR_WAIT = 5
    DYNAMIC_OPTIMIZED = 6


CANONICAL_SAMPLE_FIELDS: Mapping[str, Tuple[np.dtype, Tuple[str, ...]]] = {
    "spatial/control_points": (np.dtype("float32"), ("H_P", "D")),
    "timing/control_points": (np.dtype("float32"), ("K",)),
    "timing/duration": (np.dtype("float32"), ()),
    "timing/reference_time": (np.dtype("float32"), ("H_T",)),
    "condition/q_start": (np.dtype("float32"), ("D",)),
    "condition/q_goal": (np.dtype("float32"), ("D",)),
    "index/task_id": (np.dtype("int64"), ()),
    "index/base_path_id": (np.dtype("int64"), ()),
    "index/variant_id": (np.dtype("int16"), ()),
    "index/scene_id": (np.dtype("int64"), ()),
    "index/mode_id": (np.dtype("int16"), ()),
    "source/spatial": (np.dtype("uint8"), ()),
    "source/timing": (np.dtype("uint8"), ()),
    "source/parent_sample_id": (np.dtype("int64"), ()),
    "quality/spatial_fit_rmse": (np.dtype("float32"), ()),
    "quality/timing_fit_rmse": (np.dtype("float32"), ()),
    "quality/v_ratio_max": (np.dtype("float32"), ()),
    "quality/a_ratio_max": (np.dtype("float32"), ()),
    "quality/static_clearance_min": (np.dtype("float32"), ()),
    "quality/dynamic_clearance_min": (np.dtype("float32"), ()),
    "quality/accepted": (np.dtype("bool"), ()),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _active_urdf_joints(urdf_path: Path) -> Tuple[Tuple[str, ...], np.ndarray, np.ndarray]:
    root = ET.parse(str(urdf_path)).getroot()
    names = []
    q_min = []
    q_max = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise ValueError(f"active URDF joint {joint.attrib.get('name')} has no finite position limits")
        names.append(joint.attrib["name"])
        q_min.append(float(limit.attrib["lower"]))
        q_max.append(float(limit.attrib["upper"]))
    if not names:
        raise ValueError(f"no active joints found in {urdf_path}")
    return tuple(names), np.asarray(q_min, dtype=np.float64), np.asarray(q_max, dtype=np.float64)


def _aligned_dynamic_limits(
    joint_limits_path: Path, joint_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    with joint_limits_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"joint limits file must contain a mapping: {joint_limits_path}")
    missing = [name for name in joint_names if name not in raw]
    extra = [name for name in raw if name not in joint_names]
    if missing or extra:
        raise ValueError(f"joint limit names do not match URDF; missing={missing}, extra={extra}")
    dq_max = np.asarray([raw[name]["qdot_max"] for name in joint_names], dtype=np.float64)
    ddq_max = np.asarray([raw[name]["qddot_max"] for name in joint_names], dtype=np.float64)
    if not np.all(np.isfinite(dq_max)) or not np.all(dq_max > 0):
        raise ValueError("velocity limits must be positive and finite")
    if not np.all(np.isfinite(ddq_max)) or not np.all(ddq_max > 0):
        raise ValueError("acceleration limits must be positive and finite")
    return dq_max, ddq_max


@dataclass(frozen=True)
class RobotBundle:
    """Resolved, joint-name-aligned identity used by generation and training."""

    name: str
    urdf_path: Path
    joint_limits_path: Path
    collision_spheres_path: Path
    collision_parent_bounds_path: Path
    base_link: str
    ee_link: str
    active_joint_names: Tuple[str, ...]
    q_min: np.ndarray
    q_max: np.ndarray
    dq_max: np.ndarray
    ddq_max: np.ndarray

    @property
    def dof(self) -> int:
        return len(self.active_joint_names)

    @property
    def hashes(self) -> Dict[str, str]:
        return {
            "urdf_sha256": sha256_file(self.urdf_path),
            "joint_limits_sha256": sha256_file(self.joint_limits_path),
            "collision_spheres_sha256": sha256_file(self.collision_spheres_path),
            "collision_parent_bounds_sha256": sha256_file(self.collision_parent_bounds_path),
        }

    def validate(self) -> None:
        expected = (self.dof,)
        for name, value in (
            ("q_min", self.q_min),
            ("q_max", self.q_max),
            ("dq_max", self.dq_max),
            ("ddq_max", self.ddq_max),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN or Inf")
        if not np.all(self.q_min < self.q_max):
            raise ValueError("position lower limits must be smaller than upper limits")

    def manifest_robot_section(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "name": self.name,
            "urdf": "robot/robot.urdf",
            "joint_limits": "robot/joint_limits.yaml",
            "collision_spheres": "robot/collision_spheres.yaml",
            "collision_parent_bounds": "robot/collision_parent_bounds.yaml",
            "base_link": self.base_link,
            "ee_link": self.ee_link,
            "active_joint_names": list(self.active_joint_names),
            "dof": self.dof,
        }
        result.update(self.hashes)
        return result


def load_panda_robot_bundle(repository_root: Path) -> RobotBundle:
    """Resolve the exact Panda assets currently used by :class:`RobotPanda`."""

    data_root = repository_root / "mpd" / "torch_robotics" / "torch_robotics" / "data"
    urdf = (
        data_root
        / "urdf"
        / "robots"
        / "franka_description"
        / "robots"
        / "panda_arm_hand_no_gripper.urdf"
    )
    config_root = data_root / "configs" / "panda"
    joint_limits = config_root / "joint_limits.yaml"
    collision_spheres = config_root / "panda_sphere_config.yaml"
    collision_parent_bounds = config_root / "panda_parent_collision_bounds.yaml"
    for path in (urdf, joint_limits, collision_spheres, collision_parent_bounds):
        if not path.is_file():
            raise FileNotFoundError(path)

    joint_names, q_min, q_max = _active_urdf_joints(urdf)
    dq_max, ddq_max = _aligned_dynamic_limits(joint_limits, joint_names)
    bundle = RobotBundle(
        name="RobotPanda",
        urdf_path=urdf,
        joint_limits_path=joint_limits,
        collision_spheres_path=collision_spheres,
        collision_parent_bounds_path=collision_parent_bounds,
        base_link="panda_link0",
        ee_link="panda_hand",
        active_joint_names=joint_names,
        q_min=q_min,
        q_max=q_max,
        dq_max=dq_max,
        ddq_max=ddq_max,
    )
    bundle.validate()
    return bundle


def build_manifest(
    robot: RobotBundle,
    *,
    spatial_num_control_points: int,
    spatial_degree: int,
    spatial_num_phase_points: int,
    timing_num_control_points: int,
    timing_degree: int,
    timing_num_phase_points: int,
    timing_u_min: float,
    timing_duration_min: float,
    timing_duration_max: float,
    source_dataset: str,
    variants: Iterable[Mapping[str, object]],
) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "robot": robot.manifest_robot_section(),
        "units": {
            "joint_position": "rad",
            "joint_velocity": "rad/s",
            "joint_acceleration": "rad/s^2",
            "time": "s",
        },
        "spatial_spline": {
            "degree": int(spatial_degree),
            "num_control_points": int(spatial_num_control_points),
            "num_phase_points": int(spatial_num_phase_points),
            "zero_velocity_at_endpoints": True,
            "zero_acceleration_at_endpoints": True,
        },
        "timing_spline": {
            "representation": TIMING_REPRESENTATION,
            "degree": int(timing_degree),
            "num_control_points": int(timing_num_control_points),
            "num_phase_points": int(timing_num_phase_points),
            "u_min": float(timing_u_min),
            "duration_min": float(timing_duration_min),
            "duration_max": float(timing_duration_max),
        },
        "source": {
            "dataset": source_dataset,
            "spatial_planner": "RRTConnect",
        },
        "variants": list(variants),
    }


def materialize_dataset_contract(
    output_root: Path, robot: RobotBundle, manifest: Mapping[str, object]
) -> None:
    """Write the immutable manifest and canonical robot files."""

    output_root.mkdir(parents=True, exist_ok=True)
    robot_root = output_root / "robot"
    robot_root.mkdir(exist_ok=True)
    copies = (
        (robot.urdf_path, robot_root / "robot.urdf"),
        (robot.joint_limits_path, robot_root / "joint_limits.yaml"),
        (robot.collision_spheres_path, robot_root / "collision_spheres.yaml"),
        (robot.collision_parent_bounds_path, robot_root / "collision_parent_bounds.yaml"),
    )
    for source, destination in copies:
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise ValueError(f"refusing to overwrite mismatched robot asset: {destination}")
        if not destination.exists():
            shutil.copy2(str(source), str(destination))

    manifest_path = output_root / "manifest.yaml"
    rendered = yaml.safe_dump(dict(manifest), sort_keys=False, allow_unicode=True)
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"refusing to overwrite mismatched dataset manifest: {manifest_path}")
    manifest_path.write_text(rendered, encoding="utf-8")
    (output_root / "shards").mkdir(exist_ok=True)
    (output_root / "splits").mkdir(exist_ok=True)
