#!/usr/bin/env python3
"""Generate rigid cooperative Marvin trajectories in the warehouse scene.

The task is proposed in object ``xyz+yaw`` space, realized with continuous
paired IK, fitted/projected as a 14-D B-spline, and finally audited densely in
both collision models.  A projected constrained RRTConnect is available as a
fallback; the independent Marvin generator remains a separate entry point.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import h5py
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch
import yaml

from pb_ompl.pb_ompl import fit_bspline_to_path, ob, og
from scripts.generate_data.generate_marvin_warehouse_bimanual import (
    DEFAULT_CONFIG as INDEPENDENT_CONFIG,
    EE_GOAL_LINKS,
    JOINT_NAMES,
    MarvinWarehouseGenerator,
    file_sha256,
)
from torch_robotics.environments.env_warehouse_marvin_bimanual import EnvWarehouseMarvinBimanual


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-cooperative.yaml"
)
GRASP_PROFILES = (
    Path(__file__).resolve().parents[2] / "mpd/torch_robotics/torch_robotics/data/configs/marvin/grasp_profiles.yaml"
)
TASK_MODE = "cooperative_rigid"
OBJECT_POSE_SCHEMA = "xyz_quaternion_xyzw/v1"
DATASET_SCHEMA = "marvin_bimanual_warehouse_cooperative/v1"
PHASE_IDS = {"grasp": 0, "lift": 1, "transfer": 2, "lower": 3, "place": 4}
SOLUTION_DATASETS = (
    "sol_path",
    "q_start",
    "q_goal",
    "task_id",
    "solution_id",
    "planning_time",
    "joint_path_length",
    "object_path",
    "object_phase",
    "object_start_pose",
    "object_goal_pose",
    "closure_error",
    "T_object_left_grasp",
    "T_object_right_grasp",
    "bspline_params_tt",
    "bspline_params_cc",
    "bspline_params_k",
    "task_mode",
    "proposal_type",
    "active_joint_mask",
    "active_ee_mask",
    "ee_goal_pose",
)
CONTEXT_DATASETS = (
    "context_task_id",
    "context_object_start_pose",
    "context_object_goal_pose",
    "context_solutions_found",
    "context_failure_reason",
)


def _require_mapping(config, key):
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _positive(mapping, key, cast=float):
    value = cast(mapping.get(key, 0))
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def validate_config(config):
    """Validate the cooperative schema without accepting independent configs."""
    if config.get("env_id") != "EnvWarehouseMarvinBimanual" or config.get("robot_id") != "RobotMarvinBimanual":
        raise ValueError("requires EnvWarehouseMarvinBimanual / RobotMarvinBimanual")
    if config.get("task_mode") != TASK_MODE or config.get("task_family") != "cooperative":
        raise ValueError("task_family/task_mode must be cooperative/cooperative_rigid")

    payload = _require_mapping(config, "payload")
    if (
        payload.get("type") != "box"
        or payload.get("fixed_geometry") is not True
        or payload.get("fixed_mass") is not True
    ):
        raise ValueError("payload must be a fixed-geometry, fixed-mass box")
    size = np.asarray(payload.get("size_xyz", []), dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("payload.size_xyz must contain three positive values")
    _positive(payload, "mass")

    grasp = _require_mapping(config, "grasp")
    if grasp.get("profile") != "default_box" or grasp.get("fixed_transforms") is not True:
        raise ValueError("grasp must use fixed default_box transforms")

    object_planning = _require_mapping(config, "object_planning")
    state_space = object_planning.get("state_space")
    if state_space not in {"xyz", "xyz_yaw"}:
        raise ValueError("object_planning.state_space must be xyz or xyz_yaw")
    if _positive(object_planning, "proposals_per_task", int) < 2:
        raise ValueError("object_planning.proposals_per_task must be at least two")
    if object_planning.get("use_lift_transfer_lower") is not True:
        raise ValueError("lift-transfer-lower proposals must be enabled")
    if object_planning.get("use_rrt_connect") is not True:
        raise ValueError("object RRTConnect proposals must be enabled")
    if object_planning.get("simplify_path") is not False:
        raise ValueError("object paths must not be simplified")
    bounds = np.asarray(object_planning.get("bounds", []), dtype=float)
    dimension = 3 if state_space == "xyz" else 4
    if bounds.shape != (dimension, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError(f"object_planning.bounds must be {dimension}x2 increasing bounds")
    base_rotation = np.asarray(object_planning.get("base_rotation", []), dtype=float)
    if (
        base_rotation.shape != (3, 3)
        or not np.allclose(base_rotation.T @ base_rotation, np.eye(3))
        or not np.isclose(np.linalg.det(base_rotation), 1.0)
    ):
        raise ValueError("object_planning.base_rotation must be a rotation matrix")
    regions = _require_mapping(config, "object_regions")
    for region_name in ("start", "goal"):
        region = _require_mapping(regions, region_name)
        for axis in ("x", "y", "z", "yaw_deg"):
            interval = np.asarray(region.get(axis, []), dtype=float)
            if interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] > interval[1]:
                raise ValueError(f"object_regions.{region_name}.{axis} must be [low, high]")

    ik = _require_mapping(config, "continuous_ik")
    for key in ("initial_branches", "beam_width", "waypoints", "max_nfev"):
        _positive(ik, key, int)
    if ik.get("use_previous_seed") is not True or ik.get("use_nullspace_perturbation") is not True:
        raise ValueError("continuous IK requires previous seeds and nullspace perturbations")

    optimization = _require_mapping(config, "joint_optimization")
    if optimization.get("representation") != "bspline" or int(optimization.get("degree", 0)) != 5:
        raise ValueError("joint optimization must use a degree-5 B-spline")
    _positive(optimization, "control_points", int)
    if int(optimization["control_points"]) <= int(optimization["degree"]):
        raise ValueError("joint_optimization.control_points must exceed degree")
    if optimization.get("hard_closed_chain") is not True or optimization.get("hard_object_start_goal") is not True:
        raise ValueError("closed chain and object endpoints must be hard constraints")

    fallback = _require_mapping(config, "fallback")
    if fallback.get("planner") != "constrained_rrt_connect" or fallback.get("space") != "projected":
        raise ValueError("fallback must be constrained_rrt_connect in projected space")
    if not isinstance(fallback.get("enabled"), bool):
        raise ValueError("fallback.enabled must be boolean")

    validation = _require_mapping(config, "validation")
    for key in ("path_points", "spline_points"):
        _positive(validation, key, int)
    _positive(validation, "adaptive_max_joint_step")
    for key in (
        "closure_translation_tolerance",
        "closure_rotation_tolerance_deg",
        "object_endpoint_translation_tolerance",
        "object_endpoint_rotation_tolerance_deg",
    ):
        _positive(validation, key)
    for key in ("torch_collision", "pybullet_collision", "payload_collision"):
        if validation.get(key) is not True:
            raise ValueError(f"validation.{key} must be true")

    dataset = _require_mapping(config, "dataset")
    _positive(dataset, "num_contexts", int)
    _positive(dataset, "solutions_per_task_target", int)
    if dataset.get("group_by_task_id") is not True or dataset.get("preserve_failed_task_statistics") is not True:
        raise ValueError("dataset must group task IDs and preserve failed task statistics")

    collision = _require_mapping(config, "collision")
    for key in ("robot_environment_margin", "payload_environment_margin", "robot_payload_margin"):
        if float(collision.get(key, -1)) < 0:
            raise ValueError(f"collision.{key} must be nonnegative")

    launcher = _require_mapping(config, "launcher")
    _positive(launcher, "workers", int)
    _positive(launcher, "contexts_per_shard", int)
    _positive(launcher, "worker_lifetime_trajectories", int)
    if int(launcher.get("max_worker_restarts_per_shard", -1)) < 0:
        raise ValueError("launcher.max_worker_restarts_per_shard must be nonnegative")
    if launcher.get("resumable_shards") is not True:
        raise ValueError("launcher.resumable_shards must be true")
    return config


def _transform(xyz, rpy):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def object_transform(state, base_rotation):
    state = np.asarray(state, dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", state[3]).as_matrix() @ base_rotation
    transform[:3, 3] = state[:3]
    return transform


def transform_to_pose(transform):
    transform = np.asarray(transform)
    return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat()]


def pose_path_from_states(states, base_rotation):
    return np.stack([transform_to_pose(object_transform(state, base_rotation)) for state in states])


def interpolate_object_states(states, count):
    """Arc-length interpolate xyz and unwrap yaw without simplifying vertices."""
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 4 or len(states) < 2:
        raise ValueError("object states must have shape [N>=2, 4]")
    unwrapped = states.copy()
    unwrapped[:, 3] = np.unwrap(unwrapped[:, 3])
    scale = np.array([1.0, 1.0, 1.0, 0.25])
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(unwrapped, axis=0) * scale, axis=1))]
    distance, unique = np.unique(distance, return_index=True)
    unwrapped = unwrapped[unique]
    if len(distance) == 1:
        return np.repeat(unwrapped, count, axis=0)
    query = np.linspace(0.0, distance[-1], count)
    return np.stack([np.interp(query, distance, unwrapped[:, i]) for i in range(4)], axis=1)


def phase_schedule(count, lift_fraction=0.25, lower_fraction=0.25):
    phases = np.full(count, PHASE_IDS["transfer"], dtype=np.uint8)
    lift_end = max(2, int(round(count * lift_fraction)))
    lower_start = min(count - 2, int(round(count * (1.0 - lower_fraction))))
    phases[0], phases[-1] = PHASE_IDS["grasp"], PHASE_IDS["place"]
    phases[1:lift_end] = PHASE_IDS["lift"]
    phases[lower_start:-1] = PHASE_IDS["lower"]
    return phases


def box_surface_spheres(size_xyz, radius):
    """Conservative payload proxy: a regular sphere lattice covering the box."""
    half = np.asarray(size_xyz, dtype=float) / 2.0
    axes = [np.linspace(-h, h, max(2, int(np.ceil(2 * h / radius)) + 1)) for h in half]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    surface = np.isclose(np.abs(grid) / half, 1.0, atol=1e-7).any(axis=1)
    return grid[surface], float(radius)


def obb_signed_distance(points, object_pose, half_extents):
    """Exact point-to-oriented-box signed distance (positive outside)."""
    points = np.asarray(points, dtype=float)
    pose = np.asarray(object_pose, dtype=float)
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    delta = np.abs(local) - np.asarray(half_extents, dtype=float)
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
    inside = np.minimum(np.max(delta, axis=-1), 0.0)
    return outside + inside


def closure_vector(left_pose, right_pose, object_to_left, object_to_right):
    left_object = left_pose @ np.linalg.inv(object_to_left)
    right_object = right_pose @ np.linalg.inv(object_to_right)
    rotation = Rotation.from_matrix(left_object[:3, :3].T @ right_object[:3, :3]).as_rotvec()
    return np.r_[left_object[:3, 3] - right_object[:3, 3], rotation]


@dataclass
class ObjectProposal:
    states: np.ndarray
    phases: np.ndarray
    kind: str


class MarvinWarehouseCooperativeGenerator(MarvinWarehouseGenerator):
    """Object-centric cooperative generator with exact 14-D output."""

    def __init__(self, config, seed, progress_label=None):
        cooperative = validate_config(config)
        # Reuse the already benchmarked Marvin model/collision initialization,
        # but pass its own independent schema to the independent base class.
        compatibility = yaml.safe_load(INDEPENDENT_CONFIG.read_text())
        compatibility["min_distance_robot_env"] = float(cooperative["collision"]["robot_environment_margin"])
        compatibility["disabled_self_collision_pairs"] = cooperative["collision"].get(
            "disabled_self_collision_pairs", compatibility.get("disabled_self_collision_pairs", [])
        )
        super().__init__(compatibility, seed, progress_label=progress_label)
        self._release_rrt_setups()
        self.config = cooperative
        self.stats = Counter()
        self.failed_contexts = []
        self.deadline = float("inf")

        profile = yaml.safe_load(GRASP_PROFILES.read_text())[cooperative["grasp"]["profile"]]
        profile_size = np.asarray(profile["payload"]["size_xyz"], dtype=float)
        configured_size = np.asarray(cooperative["payload"]["size_xyz"], dtype=float)
        if not np.allclose(profile_size, configured_size):
            raise ValueError("payload geometry differs from fixed default_box grasp profile")
        self.object_to_left = _transform(profile["left_grasp"]["xyz"], profile["left_grasp"]["rpy"])
        self.object_to_right = _transform(profile["right_grasp"]["xyz"], profile["right_grasp"]["rpy"])
        self.base_rotation = np.asarray(cooperative["object_planning"]["base_rotation"], dtype=float)
        if self.base_rotation.shape != (3, 3) or not np.allclose(self.base_rotation.T @ self.base_rotation, np.eye(3)):
            raise ValueError("object_planning.base_rotation must be a rotation matrix")
        self.payload_size = configured_size
        self.payload_half = configured_size / 2.0
        proxy_radius = float(cooperative["collision"].get("payload_proxy_radius", 0.025))
        self.payload_proxy_local, self.payload_proxy_radius = box_surface_spheres(configured_size, proxy_radius)
        self._payload_body = self._create_payload_body()
        self._support_obstacles = {
            self.worker.obstacles[i] for i in cooperative["collision"].get("support_obstacle_indices", [0])
        }
        allowed_names = set(cooperative["collision"].get("allowed_payload_contact_links", []))
        self._allowed_payload_link_indices = {
            index for name, index in self.robot._link_name_to_index.items() if name in allowed_names
        }
        sphere_parent_names = tuple(self.torch_robot.collision_sphere_parent_links)
        self._payload_robot_sphere_mask = np.asarray(
            [name not in allowed_names for name in sphere_parent_names], dtype=bool
        )

    def _create_payload_body(self):
        client = self.worker.pybullet_client
        collision = client.createCollisionShape(client.GEOM_BOX, halfExtents=self.payload_half.tolist())
        return client.createMultiBody(
            baseMass=float(self.config["payload"]["mass"]),
            baseCollisionShapeIndex=collision,
            basePosition=[0.0, 0.0, -10.0],
        )

    def close(self):
        if getattr(self, "worker", None) is not None and getattr(self, "_payload_body", None) is not None:
            try:
                self.worker.pybullet_client.removeBody(self._payload_body)
            except Exception:
                pass
            self._payload_body = None
        super().close()

    @contextmanager
    def _timed(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stats[f"{name}_seconds"] += time.perf_counter() - started

    def _set_payload_pose(self, state):
        transform = object_transform(state, self.base_rotation)
        self.worker.pybullet_client.resetBasePositionAndOrientation(
            self._payload_body,
            transform[:3, 3].tolist(),
            Rotation.from_matrix(transform[:3, :3]).as_quat().tolist(),
        )
        return transform

    def _payload_environment_torch_valid(self, transform):
        world = self.payload_proxy_local @ transform[:3, :3].T + transform[:3, 3]
        points = torch.as_tensor(world[None], **self.torch_robot.tensor_args)
        distances = self.torch_object_field.object_signed_distances(points)
        margin = self.payload_proxy_radius + float(self.config["collision"]["payload_environment_margin"])
        return bool((distances > margin).all().item())

    def payload_valid(self, q, object_state, phase=PHASE_IDS["transfer"], *, torch_check=True, bullet_check=True):
        """Check payload/environment and payload/robot with phase-aware support contact."""
        transform = self._set_payload_pose(object_state)
        collision = self.config["collision"]
        allow_support = int(phase) in (PHASE_IDS["grasp"], PHASE_IDS["place"])
        allowed_penetration = float(collision.get("support_contact_penetration", 0.003))
        if torch_check and not allow_support and not self._payload_environment_torch_valid(transform):
            return False
        client = self.worker.pybullet_client
        if bullet_check:
            margin = float(collision["payload_environment_margin"])
            for obstacle in self.worker.obstacles:
                contacts = client.getClosestPoints(self._payload_body, obstacle, distance=margin)
                if not contacts:
                    continue
                if not (allow_support and obstacle in self._support_obstacles):
                    return False
                if any(float(contact[8]) < -allowed_penetration for contact in contacts):
                    return False

        self.robot.set_state(np.asarray(q, dtype=float).tolist())
        if bullet_check:
            contacts = client.getClosestPoints(
                self.robot.id,
                self._payload_body,
                distance=float(collision["robot_payload_margin"]),
            )
            if any(int(contact[3]) not in self._allowed_payload_link_indices for contact in contacts):
                return False
        if torch_check:
            qt = torch.as_tensor(np.asarray(q)[None], **self.torch_robot.tensor_args)
            centers = self.collision_positions(qt)[0].detach().cpu().numpy()
            distances = obb_signed_distance(centers, transform, self.payload_half)
            radii = self.torch_robot.link_collision_spheres_radii.detach().cpu().numpy()
            margin = radii + float(collision["robot_payload_margin"])
            if np.any(distances[self._payload_robot_sphere_mask] <= margin[self._payload_robot_sphere_mask]):
                return False
        return True

    def cooperative_valid(self, q, object_state, phase=PHASE_IDS["transfer"], *, torch_check=True, bullet_check=True):
        q = np.asarray(q, dtype=float)
        if not np.isfinite(q).all():
            return False
        if bullet_check and not self.interface.is_state_valid(q, check_bounds=True):
            return False
        if torch_check and not self._torch_state_valid(q):
            return False
        return self.payload_valid(q, object_state, phase, torch_check=torch_check, bullet_check=bullet_check)

    def _targets(self, object_state):
        world_object = object_transform(object_state, self.base_rotation)
        return {"left": world_object @ self.object_to_left, "right": world_object @ self.object_to_right}

    def _solve_arm(self, arm, target, q_seed, *, regularization=1e-3):
        sl = slice(0, 7) if arm == "left" else slice(7, 14)
        low, high = self.robot.joint_bounds_low_np[sl], self.robot.joint_bounds_high_np[sl]
        q = np.asarray(q_seed, dtype=float).copy()
        seed = np.clip(q[sl], low, high)

        def residual(x):
            q[sl] = x
            pose = self._pose(q, arm)
            error = np.r_[
                pose.translation - target[:3, 3],
                Rotation.from_matrix(target[:3, :3].T @ pose.rotation).as_rotvec(),
            ]
            return np.r_[error, regularization * (x - seed)]

        result = least_squares(
            residual,
            seed,
            bounds=(low, high),
            max_nfev=int(self.config["continuous_ik"]["max_nfev"]),
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        pos_tol = float(self.config["continuous_ik"]["position_tolerance"])
        rot_tol = np.deg2rad(float(self.config["continuous_ik"]["orientation_tolerance_deg"]))
        if np.linalg.norm(result.fun[:3]) > pos_tol or np.linalg.norm(result.fun[3:6]) > rot_tol:
            return None
        return result.x.copy()

    def _paired_from_seed(self, object_state, q_seed, phase, perturb=False):
        seed = np.asarray(q_seed, dtype=float).copy()
        if perturb:
            scale = float(self.config["continuous_ik"].get("nullspace_std", 0.12))
            seed += self.rng.normal(0.0, scale, 14)
            seed = np.clip(seed, self.robot.joint_bounds_low_np, self.robot.joint_bounds_high_np)
        targets = self._targets(object_state)
        q = seed.copy()
        for arm in ("left", "right"):
            solution = self._solve_arm(arm, targets[arm], q)
            if solution is None:
                return None
            q[slice(0, 7) if arm == "left" else slice(7, 14)] = solution
        return q if self.cooperative_valid(q, object_state, phase) else None

    def paired_ik_branches(self, object_state, reference=None, phase=PHASE_IDS["transfer"]):
        started = time.perf_counter()
        count = int(self.config["continuous_ik"]["initial_branches"])
        branches = []
        tries = int(self.config["continuous_ik"].get("initial_tries", count * 10))
        for attempt in range(tries):
            if time.perf_counter() >= self.deadline:
                break
            if reference is not None and attempt == 0:
                seed = np.asarray(reference).copy()
            else:
                seed = self.rng.uniform(self.robot.joint_bounds_low_np, self.robot.joint_bounds_high_np)
            q = self._paired_from_seed(object_state, seed, phase, perturb=False)
            if q is None:
                continue
            if any(np.linalg.norm(q - existing) < 0.08 for existing in branches):
                continue
            branches.append(q)
            if len(branches) >= count:
                break
        self.stats["paired_endpoint_ik_seconds"] += time.perf_counter() - started
        self.stats["paired_endpoint_branches"] += len(branches)
        return branches

    def _sample_object_state(self, region_name):
        region = self.config["object_regions"][region_name]
        xyz = [self.rng.uniform(*region[axis]) for axis in "xyz"]
        if self.config["object_planning"]["state_space"] == "xyz":
            yaw = float(self.config["object_planning"].get("fixed_yaw", 0.0))
        else:
            yaw = self.rng.uniform(*np.deg2rad(region["yaw_deg"]))
        return np.r_[xyz, yaw]

    def sample_context(self):
        tries = int(self.config["dataset"].get("context_sampling_tries", 30))
        for _ in range(tries):
            start = self._sample_object_state("start")
            goal = self._sample_object_state("goal")
            if np.linalg.norm(goal[:3] - start[:3]) < float(self.config["object_planning"]["min_start_goal_distance"]):
                continue
            start_branches = self.paired_ik_branches(start, phase=PHASE_IDS["grasp"])
            if not start_branches:
                self.stats["context_start_ik_failure"] += 1
                continue
            goal_branches = self.paired_ik_branches(goal, reference=start_branches[0], phase=PHASE_IDS["place"])
            if not goal_branches:
                self.stats["context_goal_ik_failure"] += 1
                continue
            pairs = [
                (np.linalg.norm(goal_q - start_q), start_q, goal_q)
                for start_q in start_branches
                for goal_q in goal_branches
            ]
            pairs = [
                pair for pair in pairs if pair[0] >= float(self.config["continuous_ik"].get("min_joint_delta", 0.2))
            ]
            if pairs:
                _, q_start, q_goal = min(pairs, key=lambda item: item[0])
                return start, goal, q_start, q_goal
        return None

    def object_state_valid(self, state, phase=PHASE_IDS["transfer"]):
        transform = self._set_payload_pose(state)
        if int(phase) not in (PHASE_IDS["grasp"], PHASE_IDS["place"]):
            if not self._payload_environment_torch_valid(transform):
                return False
        client = self.worker.pybullet_client
        margin = float(self.config["collision"]["payload_environment_margin"])
        return not any(
            client.getClosestPoints(self._payload_body, obstacle, distance=margin) for obstacle in self.worker.obstacles
        )

    def _lift_transfer_lower(self, start, goal, variant):
        planning = self.config["object_planning"]
        lift = (
            max(start[2], goal[2])
            + float(planning["lift_height"])
            + variant * float(planning.get("lift_height_step", 0.025))
        )
        a, b = start.copy(), goal.copy()
        a[2], b[2] = lift, lift
        # A lateral midpoint creates a second deterministic homotopy while
        # retaining the lift/transfer/lower phase semantics.
        if variant:
            midpoint = (a + b) / 2.0
            midpoint[1] += (-1.0 if variant % 2 else 1.0) * float(planning.get("lateral_offset", 0.08))
            vertices = np.stack([start, a, midpoint, b, goal])
        else:
            vertices = np.stack([start, a, b, goal])
        count = int(self.config["continuous_ik"]["waypoints"])
        states = interpolate_object_states(vertices, count)
        phases = phase_schedule(count)
        if not all(self.object_state_valid(state, phase) for state, phase in zip(states[1:-1], phases[1:-1])):
            return None
        return ObjectProposal(states, phases, "lift_transfer_lower")

    def _object_rrt(self, start_state, goal_state):
        planning = self.config["object_planning"]
        dimension = 3 if planning["state_space"] == "xyz" else 4
        space = ob.RealVectorStateSpace(dimension)
        bounds = ob.RealVectorBounds(dimension)
        for i, pair in enumerate(planning["bounds"]):
            bounds.setLow(i, float(pair[0]))
            bounds.setHigh(i, float(pair[1]))
        space.setBounds(bounds)
        setup = og.SimpleSetup(space)

        def expand(state):
            values = [state[i] for i in range(dimension)]
            if dimension == 3:
                values.append(float(planning.get("fixed_yaw", 0.0)))
            return np.asarray(values)

        checker = ob.StateValidityCheckerFn(lambda state: self.object_state_valid(expand(state)))
        setup.setStateValidityChecker(checker)
        setup.getSpaceInformation().setStateValidityCheckingResolution(float(planning.get("validity_resolution", 0.01)))
        planner = og.RRTConnect(setup.getSpaceInformation())
        planner.setRange(float(planning.get("rrt_range", 0.12)))
        setup.setPlanner(planner)
        start, goal = ob.State(space), ob.State(space)
        for i in range(dimension):
            start[i], goal[i] = float(start_state[i]), float(goal_state[i])
        setup.setStartAndGoalStates(start, goal)
        before = time.perf_counter()
        setup.solve(float(planning.get("rrt_allowed_time", 2.0)))
        self.stats["object_rrt_seconds"] += time.perf_counter() - before
        if not setup.haveExactSolutionPath():
            self.stats["object_rrt_failure"] += 1
            return None
        raw = np.asarray([expand(state) for state in setup.getSolutionPath().getStates()])
        states = interpolate_object_states(raw, int(self.config["continuous_ik"]["waypoints"]))
        phases = phase_schedule(len(states))
        return ObjectProposal(states, phases, "object_rrt_connect")

    def object_proposals(self, start, goal):
        count = int(self.config["object_planning"]["proposals_per_task"])
        lift_transfer_lower_count = max(1, count // 2)
        proposals = []
        for index in range(count):
            if index < lift_transfer_lower_count:
                proposal = self._lift_transfer_lower(start, goal, index)
            else:
                proposal = self._object_rrt(start, goal)
            if proposal is not None:
                proposals.append(proposal)
        self.stats["object_proposals"] += len(proposals)
        return proposals

    def continuous_paired_ik(self, proposal, q_start, q_goal):
        with self._timed("continuous_ik"):
            beam = [(0.0, [np.asarray(q_start).copy()])]
            width = int(self.config["continuous_ik"]["beam_width"])
            states, phases = proposal.states, proposal.phases
            for index in range(1, len(states) - 1):
                children = []
                for score, path in beam:
                    parent = path[-1]
                    for branch in range(width):
                        q = self._paired_from_seed(states[index], parent, phases[index], perturb=branch > 0)
                        if q is None:
                            continue
                        step = np.linalg.norm(q - parent)
                        children.append((score + step + 0.02 * np.linalg.norm(q), path + [q]))
                if not children:
                    self.stats["continuous_ik_dead_end"] += 1
                    return None
                children.sort(key=lambda item: item[0])
                beam = children[:width]
            candidates = []
            for score, path in beam:
                if not self.cooperative_valid(q_goal, states[-1], phases[-1]):
                    continue
                candidates.append((score + np.linalg.norm(q_goal - path[-1]), np.stack(path + [q_goal])))
            return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _project_path_to_object(self, q_path, object_states, phases):
        projected = [np.asarray(q_path[0]).copy()]
        for index in range(1, len(q_path) - 1):
            q = self._paired_from_seed(object_states[index], q_path[index], phases[index], perturb=False)
            if q is None:
                return None
            projected.append(q)
        projected.append(np.asarray(q_path[-1]).copy())
        return np.stack(projected)

    def optimize_bspline(self, coarse_path, proposal, q_start, q_goal):
        """Alternating 14-D spline fit and closed-chain manifold projection."""
        optimization = self.config["joint_optimization"]
        count = int(optimization.get("projection_points", 64))
        q_path = np.stack(
            [
                np.interp(np.linspace(0, 1, count), np.linspace(0, 1, len(coarse_path)), coarse_path[:, j])
                for j in range(14)
            ],
            axis=1,
        )
        object_states = interpolate_object_states(proposal.states, count)
        phases = phase_schedule(count)
        q_path[0], q_path[-1] = q_start, q_goal
        with self._timed("joint_optimization"):
            try:
                for _ in range(int(optimization.get("projection_iterations", 2))):
                    tt, cc, degree = fit_bspline_to_path(
                        q_path,
                        bspline_degree=int(optimization["degree"]),
                        bspline_num_control_points=int(optimization["control_points"]),
                        bspline_zero_vel_at_start_and_goal=True,
                        bspline_zero_acc_at_start_and_goal=True,
                    )
                    spline = BSpline(np.asarray(tt), np.asarray(cc).T, degree)
                    q_path = spline(np.linspace(0.0, 1.0, count))
                    q_path[0], q_path[-1] = q_start, q_goal
                    q_path = self._project_path_to_object(q_path, object_states, phases)
                    if q_path is None:
                        return None
                tt, cc, degree = fit_bspline_to_path(
                    q_path,
                    bspline_degree=int(optimization["degree"]),
                    bspline_num_control_points=int(optimization["control_points"]),
                    bspline_zero_vel_at_start_and_goal=True,
                    bspline_zero_acc_at_start_and_goal=True,
                )
                return np.asarray(tt), np.asarray(cc), int(degree)
            except (ValueError, np.linalg.LinAlgError):
                return None

    def object_state_from_q(self, q):
        left = self._pose(q, "left").homogeneous
        world_object = left @ np.linalg.inv(self.object_to_left)
        yaw_relative = world_object[:3, :3] @ self.base_rotation.T
        return np.r_[world_object[:3, 3], Rotation.from_matrix(yaw_relative).as_euler("zyx")[0]]

    def closure_errors(self, q_path):
        errors = []
        for q in np.asarray(q_path):
            errors.append(
                closure_vector(
                    self._pose(q, "left").homogeneous,
                    self._pose(q, "right").homogeneous,
                    self.object_to_left,
                    self.object_to_right,
                )
            )
        errors = np.asarray(errors)
        return np.linalg.norm(errors[:, :3], axis=1), np.linalg.norm(errors[:, 3:], axis=1)

    def validate_spline(self, spline, object_start, object_goal):
        validation = self.config["validation"]
        tt, cc, degree = spline
        q = BSpline(tt, cc.T, degree)(np.linspace(0.0, 1.0, int(validation["spline_points"])))
        if not np.allclose(q[[0, -1]], np.stack([cc[:, 0], cc[:, -1]]), atol=1e-5):
            return None
        inferred = np.stack([self.object_state_from_q(state) for state in q])
        translation, rotation = self.closure_errors(q)
        if translation.max() > float(validation["closure_translation_tolerance"]):
            self.stats["validator_closure_translation"] += 1
            return None
        if rotation.max() > np.deg2rad(float(validation["closure_rotation_tolerance_deg"])):
            self.stats["validator_closure_rotation"] += 1
            return None
        for actual, expected, label in ((inferred[0], object_start, "start"), (inferred[-1], object_goal, "goal")):
            if np.linalg.norm(actual[:3] - expected[:3]) > float(validation["object_endpoint_translation_tolerance"]):
                self.stats[f"validator_object_{label}"] += 1
                return None
            if abs(np.arctan2(np.sin(actual[3] - expected[3]), np.cos(actual[3] - expected[3]))) > np.deg2rad(
                float(validation["object_endpoint_rotation_tolerance_deg"])
            ):
                self.stats[f"validator_object_{label}"] += 1
                return None

        # Adaptive joint-space subdivision is applied in addition to the fixed
        # 512-point audit so fast-moving spline spans cannot jump obstacles.
        step = float(validation["adaptive_max_joint_step"])
        dense_q, dense_phase = [q[:1]], [np.asarray([PHASE_IDS["grasp"]])]
        for index, (a, b) in enumerate(zip(q[:-1], q[1:])):
            subdivisions = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
            fraction = np.linspace(0.0, 1.0, subdivisions + 1)[1:]
            segment_q = a + fraction[:, None] * (b - a)
            dense_q.append(segment_q)
            phase = PHASE_IDS["transfer"]
            if index == 0:
                phase = PHASE_IDS["lift"]
            elif index == len(q) - 2:
                phase = PHASE_IDS["lower"]
            dense_phase.append(np.full(subdivisions, phase, dtype=np.uint8))
        dense_q = np.concatenate(dense_q)
        # Recompute the carried object from exact FK after subdivision; pose
        # interpolation would under-approximate payload motion between joints.
        dense_object = np.stack([self.object_state_from_q(state) for state in dense_q])
        dense_phase = np.concatenate(dense_phase)
        dense_phase[0], dense_phase[-1] = PHASE_IDS["grasp"], PHASE_IDS["place"]
        with self._timed("validator"):
            dense_translation, dense_rotation = self.closure_errors(dense_q)
            if dense_translation.max() > float(validation["closure_translation_tolerance"]):
                self.stats["validator_adaptive_closure_translation"] += 1
                return None
            if dense_rotation.max() > np.deg2rad(float(validation["closure_rotation_tolerance_deg"])):
                self.stats["validator_adaptive_closure_rotation"] += 1
                return None
            for state, obj, phase in zip(dense_q, dense_object, dense_phase):
                if not self.cooperative_valid(
                    state,
                    obj,
                    phase,
                    torch_check=bool(validation["torch_collision"]),
                    bullet_check=bool(validation["pybullet_collision"]),
                ):
                    self.stats["validator_collision"] += 1
                    return None
        count = int(validation["path_points"])
        output_q = BSpline(tt, cc.T, degree)(np.linspace(0.0, 1.0, count))
        output_object = np.stack([self.object_state_from_q(state) for state in output_q])
        output_phase = phase_schedule(count)
        return output_q, output_object, output_phase, np.array([dense_translation.max(), dense_rotation.max()])

    def constrained_rrt_fallback(self, q_start, q_goal):
        """Plan on the 8-D rigid-grasp manifold embedded in the 14-D space."""
        generator = self

        class RigidGraspConstraint(ob.Constraint):
            def __init__(self):
                super().__init__(14, 6)

            def function(self, x, out):
                q = np.asarray([x[i] for i in range(14)])
                residual = closure_vector(
                    generator._pose(q, "left").homogeneous,
                    generator._pose(q, "right").homogeneous,
                    generator.object_to_left,
                    generator.object_to_right,
                )
                for index, value in enumerate(residual):
                    out[index] = float(value)

            def jacobian(self, x, out):
                q = np.asarray([x[i] for i in range(14)])
                epsilon = float(generator.config["fallback"].get("jacobian_epsilon", 1e-5))
                for index in range(14):
                    plus, minus = q.copy(), q.copy()
                    plus[index] += epsilon
                    minus[index] -= epsilon
                    out[:, index] = (
                        closure_vector(
                            generator._pose(plus, "left").homogeneous,
                            generator._pose(plus, "right").homogeneous,
                            generator.object_to_left,
                            generator.object_to_right,
                        )
                        - closure_vector(
                            generator._pose(minus, "left").homogeneous,
                            generator._pose(minus, "right").homogeneous,
                            generator.object_to_left,
                            generator.object_to_right,
                        )
                    ) / (2.0 * epsilon)

        fallback = self.config["fallback"]
        ambient = ob.RealVectorStateSpace(14)
        bounds = ob.RealVectorBounds(14)
        for i, (low, high) in enumerate(zip(self.robot.joint_bounds_low_np, self.robot.joint_bounds_high_np)):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        ambient.setBounds(bounds)
        constraint = RigidGraspConstraint()
        constraint.setTolerance(float(fallback.get("projection_tolerance", 1e-3)))
        constraint.setMaxIterations(int(fallback.get("projection_iterations", 30)))
        space = ob.ProjectedStateSpace(ambient, constraint)
        information = ob.ConstrainedSpaceInformation(space)
        space.setDelta(float(fallback.get("delta", 0.08)))
        space.setLambda(float(fallback.get("lambda", 2.0)))
        space.setup()
        setup = og.SimpleSetup(information)

        def valid(state):
            q = np.asarray([state[i] for i in range(14)])
            obj = self.object_state_from_q(q)
            return self.cooperative_valid(q, obj, PHASE_IDS["transfer"])

        checker = ob.StateValidityCheckerFn(valid)
        setup.setStateValidityChecker(checker)
        planner = og.RRTConnect(information)
        planner.setRange(float(fallback.get("range", 0.2)))
        setup.setPlanner(planner)
        start, goal = ob.State(space), ob.State(space)
        for i in range(14):
            start[i], goal[i] = float(q_start[i]), float(q_goal[i])
        setup.setStartAndGoalStates(start, goal)
        with self._timed("fallback_rrt"):
            setup.solve(float(fallback.get("allowed_time", 10.0)))
        if not setup.haveExactSolutionPath():
            self.stats["fallback_failure"] += 1
            return None
        raw = np.asarray([[state[i] for i in range(14)] for state in setup.getSolutionPath().getStates()])
        count = int(self.config["continuous_ik"]["waypoints"])
        q = np.stack(
            [np.interp(np.linspace(0, 1, count), np.linspace(0, 1, len(raw)), raw[:, j]) for j in range(14)], axis=1
        )
        states = np.stack([self.object_state_from_q(state) for state in q])
        return q, ObjectProposal(states, phase_schedule(count), "projected_rrt_connect")

    def _is_diverse(self, path, accepted):
        threshold = float(self.config["dataset"].get("diversity_mean_joint_distance", 0.08))
        return all(np.linalg.norm(path - other[0], axis=1).mean() >= threshold for other in accepted)

    def solve_context(self, task_id, context):
        object_start, object_goal, q_start, q_goal = context
        accepted = []
        reasons = Counter()
        started = time.perf_counter()
        for proposal in self.object_proposals(object_start, object_goal):
            coarse = self.continuous_paired_ik(proposal, q_start, q_goal)
            if coarse is None:
                reasons["continuous_ik"] += 1
                self.stats["solution_failure_continuous_ik"] += 1
                continue
            spline = self.optimize_bspline(coarse, proposal, q_start, q_goal)
            if spline is None:
                reasons["joint_optimization"] += 1
                self.stats["solution_failure_joint_optimization"] += 1
                continue
            validated = self.validate_spline(spline, object_start, object_goal)
            if validated is None:
                reasons["validation"] += 1
                self.stats["solution_failure_validation"] += 1
                continue
            path, object_states, phases, closure = validated
            if not self._is_diverse(path, accepted):
                reasons["duplicate"] += 1
                self.stats["diversity_rejected"] += 1
                continue
            accepted.append((path, object_states, phases, closure, spline, proposal.kind))
            if len(accepted) >= int(self.config["dataset"]["solutions_per_task_target"]):
                break

        if (
            len(accepted) < int(self.config["dataset"]["solutions_per_task_target"])
            and self.config["fallback"]["enabled"]
        ):
            fallback = self.constrained_rrt_fallback(q_start, q_goal)
            if fallback is not None:
                coarse, proposal = fallback
                spline = self.optimize_bspline(coarse, proposal, q_start, q_goal)
                validated = None if spline is None else self.validate_spline(spline, object_start, object_goal)
                if validated is not None:
                    path, object_states, phases, closure = validated
                    if self._is_diverse(path, accepted):
                        accepted.append((path, object_states, phases, closure, spline, proposal.kind))
                    else:
                        reasons["duplicate"] += 1
                else:
                    reasons["fallback_validation"] += 1

        metadata = []
        for solution_id, (path, object_states, phases, closure, spline, proposal_type) in enumerate(accepted):
            metadata.append(
                dict(
                    task_id=task_id,
                    solution_id=solution_id,
                    task_mode=TASK_MODE,
                    proposal_type=proposal_type,
                    q_start=q_start.copy(),
                    q_goal=q_goal.copy(),
                    object_start=object_start.copy(),
                    object_goal=object_goal.copy(),
                    object_states=object_states,
                    object_phases=phases,
                    closure_error=closure,
                    bspline=spline,
                    planning_time=time.perf_counter() - started,
                    joint_path_length=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                    ee_goal_pose=self.dual_ee_goal_pose(q_goal),
                )
            )
        if len(accepted) >= int(self.config["dataset"]["solutions_per_task_target"]):
            reason = "quota_met"
        elif reasons:
            reason = reasons.most_common(1)[0][0]
        elif accepted:
            reason = "under_quota"
        else:
            reason = "no_proposal"
        return [item[0] for item in accepted], metadata, reason

    def generate(self, num_contexts, start_task_id=0):
        paths, metadata, contexts = [], [], []
        for offset in range(num_contexts):
            task_id = start_task_id + offset
            self.stats["contexts_attempted"] += 1
            self.deadline = time.perf_counter() + float(self.config["dataset"].get("context_timeout_seconds", 600))
            with self._timed("context_sampling"):
                context = self.sample_context()
            if context is None:
                self.stats["context_sampling_failure"] += 1
                contexts.append(
                    dict(
                        task_id=task_id,
                        object_start=np.full(4, np.nan),
                        object_goal=np.full(4, np.nan),
                        solutions=0,
                        failure_reason="context_sampling",
                    )
                )
                continue
            new_paths, new_metadata, reason = self.solve_context(task_id, context)
            paths.extend(new_paths)
            metadata.extend(new_metadata)
            contexts.append(
                dict(
                    task_id=task_id,
                    object_start=context[0],
                    object_goal=context[1],
                    solutions=len(new_paths),
                    failure_reason=reason,
                )
            )
            self.stats["solutions_generated"] += len(new_paths)
            if len(new_paths) == int(self.config["dataset"]["solutions_per_task_target"]):
                self.stats["contexts_quota_met"] += 1
            else:
                self.stats["contexts_under_quota"] += 1
            prefix = f"[{self.progress_label}] " if self.progress_label else ""
            print(f"{prefix}context {offset + 1}/{num_contexts}: {len(new_paths)} solutions ({reason})", flush=True)
        return paths, metadata, contexts


def _empty_solution_array(config, key):
    path_points = int(config["validation"]["path_points"])
    control_points = int(config["joint_optimization"]["control_points"])
    degree = int(config["joint_optimization"]["degree"])
    shapes = {
        "sol_path": (0, path_points, 14),
        "q_start": (0, 14),
        "q_goal": (0, 14),
        "task_id": (0,),
        "solution_id": (0,),
        "planning_time": (0,),
        "joint_path_length": (0,),
        "object_path": (0, path_points, 7),
        "object_phase": (0, path_points),
        "object_start_pose": (0, 7),
        "object_goal_pose": (0, 7),
        "closure_error": (0, 2),
        "T_object_left_grasp": (0, 4, 4),
        "T_object_right_grasp": (0, 4, 4),
        "bspline_params_tt": (0, control_points + degree + 1),
        "bspline_params_cc": (0, 14, control_points),
        "bspline_params_k": (0,),
        "active_joint_mask": (0, 14),
        "active_ee_mask": (0, 2),
        "ee_goal_pose": (0, 2, 3, 4),
    }
    return np.empty(shapes[key], dtype=np.float32)


def write_dataset(output, config, paths, metadata, contexts, seed, stats=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "dataset_merged.hdf5"
    if target.exists():
        raise FileExistsError(target)
    base_rotation = np.asarray(config["object_planning"]["base_rotation"], dtype=float)
    profile = yaml.safe_load(GRASP_PROFILES.read_text())[config["grasp"]["profile"]]
    left = _transform(profile["left_grasp"]["xyz"], profile["left_grasp"]["rpy"])
    right = _transform(profile["right_grasp"]["xyz"], profile["right_grasp"]["rpy"])
    string_dtype = h5py.string_dtype()
    with h5py.File(target, "x") as handle:
        numeric = {
            "sol_path": np.asarray(paths) if paths else _empty_solution_array(config, "sol_path"),
            "q_start": (
                np.asarray([m["q_start"] for m in metadata]) if metadata else _empty_solution_array(config, "q_start")
            ),
            "q_goal": (
                np.asarray([m["q_goal"] for m in metadata]) if metadata else _empty_solution_array(config, "q_goal")
            ),
            "task_id": np.asarray([m["task_id"] for m in metadata], dtype=np.int64),
            "solution_id": np.asarray([m["solution_id"] for m in metadata], dtype=np.int16),
            "planning_time": np.asarray([m["planning_time"] for m in metadata]),
            "joint_path_length": np.asarray([m["joint_path_length"] for m in metadata]),
            "object_path": (
                np.asarray([pose_path_from_states(m["object_states"], base_rotation) for m in metadata])
                if metadata
                else _empty_solution_array(config, "object_path")
            ),
            "object_phase": (
                np.asarray([m["object_phases"] for m in metadata], dtype=np.uint8)
                if metadata
                else _empty_solution_array(config, "object_phase").astype(np.uint8)
            ),
            "object_start_pose": (
                np.asarray([transform_to_pose(object_transform(m["object_start"], base_rotation)) for m in metadata])
                if metadata
                else _empty_solution_array(config, "object_start_pose")
            ),
            "object_goal_pose": (
                np.asarray([transform_to_pose(object_transform(m["object_goal"], base_rotation)) for m in metadata])
                if metadata
                else _empty_solution_array(config, "object_goal_pose")
            ),
            "closure_error": (
                np.asarray([m["closure_error"] for m in metadata])
                if metadata
                else _empty_solution_array(config, "closure_error")
            ),
            "T_object_left_grasp": np.repeat(left[None], len(metadata), axis=0),
            "T_object_right_grasp": np.repeat(right[None], len(metadata), axis=0),
            "bspline_params_tt": (
                np.asarray([m["bspline"][0] for m in metadata])
                if metadata
                else _empty_solution_array(config, "bspline_params_tt")
            ),
            "bspline_params_cc": (
                np.asarray([m["bspline"][1] for m in metadata])
                if metadata
                else _empty_solution_array(config, "bspline_params_cc")
            ),
            "bspline_params_k": np.asarray([m["bspline"][2] for m in metadata], dtype=np.int8),
            "active_joint_mask": np.ones((len(metadata), 14), dtype=bool),
            "active_ee_mask": np.ones((len(metadata), 2), dtype=bool),
            "ee_goal_pose": (
                np.asarray([m["ee_goal_pose"] for m in metadata])
                if metadata
                else _empty_solution_array(config, "ee_goal_pose")
            ),
        }
        for key, value in numeric.items():
            handle.create_dataset(key, data=value, compression="gzip" if value.ndim > 1 and len(value) else None)
        handle.create_dataset("task_mode", data=[m["task_mode"] for m in metadata], dtype=string_dtype)
        handle.create_dataset("proposal_type", data=[m["proposal_type"] for m in metadata], dtype=string_dtype)
        handle.create_dataset("context_task_id", data=np.asarray([c["task_id"] for c in contexts], dtype=np.int64))
        handle.create_dataset(
            "context_object_start_pose",
            data=np.asarray(
                [
                    (
                        transform_to_pose(object_transform(c["object_start"], base_rotation))
                        if np.isfinite(c["object_start"]).all()
                        else np.full(7, np.nan)
                    )
                    for c in contexts
                ]
            ),
        )
        handle.create_dataset(
            "context_object_goal_pose",
            data=np.asarray(
                [
                    (
                        transform_to_pose(object_transform(c["object_goal"], base_rotation))
                        if np.isfinite(c["object_goal"]).all()
                        else np.full(7, np.nan)
                    )
                    for c in contexts
                ]
            ),
        )
        handle.create_dataset(
            "context_solutions_found", data=np.asarray([c["solutions"] for c in contexts], dtype=np.int16)
        )
        handle.create_dataset(
            "context_failure_reason", data=[c["failure_reason"] for c in contexts], dtype=string_dtype
        )
        handle.attrs["joint_names"] = np.asarray(JOINT_NAMES, dtype=object)
        handle.attrs["scene_version"] = EnvWarehouseMarvinBimanual.scene_version
        handle.attrs["object_pose_schema"] = OBJECT_POSE_SCHEMA
        handle.attrs["phase_ids"] = json.dumps(PHASE_IDS, sort_keys=True)
        handle.attrs["validated_splines"] = True
        handle.attrs["solution_datasets"] = json.dumps(SOLUTION_DATASETS)
        handle.attrs["context_datasets"] = json.dumps(CONTEXT_DATASETS)
        handle.attrs["ee_goal_links"] = np.asarray(EE_GOAL_LINKS, dtype=object)
    manifest = {
        "schema": DATASET_SCHEMA,
        "scene_version": EnvWarehouseMarvinBimanual.scene_version,
        "task_mode": TASK_MODE,
        "num_contexts": len(contexts),
        "num_trajectories": len(paths),
        "solutions_per_task_target": int(config["dataset"]["solutions_per_task_target"]),
        "context_solution_histogram": dict(Counter(c["solutions"] for c in contexts)),
        "failed_context_reasons": dict(
            Counter(c["failure_reason"] for c in contexts if c["failure_reason"] != "quota_met")
        ),
        "proposal_counts": dict(Counter(m["proposal_type"] for m in metadata)),
        "joint_names": list(JOINT_NAMES),
        "dataset_sha256": file_sha256(target),
        "stats": dict(stats or {}),
    }
    robot_dir = Path(__file__).resolve().parents[2] / "mpd/torch_robotics/torch_robotics/data/urdf/robots/marvin"
    manifest["model_sha256"] = file_sha256(robot_dir / "marvin_pika_bimanual_mpd.urdf")
    (output / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (output / "generation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "args.yaml").write_text(
        yaml.safe_dump(
            {
                "env_id": config["env_id"],
                "robot_id": config["robot_id"],
                "task_family": "cooperative",
                "task_mode": TASK_MODE,
                "joint_names": list(JOINT_NAMES),
                "seed": seed,
            },
            sort_keys=False,
        )
    )
    (output / "generation_summary.json").write_text(json.dumps(manifest, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-contexts", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--start-task-id", type=int, default=0)
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.disable_fallback:
        config["fallback"]["enabled"] = False
    validate_config(config)
    count = args.num_contexts if args.num_contexts is not None else int(config["dataset"]["num_contexts"])
    if count <= 0 or args.start_task_id < 0:
        raise ValueError("num-contexts must be positive and start-task-id nonnegative")
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    output = args.output_dir or Path(config["output_dir"])
    if args.dry_run:
        print(
            yaml.safe_dump(
                {
                    "task_mode": TASK_MODE,
                    "output_dir": str(output),
                    "num_contexts": count,
                    "solutions_per_task_target": config["dataset"]["solutions_per_task_target"],
                    "object_proposals_per_task": config["object_planning"]["proposals_per_task"],
                    "fallback": config["fallback"],
                },
                sort_keys=False,
            )
        )
        return 0
    generator = MarvinWarehouseCooperativeGenerator(config, seed)
    try:
        paths, metadata, contexts = generator.generate(count, args.start_task_id)
        write_dataset(output, config, paths, metadata, contexts, seed, generator.stats)
    finally:
        generator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
