#!/usr/bin/env python3
"""Marvin independent paths: bounded pose IK or joint/FK rejection sampling.

One RRT attempt per sampled pair. Validate endpoints, dense paths and training
splines against both mesh and sphere collision models.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import weakref

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import h5py
import numpy as np
import pinocchio as pin
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch
import yaml

from pb_ompl.pb_ompl import PbOMPLRobot, fit_bspline_to_path, ob, og
from scripts.generate_data.generate_trajectories import GenerateDataOMPL
from torch_robotics.environments.env_warehouse_marvin_bimanual import EnvWarehouseMarvinBimanual
from torch_robotics.torch_planning_objectives.fields.distance_fields import CollisionObjectDistanceField

JOINT_NAMES = tuple([f"Joint{i}_L" for i in range(1, 8)] + [f"Joint{i}_R" for i in range(1, 8)])
ARM_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}
MODE_SCHEDULE = ("dual_independent",) * 3 + ("left_only", "right_only")
ARM_REGIONS = {"left": ("left_table", "left_cabinet"), "right": ("right_table", "right_cabinet")}
DIRECTIONS = (
    "placement_to_placement",
    "random_to_placement",
    "placement_to_random",
    "random_to_random",
)
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-independent.yaml"
)
PRE_RRT_FILTERS = ("none", "endpoint_clearance", "endpoint_clearance_and_sparse_line")
EE_GOAL_SCHEMA = "marvin_dual_pika_tcp/v1"
EE_GOAL_LINKS = ("left_pika_gripper_tcp", "right_pika_gripper_tcp")


class TaskSamplingBudgetExhausted(RuntimeError):
    """A fixed task/region contract could not be solved within its hard budget."""


def _axis_intervals(value, *, field):
    """Return one or more finite ``[low, high]`` intervals for one axis."""
    bounds = np.asarray(value, dtype=float)
    if (
        bounds.ndim != 2
        or bounds.shape[0] < 1
        or bounds.shape[1] != 2
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise ValueError(f"{field} must be a non-empty list of [low, high] intervals")
    return bounds


def _value_in_intervals(value, intervals, *, tolerance=0.0):
    bounds = np.asarray(intervals, dtype=float)
    return bool(
        np.any(
            (bounds[:, 0] - tolerance <= value)
            & (value <= bounds[:, 1] + tolerance)
        )
    )


def _sample_from_intervals(rng, intervals):
    """Choose an interval uniformly, then sample uniformly inside it."""
    bounds = np.asarray(intervals, dtype=float)
    selected = bounds[int(rng.integers(bounds.shape[0]))]
    return float(rng.uniform(selected[0], selected[1]))


def _workspace_boxes(value, *, field):
    """Normalize one legacy XYZ box or an explicit union of atomic boxes."""
    boxes = [value] if isinstance(value, dict) else value
    if not isinstance(boxes, (list, tuple)) or not boxes:
        raise ValueError(f"{field} must be an XYZ mapping or non-empty list of XYZ mappings")
    for index, box in enumerate(boxes):
        if not isinstance(box, dict):
            raise ValueError(f"{field}[{index}] must be an XYZ mapping")
        for axis in "xyz":
            _axis_intervals(box.get(axis, []), field=f"{field}[{index}].{axis}")
    return tuple(boxes)


def _position_in_workspace(position, region):
    boxes = (region,) if isinstance(region, dict) else region
    return any(
        all(_value_in_intervals(position[i], box[axis]) for i, axis in enumerate("xyz"))
        for box in boxes
    )


def _load_regions(config):
    regions = config.get("placement_regions", {})
    required = {
        name
        for arm in ARM_SLICES
        for name in config.get("arm_placement_regions", {}).get(arm, ARM_REGIONS[arm])
    }
    for name in required:
        region = regions.get(name, {})
        for component in ("translation", "rotation"):
            for axis in "xyz":
                _axis_intervals(
                    region.get(component, {}).get(axis, []),
                    field=f"{name}.{component}.{axis}",
                )
        base = np.asarray(region["rotation"].get("base"), dtype=float)
        if base.shape != (3, 3) or not np.allclose(base.T @ base, np.eye(3)) or not np.isclose(np.linalg.det(base), 1):
            raise ValueError(f"{name}.rotation.base must be a rotation matrix")
    for arm in ARM_SLICES:
        region = config.get("random_regions", {}).get(arm)
        if region is None:
            continue
        _workspace_boxes(region, field=f"random_regions.{arm}")
    return regions


def _placement_distributions(config):
    configured = config.get("arm_placement_regions")
    weighting = config.get("placement_region_weighting", "explicit_or_uniform")
    if weighting not in {"explicit_or_uniform", "volume"}:
        raise ValueError("placement_region_weighting must be explicit_or_uniform or volume")
    configured_boosts = config.get("placement_region_difficulty_boosts", {})
    result = {}
    for arm in ARM_SLICES:
        values = (configured or {}).get(arm, ARM_REGIONS[arm])
        if isinstance(values, dict):
            names = list(values)
        else:
            names = list(values)
        if not names or len(set(names)) != len(names):
            raise ValueError(f"arm_placement_regions.{arm} must contain unique region names")
        missing = set(names) - set(config.get("placement_regions", {}))
        if missing:
            raise ValueError(f"arm_placement_regions.{arm} contains missing regions: {sorted(missing)}")
        if isinstance(values, dict):
            weights = np.asarray(list(values.values()), dtype=float)
        else:
            if weighting == "volume":
                weights = np.asarray(
                    [
                        np.prod(
                            [
                                np.diff(
                                    _axis_intervals(
                                        config["placement_regions"][name]["translation"][axis],
                                        field=f"{name}.translation.{axis}",
                                    ),
                                    axis=1,
                                ).sum()
                                for axis in "xyz"
                            ]
                        )
                        for name in names
                    ],
                    dtype=float,
                )
            else:
                weights = np.ones(len(values), dtype=float)
        boosts = configured_boosts.get(arm, {})
        unknown_boosts = set(boosts) - set(names)
        if unknown_boosts:
            raise ValueError(
                f"placement_region_difficulty_boosts.{arm} contains unused regions: {sorted(unknown_boosts)}"
            )
        factors = np.asarray([float(boosts.get(name, 1.0)) for name in names])
        if not np.isfinite(factors).all() or np.any(factors < 1.0) or np.any(factors > 1.5):
            raise ValueError(
                f"placement_region_difficulty_boosts.{arm} values must lie in [1.0, 1.5]"
            )
        weights *= factors
        if weights.shape != (len(names),) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError(f"arm_placement_regions.{arm} weights must be finite and positive")
        result[arm] = (tuple(names), weights / weights.sum())
    return result


def _direction_schedule(direction_weights=None):
    if direction_weights is None:
        direction_weights = {"random_to_placement": 0.5, "placement_to_placement": 0.5}
    if set(direction_weights) - set(DIRECTIONS):
        raise ValueError(f"trajectory_direction_weights supports only {DIRECTIONS}")
    weights = np.asarray([float(direction_weights.get(name, 0.0)) for name in DIRECTIONS])
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("trajectory_direction_weights must be nonnegative and sum to 1")
    blocks = np.rint(weights * 20).astype(int)
    if not np.allclose(blocks / 20.0, weights):
        raise ValueError("trajectory_direction_weights must use 0.05 increments")
    # Smooth weighted round-robin avoids long runs of one direction while each
    # five-task block retains the 3:1:1 mode quota.
    remaining = blocks.copy()
    schedule = []
    for step in range(20):
        deficits = weights * (step + 1) - np.asarray([schedule.count(name) for name in DIRECTIONS])
        deficits[remaining == 0] = -np.inf
        index = int(np.argmax(deficits))
        schedule.append(DIRECTIONS[index])
        remaining[index] -= 1
    return tuple(schedule)


def validate_config(config):
    if config.get("env_id") != "EnvWarehouseMarvinBimanual" or config.get("robot_id") != "RobotMarvinBimanual":
        raise ValueError("requires EnvWarehouseMarvinBimanual / RobotMarvinBimanual")
    if config.get("task_family") != "independent":
        raise ValueError("only independent motion is implemented here")
    if config.get("sampler", "region_ik") not in {"region_ik", "joint_fk"}:
        raise ValueError("sampler must be region_ik or joint_fk")
    max_attempts = config.get(
        "max_attempts_per_task",
        config.get("max_attempts_per_trajectory", 30),
    )
    if int(max_attempts) != max_attempts or int(max_attempts) < 1:
        raise ValueError("max_attempts_per_task must be a positive integer")
    direction_weights = config.get("trajectory_direction_weights")
    if direction_weights is None and config.get("random_to_placement_fraction", 0.5) != 0.5:
        raise ValueError(
            "random_to_placement_fraction only supports the legacy 0.5 mix; "
            "use trajectory_direction_weights for four-way sampling"
        )
    _direction_schedule(direction_weights)
    if config.get("planner", "RRTConnect") != "RRTConnect":
        raise ValueError("this entrypoint uses RRTConnect")
    if not 0 < float(config.get("state_validity_resolution", 0.002)) <= 1:
        raise ValueError("state_validity_resolution must lie in (0, 1]")
    planner_range = config.get("planner_range", 0.35)
    if planner_range is not None and float(planner_range) <= 0:
        raise ValueError("planner_range must be positive")
    if not isinstance(config.get("simplify_path", False), bool):
        raise ValueError("simplify_path must be boolean")
    pre_rrt_filter = config.get("pre_rrt_filter", "none")
    if pre_rrt_filter not in PRE_RRT_FILTERS:
        raise ValueError(
            "pre_rrt_filter must be none, endpoint_clearance, or endpoint_clearance_and_sparse_line"
        )
    for key in ("pre_rrt_endpoint_environment_clearance", "pre_rrt_endpoint_self_clearance"):
        if float(config.get(key, 0.005)) < 0:
            raise ValueError(f"{key} must be nonnegative")
    if int(config.get("pre_rrt_sparse_line_points", 9)) < 1:
        raise ValueError("pre_rrt_sparse_line_points must be positive")
    sparse_fraction = float(config.get("pre_rrt_sparse_line_min_valid_fraction", 0.5))
    if not 0 <= sparse_fraction <= 1:
        raise ValueError("pre_rrt_sparse_line_min_valid_fraction must lie in [0, 1]")
    if not 0 <= config.get("ik_reference_seed_fraction", 0.5) <= 1:
        raise ValueError("ik_reference_seed_fraction must lie in [0, 1]")
    _load_regions(config)
    _placement_distributions(config)
    return config


def task_spec(task_id, direction_weights=None):
    # A 100-task period preserves the exact configurable direction quota; every
    # direction count is a five-task block with the 3:1:1 mode schedule.
    return MODE_SCHEDULE[task_id % 5], _direction_schedule(direction_weights)[(task_id // 5) % 20]


def active_arms(mode):
    if mode == "dual_independent":
        return ("left", "right")
    if mode in ("left_only", "right_only"):
        return (mode.split("_")[0],)
    raise ValueError(mode)


def file_sha256(path):
    with open(path, "rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MarvinWarehouseGenerator:
    def __init__(self, config, seed, progress_label=None):
        self.config = validate_config(config)
        self.rng = np.random.default_rng(seed)
        self.progress_label = progress_label
        self.stats = Counter()
        self.regions = _load_regions(config)
        self.placement_distributions = _placement_distributions(config)
        self.worker = GenerateDataOMPL(
            env_id=config["env_id"],
            robot_id=config["robot_id"],
            planner="RRTConnect",
            min_distance_robot_env=float(config.get("min_distance_robot_env", 0.02)),
            tensor_args={"device": "cpu", "dtype": torch.float32},
            pybullet_mode="DIRECT",
            debug=False,
            gripper=False,
        )
        self.interface = self.worker.pbompl_interface
        self.robot = self.worker.robot_pbompl
        self.torch_robot = self.worker.robot_tr
        self.torch_object_field = CollisionObjectDistanceField(
            self.torch_robot,
            df_obj_list_fn=self.worker.env_tr.get_df_obj_list,
            link_margins_for_object_collision_checking_tensor=self.torch_robot.link_collision_spheres_radii,
            cutoff_margin=float(config.get("min_distance_robot_env", 0.02)),
            tensor_args=self.torch_robot.tensor_args,
        )
        self._configure_marvin_self_collision_pairs()
        self.right_robot = PbOMPLRobot(
            self.worker.pybullet_client,
            self.robot.id,
            urdf_path=self.torch_robot.robot_urdf_file,
            link_name_ee=self.torch_robot.link_name_ee_right,
        )
        bullet_names = tuple(
            self.worker.pybullet_client.getJointInfo(self.robot.id, j)[1].decode() for j in self.robot.joint_idx
        )
        if bullet_names != JOINT_NAMES or tuple(self.robot.pinocchio_robot_model.names)[1:] != JOINT_NAMES:
            raise ValueError("PyBullet/Pinocchio joint order differs from dataset contract")
        # Use the compact source URDF for both TCPs. The generic worker's
        # temporary URDF contains >1000 collision-sphere frames.
        self._pin_model = self.right_robot.pinocchio_robot_model
        self._pin_frames = {
            arm: self._pin_model.getFrameId(getattr(self.torch_robot, f"link_name_ee_{arm}")) for arm in ARM_SLICES
        }
        self._pin_data = {arm: self._pin_model.createData() for arm in ARM_SLICES}
        self.deadline = float("inf")
        self._closed = False
        self._rrt_setups = {
            mode: self._build_rrt_setup(mode)
            for mode in ("dual_independent", "left_only", "right_only")
        }

    def _configure_marvin_self_collision_pairs(self):
        config = yaml.safe_load(Path(self.torch_robot.self_collision_pairs_file_path).read_text())
        disabled = {tuple(pair) for pair in self.config.get("disabled_self_collision_pairs", [])}
        indices = self.robot._link_name_to_index
        pairs = []
        for a, b in config["pairs"]:
            if a not in indices or b not in indices:
                raise ValueError(f"collision pair contains missing link: {a}, {b}")
            if (a, b) not in disabled and (b, a) not in disabled:
                pairs.append((indices[a], indices[b]))
        if not pairs:
            raise ValueError("empty Marvin collision pair list")
        self.interface.check_link_pairs = pairs

    def _build_rrt_setup(self, mode):
        """Create one persistent OMPL setup for a task mode in this worker."""
        indices = np.concatenate([np.arange(14)[ARM_SLICES[a]] for a in active_arms(mode)])
        space = ob.RealVectorStateSpace(len(indices))
        bounds = ob.RealVectorBounds(len(indices))
        for i, j in enumerate(indices):
            bounds.setLow(i, float(self.robot.joint_bounds_low_np[j]))
            bounds.setHigh(i, float(self.robot.joint_bounds_high_np[j]))
        space.setBounds(bounds)
        setup = og.SimpleSetup(space)

        # The callback is installed once and reads the current start state from
        # this mutable slot.  A weak reference avoids a Python reference cycle
        # self -> setup -> callback -> self during native teardown.
        state_context = {"q_start": np.zeros(14, dtype=float)}

        def expand(state, *, _indices=indices, _context=state_context):
            q = _context["q_start"].copy()
            q[_indices] = [state[i] for i in range(len(_indices))]
            return q

        owner_ref = weakref.ref(self)

        def is_valid(state, *, _owner_ref=owner_ref, _expand=expand):
            owner = _owner_ref()
            return owner is not None and owner.valid(_expand(state))

        validity_checker = ob.StateValidityCheckerFn(is_valid)
        setup.setStateValidityChecker(validity_checker)
        setup.getSpaceInformation().setStateValidityCheckingResolution(
            float(self.config.get("state_validity_resolution", 0.002))
        )
        planner = og.RRTConnect(setup.getSpaceInformation())
        planner_range = self.config.get("planner_range", 0.35)
        if planner_range is not None:
            planner.setRange(float(planner_range))
        setup.setPlanner(planner)
        self.stats["rrt_setup_builds"] += 1
        return {
            "indices": indices,
            "space": space,
            "setup": setup,
            "planner": planner,
            "validity_checker": validity_checker,
            "expand": expand,
            "state_context": state_context,
        }

    def _release_rrt_setups(self):
        """Destroy cached OMPL objects while the Bullet client is still live."""
        setups = getattr(self, "_rrt_setups", None)
        self._rrt_setups = {}
        if not setups:
            return
        for bundle in setups.values():
            setup = bundle.get("setup")
            if setup is not None:
                try:
                    setup.clear()
                except Exception:
                    # Cleanup must continue so the remaining native references
                    # are released before Bullet disconnects.
                    pass
            bundle.clear()
            del setup
        setups.clear()
        gc.collect()

    def _release_generic_pbompl_setup(self):
        """Release the unused full-14D setup owned by GenerateDataOMPL."""
        interface = getattr(self, "interface", None)
        if interface is None:
            return
        setup = getattr(interface, "ss", None)
        if setup is not None:
            try:
                setup.clear()
            except Exception:
                pass
        # PbOMPL holds a bound-method validity callback, so detach all Python
        # owners before dropping the last local SimpleSetup reference.
        for attr in ("planner", "ss", "si", "space"):
            if hasattr(interface, attr):
                setattr(interface, attr, None)
        del setup
        gc.collect()

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._release_rrt_setups()
        self._release_generic_pbompl_setup()
        worker = getattr(self, "worker", None)
        try:
            if worker is not None:
                worker.terminate()
        finally:
            self.interface = None
            self.worker = None

    def dual_ee_goal_pose(self, q_goal):
        """Return the two Pika TCP poses for a canonical 14-D joint goal.

        This is deliberately called only after both the raw path and fitted
        spline have passed validation.  It is metadata construction and does
        not alter endpoint sampling, IK, or 14-D RRT planning.
        """
        q_goal_t = torch.as_tensor(
            q_goal,
            dtype=self.torch_robot.q_pos_min.dtype,
            device=self.torch_robot.q_pos_min.device,
        )
        with torch.no_grad():
            poses = torch.stack(
                (self.torch_robot.fk_left(q_goal_t), self.torch_robot.fk_right(q_goal_t)),
                dim=0,
            )
        if poses.shape != (2, 3, 4) or not torch.isfinite(poses).all():
            raise ValueError(f"invalid dual Pika TCP pose: {tuple(poses.shape)}")
        return poses.detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def _torch_state_valid(self, q):
        states = np.atleast_2d(q)
        if not np.isfinite(states).all():
            return False
        for start in range(0, len(states), 32):
            qt = torch.as_tensor(states[start : start + 32], **self.torch_robot.tensor_args)
            if ((qt < self.torch_robot.q_pos_min) | (qt > self.torch_robot.q_pos_max)).any():
                return False
            positions = self.collision_positions(qt)
            self_collision = self.torch_robot.df_collision_self.compute_cost(qt, positions, field_type="occupancy")
            object_collision = self.torch_object_field.compute_cost(qt, positions, field_type="occupancy")
            if (self_collision | object_collision).any():
                return False
        return True

    def collision_positions(self, q):
        # Expand exact fine-sphere centers from their physical parent links.
        # Traversing >1000 fixed sphere frames per RRT state is unnecessary.
        robot = self.torch_robot
        parent_poses = torch.stack(robot.fk_collision_sphere_parent_links(q), dim=1)
        selected = parent_poses[:, robot.collision_sphere_parent_indices]
        return (
            torch.einsum("bsij,sj->bsi", selected[..., :3, :3], robot.collision_sphere_local_positions)
            + selected[..., :3, 3]
        )

    def valid(self, q):
        return self.interface.is_state_valid(np.asarray(q), check_bounds=True) and self._torch_state_valid(q)

    @torch.no_grad()
    def endpoint_clearances(self, q_start, q_goal):
        """Return minimum clearance beyond the already enforced margins."""
        states = np.stack([q_start, q_goal])
        qt = torch.as_tensor(states, **self.torch_robot.tensor_args)
        positions = self.collision_positions(qt)
        object_distances = self.torch_object_field.object_signed_distances(positions)
        object_margins = (
            self.torch_robot.link_collision_spheres_radii
            + float(self.torch_object_field.cutoff_margin)
        )
        environment = object_distances - object_margins
        self_distances = self.torch_robot.df_collision_self.compute_embodiment_signed_distances(qt, positions)
        return float(environment.min().item()), float(self_distances.min().item())

    def pre_rrt_diagnostics(self, q_start, q_goal, include_sparse_line):
        environment, self_clearance = self.endpoint_clearances(q_start, q_goal)
        result = {
            "endpoint_environment_clearance": environment,
            "endpoint_self_clearance": self_clearance,
        }
        if include_sparse_line:
            count = int(self.config.get("pre_rrt_sparse_line_points", 9))
            fractions = np.linspace(0.0, 1.0, count + 2)[1:-1]
            samples = q_start[None] + fractions[:, None] * (q_goal - q_start)[None]
            valid = [self.interface.is_state_valid(q, check_bounds=True) for q in samples]
            result["sparse_line_valid_fraction"] = float(np.mean(valid))
        return result

    def pre_rrt_accept(self, q_start, q_goal):
        mode = self.config.get("pre_rrt_filter", "none")
        if mode == "none":
            return True, {"pre_rrt_filter": mode, "pre_rrt_rejection_reason": "accepted"}
        started = time.perf_counter()
        diagnostics = self.pre_rrt_diagnostics(
            q_start,
            q_goal,
            include_sparse_line=mode == "endpoint_clearance_and_sparse_line",
        )
        environment_ok = diagnostics["endpoint_environment_clearance"] >= float(
            self.config.get("pre_rrt_endpoint_environment_clearance", 0.005)
        )
        self_ok = diagnostics["endpoint_self_clearance"] >= float(
            self.config.get("pre_rrt_endpoint_self_clearance", 0.005)
        )
        sparse_ok = diagnostics.get("sparse_line_valid_fraction", 1.0) >= float(
            self.config.get("pre_rrt_sparse_line_min_valid_fraction", 0.5)
        )
        if not environment_ok:
            reason = "endpoint_environment_clearance"
        elif not self_ok:
            reason = "endpoint_self_clearance"
        elif not sparse_ok:
            reason = "sparse_line"
        else:
            reason = "accepted"
        accepted = reason == "accepted"
        elapsed = time.perf_counter() - started
        self.stats["pre_rrt_filter_calls"] += 1
        self.stats["pre_rrt_filter_seconds"] += elapsed
        self.stats[f"pre_rrt_{'accepted' if accepted else 'rejected'}"] += 1
        if not accepted:
            self.stats[f"pre_rrt_rejected_{reason}"] += 1
        diagnostics.update(
            pre_rrt_filter=mode,
            pre_rrt_filter_seconds=elapsed,
            pre_rrt_rejection_reason=reason,
        )
        return accepted, diagnostics

    def _random_valid_state(self):
        # Factor the independent arm proposals; drawing both full joint
        # vectors at once multiplies two small workspace acceptance rates.
        # Home is only an intermediate reference, NEVER a returned fallback.
        q = np.zeros(14)
        budget = int(self.config.get("state_sample_tries", 2000))
        for arm in self.rng.permutation(["left", "right"]):
            sl = ARM_SLICES[arm]
            region = self.config.get("random_regions", {}).get(arm)
            found = False
            while budget > 0 and time.perf_counter() < self.deadline:
                budget -= 1
                self.stats["random_candidates"] += 1
                candidate = q.copy()
                candidate[sl] = self.rng.uniform(
                    self.robot.joint_bounds_low_np[sl], self.robot.joint_bounds_high_np[sl]
                )
                position = self._pose(candidate, arm).translation
                if region is not None and not _position_in_workspace(position, region):
                    continue
                if self.valid(candidate):
                    q = candidate
                    found = True
                    break
            if not found:
                return None
        return q

    def _pose(self, q, arm):
        data = self._pin_data[arm]
        pin.framesForwardKinematics(self._pin_model, data, q)
        return data.oMf[self._pin_frames[arm]].copy()

    def pose_in_region(self, q, arm, name):
        pose = self._pose(q, arm)
        region = self.regions[name]
        if not all(
            _value_in_intervals(pose.translation[i], region["translation"][a])
            for i, a in enumerate("xyz")
        ):
            return False
        angles = Rotation.from_matrix(np.asarray(region["rotation"]["base"]).T @ pose.rotation).as_euler(
            "xyz", degrees=True
        )
        return all(
            _value_in_intervals(angles[i], region["rotation"][a], tolerance=1e-6)
            for i, a in enumerate("xyz")
        )

    def _sample_pose(self, name):
        region = self.regions[name]
        pos = np.array(
            [_sample_from_intervals(self.rng, region["translation"][a]) for a in "xyz"]
        )
        angles = [
            _sample_from_intervals(self.rng, region["rotation"][a]) for a in "xyz"
        ]
        rot = np.asarray(region["rotation"]["base"]) @ Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        return pos, rot

    def _target_state(self, q_reference, arm, region_name):
        sampler = self.config.get("sampler", "region_ik")
        tries = int(self.config.get("ik_tries", 30) if sampler == "region_ik" else self.config.get("fk_tries", 20000))
        sl = ARM_SLICES[arm]
        low, high = self.robot.joint_bounds_low_np[sl], self.robot.joint_bounds_high_np[sl]
        started = time.perf_counter()
        try:
            for _ in range(tries):
                if time.perf_counter() >= self.deadline:
                    return None
                self.stats["target_candidates"] += 1
                q = np.array(q_reference, dtype=float).copy()
                q[sl] = self.rng.uniform(low, high)
                if sampler == "region_ik":
                    # Mix a local IK branch with uniform restarts. Local seeds
                    # help RRT connectivity; uniform seeds retain redundancy.
                    if self.rng.random() < self.config.get("ik_reference_seed_fraction", 0.5):
                        q[sl] = np.clip(np.asarray(q_reference)[sl], low, high)
                    pos, rot = self._sample_pose(region_name)

                    def residual(x):
                        q[sl] = x
                        pose = self._pose(q, arm)
                        return np.r_[pose.translation - pos, Rotation.from_matrix(rot.T @ pose.rotation).as_rotvec()]

                    result = least_squares(
                        residual,
                        q[sl].copy(),
                        bounds=(low, high),
                        max_nfev=int(self.config.get("ik_iterations", 100)),
                        ftol=1e-6,
                        xtol=1e-6,
                        gtol=1e-6,
                    )
                    q[sl] = result.x
                    if np.linalg.norm(result.fun[:3]) > self.config.get(
                        "ik_position_tolerance", 0.003
                    ) or np.linalg.norm(result.fun[3:]) > np.deg2rad(
                        self.config.get("ik_orientation_tolerance_deg", 2.0)
                    ):
                        continue
                if not self.pose_in_region(q, arm, region_name):
                    continue
                self.stats["target_pose_hits"] += 1
                if self.valid(q):
                    self.stats["target_accepted"] += 1
                    return q.copy()
            return None
        finally:
            self.stats["target_seconds"] += time.perf_counter() - started

    def _arm_region(self, arm, exclude=None):
        names, weights = self.placement_distributions[arm]
        if exclude is not None and len(names) > 1:
            keep = np.asarray([name != exclude for name in names])
            names = tuple(name for name, selected in zip(names, keep) if selected)
            weights = weights[keep] / weights[keep].sum()
        return str(self.rng.choice(names, p=weights))

    def _sample_random_endpoint(self, q_reference, mode):
        """Resample active arm joints inside random workspace; freeze inactive arms."""
        q = np.asarray(q_reference, dtype=float).copy()
        budget = int(self.config.get("state_sample_tries", 2000))
        for arm in self.rng.permutation(active_arms(mode)):
            sl = ARM_SLICES[arm]
            region = self.config.get("random_regions", {}).get(arm)
            found = False
            while budget > 0 and time.perf_counter() < self.deadline:
                budget -= 1
                self.stats["random_candidates"] += 1
                candidate = q.copy()
                candidate[sl] = self.rng.uniform(
                    self.robot.joint_bounds_low_np[sl], self.robot.joint_bounds_high_np[sl]
                )
                position = self._pose(candidate, arm).translation
                if region is not None and not _position_in_workspace(position, region):
                    continue
                if self.valid(candidate):
                    q, found = candidate, True
                    break
            if not found:
                return None
        return q

    def _sample_endpoint(self, q_reference, mode, regions):
        q = np.array(q_reference).copy()
        for arm in active_arms(mode):
            q = self._target_state(q, arm, regions[arm])
            if q is None:
                return None
        return q if self.valid(q) else None

    def _task_regions(self, mode, direction):
        """Choose the categorical task once; retries must keep this contract."""
        arms = active_arms(mode)
        source = {arm: "inactive" for arm in ARM_SLICES}
        goal = {arm: "inactive" for arm in ARM_SLICES}
        if direction in {"placement_to_placement", "placement_to_random"}:
            source.update({arm: self._arm_region(arm) for arm in arms})
        else:
            source.update({arm: "random" for arm in arms})
        if direction in {"random_to_placement", "placement_to_placement"}:
            differ = bool(self.config.get("placement_goal_must_differ_from_source_region", True))
            goal.update(
                {
                    arm: self._arm_region(
                        arm,
                        source[arm] if differ and source[arm] != "random" else None,
                    )
                    for arm in arms
                }
            )
        else:
            goal.update({arm: "random" for arm in arms})
        return source, goal

    def _sample_task(self, mode, direction, source=None, goal=None):
        if source is None or goal is None:
            source, goal = self._task_regions(mode, direction)
        q_start = self._random_valid_state()
        if q_start is None:
            return None
        arms = active_arms(mode)
        if direction in {"placement_to_placement", "placement_to_random"}:
            q_start = self._sample_endpoint(q_start, mode, source)
            if q_start is None:
                return None
        if direction in {"random_to_placement", "placement_to_placement"}:
            q_goal = self._sample_endpoint(q_start, mode, goal)
        elif direction in {"placement_to_random", "random_to_random"}:
            q_goal = self._sample_random_endpoint(q_start, mode)
        else:
            raise ValueError(f"unsupported trajectory direction: {direction}")
        if q_goal is None or any(
            np.linalg.norm(q_goal[ARM_SLICES[a]] - q_start[ARM_SLICES[a]])
            < self.config.get("min_active_joint_delta", 0.08)
            for a in arms
        ):
            return None
        return q_start, q_goal, source, goal

    def plan_once(self, q_start, q_goal, mode):
        """A 7D subspace freezes the inactive arm during ALL RRT operations."""
        try:
            bundle = self._rrt_setups[mode]
        except KeyError as error:
            raise ValueError(f"unsupported RRT task mode: {mode}") from error
        indices = bundle["indices"]
        space = bundle["space"]
        setup = bundle["setup"]
        expand = bundle["expand"]
        bundle["state_context"]["q_start"] = np.asarray(q_start, dtype=float).copy()

        # Reuse the mode-specific state space, validity callback and planner.
        # clear() discards the previous problem/tree while retaining the setup.
        setup.clear()
        start, goal = ob.State(space), ob.State(space)
        for i, j in enumerate(indices):
            start[i], goal[i] = float(q_start[j]), float(q_goal[j])
        setup.setStartAndGoalStates(start, goal)
        self.stats["rrt_attempts"] += 1
        before = time.perf_counter()
        setup.solve(float(self.config.get("planner_allowed_time", 10.0)))
        self.stats["rrt_seconds"] += time.perf_counter() - before
        if not setup.haveExactSolutionPath():
            self.stats["rrt_no_exact_solution"] += 1
            return None
        self.stats["rrt_exact"] += 1
        path = setup.getSolutionPath()
        # Disabled by default for Marvin: the paired 100-path benchmark found
        # that a nominal maxTime=0.1 can spend minutes in one collision-heavy
        # motion check. Retain the opt-in only for controlled experiments.
        if self.config.get("simplify_path", False):
            before = time.perf_counter()
            og.PathSimplifier(setup.getSpaceInformation()).simplify(path, maxTime=0.1)
            self.stats["path_simplify_seconds"] += time.perf_counter() - before
        before = time.perf_counter()
        path.interpolate(int(self.config.get("interpolate_num", 128)))
        raw = np.array([expand(state) for state in path.getStates()])
        # OMPL interpolate never removes vertices; enforce a fixed shape.
        t = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
        t, unique = np.unique(t, return_index=True)
        raw = raw[unique]
        if len(t) < 2:
            return None
        path = np.stack(
            [
                np.interp(np.linspace(0, t[-1], int(self.config.get("interpolate_num", 128))), t, raw[:, j])
                for j in range(14)
            ],
            axis=1,
        )
        self.stats["path_resample_seconds"] += time.perf_counter() - before
        if not self.path_valid(path, stats_prefix="path"):
            self.stats["path_rejected"] += 1
            return None
        return path

    def path_valid(self, path, stats_prefix="path"):
        started = time.perf_counter()
        step = float(self.config.get("collision_max_joint_step", 0.025))
        samples = [path[:1]]
        for a, b in zip(path[:-1], path[1:]):
            n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
            samples.append(a + np.linspace(0.0, 1.0, n + 1)[1:, None] * (b - a))
        dense = np.concatenate(samples)
        if hasattr(self, "stats"):
            self.stats[f"{stats_prefix}_densify_seconds"] += time.perf_counter() - started
            self.stats[f"{stats_prefix}_dense_states"] += len(dense)
        started = time.perf_counter()
        torch_valid = self._torch_state_valid(dense)
        if hasattr(self, "stats"):
            self.stats[f"{stats_prefix}_torch_audit_seconds"] += time.perf_counter() - started
        if not torch_valid:
            return False
        started = time.perf_counter()
        bullet_valid = all(self.interface.is_state_valid(q, check_bounds=True) for q in dense)
        if hasattr(self, "stats"):
            self.stats[f"{stats_prefix}_pybullet_audit_seconds"] += time.perf_counter() - started
        return bullet_valid

    def validated_spline(self, path):
        started = time.perf_counter()
        try:
            tt, cc, k = fit_bspline_to_path(
                path,
                bspline_degree=int(self.config.get("bspline_degree", 5)),
                bspline_num_control_points=int(self.config.get("bspline_num_control_points", 22)),
                bspline_zero_vel_at_start_and_goal=True,
                bspline_zero_acc_at_start_and_goal=True,
            )
            cc = np.asarray(cc)
            spline_path = BSpline(tt, cc.T, k)(np.linspace(0, 1, int(self.config.get("spline_validation_points", 512))))
            if hasattr(self, "stats"):
                self.stats["spline_fit_evaluate_seconds"] += time.perf_counter() - started
            if not np.allclose(spline_path[[0, -1]], path[[0, -1]], atol=1e-5) or not self.path_valid(
                spline_path, stats_prefix="spline"
            ):
                return None
            return np.asarray(tt), cc, k
        except (ValueError, np.linalg.LinAlgError):
            if hasattr(self, "stats"):
                self.stats["spline_fit_evaluate_seconds"] += time.perf_counter() - started
            return None

    def generate(self, num_trajectories, start_task_id=0):
        paths, metadata = [], []
        pending_task_id = None
        pending_spec = None
        task_attempts = 0
        max_attempts = int(
            self.config.get(
                "max_attempts_per_task",
                self.config.get("max_attempts_per_trajectory", 30),
            )
        )
        while len(paths) < num_trajectories:
            task_id = start_task_id + len(paths)
            if pending_task_id != task_id:
                mode, direction = task_spec(task_id, self.config.get("trajectory_direction_weights"))
                source, goal = self._task_regions(mode, direction)
                pending_task_id = task_id
                pending_spec = (mode, direction, source, goal)
                task_attempts = 0
            mode, direction, source, goal = pending_spec
            if task_attempts >= max_attempts:
                raise TaskSamplingBudgetExhausted(
                    f"task {task_id} exhausted {max_attempts} attempts; "
                    f"completed={len(paths)}/{num_trajectories}, mode={mode}, "
                    f"direction={direction}, source={source}, goal={goal}, stats={dict(self.stats)}"
                )
            task_attempts += 1
            self.stats["task_attempts"] += 1
            self.deadline = time.perf_counter() + float(self.config.get("task_timeout_seconds", 300))
            sampling_started = time.perf_counter()
            sampled = self._sample_task(mode, direction, source, goal)
            self.stats["endpoint_sampling_seconds"] += time.perf_counter() - sampling_started
            self.stats["task_samples"] += 1
            if sampled is None:
                self.stats["endpoint_failures"] += 1
                continue
            q_start, q_goal, source, goal = sampled
            pre_rrt_accepted, pre_rrt = self.pre_rrt_accept(q_start, q_goal)
            if not pre_rrt_accepted:
                continue
            before = time.perf_counter()
            path = self.plan_once(q_start, q_goal, mode)
            if path is None:  # Abandon this pair; next iteration samples NEW endpoints.
                continue
            spline_started = time.perf_counter()
            spline = self.validated_spline(path)
            self.stats["spline_total_seconds"] += time.perf_counter() - spline_started
            if spline is None:
                self.stats["spline_rejected"] += 1
                continue
            ee_goal_pose = self.dual_ee_goal_pose(q_goal)
            paths.append(path.copy())
            item = dict(
                task_id=task_id,
                task_mode=mode,
                direction=direction,
                q_start=q_start,
                q_goal=q_goal,
                planning_time=time.perf_counter() - before,
                bspline=spline,
                ee_goal_pose=ee_goal_pose,
                joint_path_length=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                **pre_rrt,
            )
            for arm in ARM_SLICES:
                item[f"source_region_{arm}"] = source.get(arm, "inactive")
                item[f"goal_region_{arm}"] = goal.get(arm, "inactive")
            metadata.append(item)
            progress_label = getattr(self, "progress_label", None)
            prefix = f"[{progress_label}] " if progress_label else ""
            print(f"{prefix}generated {len(paths)}/{num_trajectories}: {mode}, {direction}", flush=True)
        return paths, metadata


def _write_dataset(output, config, paths, metadata, seed, stats=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "dataset_merged.hdf5").exists():
        raise FileExistsError(f"dataset already exists: {output}")
    masks = np.zeros((len(paths), 14), dtype=bool)
    ee_masks = np.zeros((len(paths), 2), dtype=bool)
    for i, item in enumerate(metadata):
        for arm in active_arms(item["task_mode"]):
            masks[i, ARM_SLICES[arm]] = True
            ee_masks[i, 0 if arm == "left" else 1] = True
    ee_goal_poses = np.asarray([item["ee_goal_pose"] for item in metadata], dtype=np.float32)
    if ee_goal_poses.shape != (len(paths), 2, 3, 4):
        raise ValueError(f"ee_goal_pose must have shape (N, 2, 3, 4), got {ee_goal_poses.shape}")
    with h5py.File(output / "dataset_merged.hdf5", "x") as handle:
        handle.create_dataset("sol_path", data=np.stack(paths), compression="gzip")
        for key in ("q_start", "q_goal", "task_id", "planning_time"):
            handle.create_dataset(key, data=np.asarray([m[key] for m in metadata]))
        handle.create_dataset(
            "joint_path_length",
            data=np.asarray([m.get("joint_path_length", np.nan) for m in metadata]),
        )
        for key in (
            "task_mode",
            "direction",
            "source_region_left",
            "source_region_right",
            "goal_region_left",
            "goal_region_right",
        ):
            handle.create_dataset(key, data=[m[key] for m in metadata], dtype=h5py.string_dtype())
        handle.create_dataset("active_joint_mask", data=masks)
        handle.create_dataset("ee_goal_pose", data=ee_goal_poses)
        handle.create_dataset("active_ee_mask", data=ee_masks)
        for i, key in enumerate(("bspline_params_tt", "bspline_params_cc", "bspline_params_k")):
            handle.create_dataset(key, data=np.asarray([m["bspline"][i] for m in metadata]))
        handle.attrs["joint_names"] = np.asarray(JOINT_NAMES, dtype=object)
        handle.attrs["scene_version"] = EnvWarehouseMarvinBimanual.scene_version
        handle.attrs["validated_splines"] = True
        handle.attrs["ee_goal_schema"] = EE_GOAL_SCHEMA
        handle.attrs["ee_goal_links"] = np.asarray(EE_GOAL_LINKS, dtype=object)
    args = dict(
        env_id=config["env_id"],
        robot_id=config["robot_id"],
        min_distance_robot_env=float(config.get("min_distance_robot_env", 0.02)),
        planner="RRTConnect",
        joint_names=list(JOINT_NAMES),
        task_family="independent",
        task_mode="dual_independent",
        seed=seed,
    )
    (output / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False))
    manifest = dict(
        schema="marvin_bimanual_warehouse_dataset/v3",
        ee_goal_schema=EE_GOAL_SCHEMA,
        ee_goal_links=list(EE_GOAL_LINKS),
        scene_version=EnvWarehouseMarvinBimanual.scene_version,
        num_trajectories=len(paths),
        joint_names=list(JOINT_NAMES),
        task_counts=dict(Counter(m["task_mode"] for m in metadata)),
        direction_counts=dict(Counter(m["direction"] for m in metadata)),
        region_counts={
            key: dict(Counter(m[key] for m in metadata))
            for key in (
                "source_region_left",
                "source_region_right",
                "goal_region_left",
                "goal_region_right",
            )
        },
        dataset_sha256=file_sha256(output / "dataset_merged.hdf5"),
        stats=dict(stats or {}),
    )
    robot_dir = Path(__file__).resolve().parents[2] / "mpd/torch_robotics/torch_robotics/data/urdf/robots/marvin"
    manifest["model_sha256"] = file_sha256(robot_dir / "marvin_pika_bimanual_mpd.urdf")
    manifest["asset_sha256"] = yaml.safe_load((robot_dir / "pika_assets.lock.yaml").read_text())["asset_sha256"]
    (output / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (output / "generation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "generation_summary.json").write_text(json.dumps(manifest, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--num-trajectories", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sampler", choices=("region_ik", "joint_fk"))
    parser.add_argument("--pre-rrt-filter", choices=PRE_RRT_FILTERS)
    parser.add_argument("--start-task-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.sampler:
        config["sampler"] = args.sampler
    if args.pre_rrt_filter:
        config["pre_rrt_filter"] = args.pre_rrt_filter
    validate_config(config)
    n = args.num_trajectories if args.num_trajectories is not None else int(config.get("num_trajectories", 1000))
    if n < 1 or n % 10 or args.start_task_id < 0 or args.start_task_id % 10:
        raise ValueError(
            "count/start-task-id must be multiples of 10 to retain mode-balanced worker blocks; "
            "the configured direction mix is exact over complete 100-task periods"
        )
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    output = args.output_dir or Path(config["output_dir"])
    if args.dry_run:
        placement_distributions = _placement_distributions(config)
        print(
            yaml.safe_dump(
                dict(
                    output_dir=str(output),
                    num_trajectories=n,
                    seed=seed,
                    sampler=config.get("sampler"),
                    pre_rrt_filter=config.get("pre_rrt_filter", "none"),
                    direction_counts_per_100=dict(
                        Counter(
                            task_spec(i, config.get("trajectory_direction_weights"))[1]
                            for i in range(100)
                        )
                    ),
                    random_regions=config.get("random_regions", {}),
                    placement_region_probabilities={
                        arm: {name: float(weight) for name, weight in zip(names, weights)}
                        for arm, (names, weights) in placement_distributions.items()
                    },
                    regions=_load_regions(config),
                )
            )
        )
        return 0
    if (output / "dataset_merged.hdf5").exists():
        raise FileExistsError(output)
    generator = MarvinWarehouseGenerator(config, seed)
    try:
        paths, metadata = generator.generate(n, args.start_task_id)
        _write_dataset(output, config, paths, metadata, seed, generator.stats)
    finally:
        generator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
