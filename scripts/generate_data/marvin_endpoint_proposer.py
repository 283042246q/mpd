"""Collision-agnostic Marvin endpoint proposals using only Pinocchio/SciPy.

Candidates from this module are not accepted endpoints.  They must pass the
GPU sphere model and the independent PyBullet mesh worker before planning.
Keeping collision libraries out of this process makes it safe to scale IK
proposal workers without recreating the old OMPL worker pool.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml


JOINT_NAMES = tuple(
    [f"Joint{index}_L" for index in range(1, 8)]
    + [f"Joint{index}_R" for index in range(1, 8)]
)
ARM_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}


def active_arms(mode):
    if mode == "dual_independent":
        return ("left", "right")
    if mode in ("left_only", "right_only"):
        return (mode.split("_", 1)[0],)
    raise ValueError(f"unsupported task mode: {mode}")


def _in_intervals(value, intervals, tolerance=0.0):
    bounds = np.asarray(intervals, dtype=float)
    return bool(
        np.any(
            (bounds[:, 0] - tolerance <= value)
            & (value <= bounds[:, 1] + tolerance)
        )
    )


def _sample_intervals(rng, intervals):
    bounds = np.asarray(intervals, dtype=float)
    selected = bounds[int(rng.integers(len(bounds)))]
    return float(rng.uniform(selected[0], selected[1]))


def _position_in_workspace(position, region):
    boxes = (region,) if isinstance(region, dict) else region
    return any(
        all(_in_intervals(position[index], box[axis]) for index, axis in enumerate("xyz"))
        for box in boxes
    )


class MarvinEndpointProposer:
    """Persistent, collision-free IK proposal worker."""

    def __init__(self, config):
        self.config = dict(config)
        repository = Path(__file__).resolve().parents[2]
        robot_root = (
            repository
            / "mpd/torch_robotics/torch_robotics/data/urdf/robots/marvin"
        )
        config_root = (
            repository
            / "mpd/torch_robotics/torch_robotics/data/configs/marvin/pika"
        )
        self.model = pin.buildModelFromUrdf(
            str(robot_root / "marvin_pika_bimanual_mpd.urdf")
        )
        if tuple(self.model.names[1:]) != JOINT_NAMES:
            raise ValueError("Pinocchio joint order differs from the dataset contract")
        self.data = {
            arm: self.model.createData() for arm in ARM_SLICES
        }
        self.frames = {
            "left": self.model.getFrameId("left_pika_gripper_tcp"),
            "right": self.model.getFrameId("right_pika_gripper_tcp"),
        }
        limits = yaml.safe_load((config_root / "joint_limits.yaml").read_text())
        if tuple(limits) != JOINT_NAMES:
            raise ValueError("joint-limit order differs from the dataset contract")
        self.low = np.asarray([limits[name]["qmin"] for name in JOINT_NAMES])
        self.high = np.asarray([limits[name]["qmax"] for name in JOINT_NAMES])
        self.regions = self.config["placement_regions"]

    def pose(self, q, arm):
        data = self.data[arm]
        pin.framesForwardKinematics(self.model, data, np.asarray(q, dtype=float))
        return data.oMf[self.frames[arm]].copy()

    def _pose_in_region(self, q, arm, name):
        pose = self.pose(q, arm)
        region = self.regions[name]
        if not all(
            _in_intervals(pose.translation[index], region["translation"][axis])
            for index, axis in enumerate("xyz")
        ):
            return False
        angles = Rotation.from_matrix(
            np.asarray(region["rotation"]["base"]).T @ pose.rotation
        ).as_euler("xyz", degrees=True)
        return all(
            _in_intervals(
                angles[index], region["rotation"][axis], tolerance=1e-6
            )
            for index, axis in enumerate("xyz")
        )

    def _sample_pose(self, rng, name):
        region = self.regions[name]
        position = np.asarray(
            [_sample_intervals(rng, region["translation"][axis]) for axis in "xyz"]
        )
        angles = [
            _sample_intervals(rng, region["rotation"][axis]) for axis in "xyz"
        ]
        rotation = np.asarray(region["rotation"]["base"]) @ Rotation.from_euler(
            "xyz", angles, degrees=True
        ).as_matrix()
        return position, rotation

    def _random_reference(self, rng, deadline):
        q = np.zeros(14, dtype=float)
        budget = int(self.config.get("state_sample_tries", 2000))
        for arm in rng.permutation(("left", "right")):
            joint_slice = ARM_SLICES[arm]
            region = self.config.get("random_regions", {}).get(arm)
            found = False
            while budget > 0 and time.perf_counter() < deadline:
                budget -= 1
                candidate = q.copy()
                candidate[joint_slice] = rng.uniform(
                    self.low[joint_slice], self.high[joint_slice]
                )
                if region is None or _position_in_workspace(
                    self.pose(candidate, arm).translation, region
                ):
                    q = candidate
                    found = True
                    break
            if not found:
                return None
        return q

    def _target_state(self, q_reference, arm, region_name, rng, deadline):
        sampler = self.config.get("sampler", "region_ik")
        tries = int(
            self.config.get("ik_tries", 30)
            if sampler == "region_ik"
            else self.config.get("fk_tries", 20000)
        )
        joint_slice = ARM_SLICES[arm]
        low, high = self.low[joint_slice], self.high[joint_slice]
        for _ in range(tries):
            if time.perf_counter() >= deadline:
                return None
            q = np.asarray(q_reference, dtype=float).copy()
            q[joint_slice] = rng.uniform(low, high)
            if sampler == "region_ik":
                if rng.random() < float(
                    self.config.get("ik_reference_seed_fraction", 0.5)
                ):
                    q[joint_slice] = np.clip(q_reference[joint_slice], low, high)
                position, rotation = self._sample_pose(rng, region_name)

                def residual(active):
                    q[joint_slice] = active
                    pose = self.pose(q, arm)
                    return np.r_[
                        pose.translation - position,
                        Rotation.from_matrix(rotation.T @ pose.rotation).as_rotvec(),
                    ]

                result = least_squares(
                    residual,
                    q[joint_slice].copy(),
                    bounds=(low, high),
                    max_nfev=int(self.config.get("ik_iterations", 100)),
                    ftol=1e-6,
                    xtol=1e-6,
                    gtol=1e-6,
                )
                q[joint_slice] = result.x
                if np.linalg.norm(result.fun[:3]) > float(
                    self.config.get("ik_position_tolerance", 0.003)
                ) or np.linalg.norm(result.fun[3:]) > np.deg2rad(
                    float(self.config.get("ik_orientation_tolerance_deg", 2.0))
                ):
                    continue
            if self._pose_in_region(q, arm, region_name):
                return q.copy()
        return None

    def _placement_endpoint(self, q_reference, mode, regions, rng, deadline):
        q = np.asarray(q_reference, dtype=float).copy()
        for arm in active_arms(mode):
            q = self._target_state(q, arm, regions[arm], rng, deadline)
            if q is None:
                return None
        return q

    def _random_endpoint(self, q_reference, mode, rng, deadline):
        q = np.asarray(q_reference, dtype=float).copy()
        budget = int(self.config.get("state_sample_tries", 2000))
        for arm in rng.permutation(active_arms(mode)):
            joint_slice = ARM_SLICES[arm]
            region = self.config.get("random_regions", {}).get(arm)
            found = False
            while budget > 0 and time.perf_counter() < deadline:
                budget -= 1
                candidate = q.copy()
                candidate[joint_slice] = rng.uniform(
                    self.low[joint_slice], self.high[joint_slice]
                )
                if region is None or _position_in_workspace(
                    self.pose(candidate, arm).translation, region
                ):
                    q = candidate
                    found = True
                    break
            if not found:
                return None
        return q

    def propose(self, mode, direction, source, goal, seed):
        rng = np.random.default_rng(int(seed))
        deadline = time.perf_counter() + float(
            self.config.get("task_timeout_seconds", 300)
        )
        q_start = self._random_reference(rng, deadline)
        if q_start is None:
            return None
        if direction in {"placement_to_placement", "placement_to_random"}:
            q_start = self._placement_endpoint(
                q_start, mode, source, rng, deadline
            )
            if q_start is None:
                return None
        if direction in {"random_to_placement", "placement_to_placement"}:
            q_goal = self._placement_endpoint(q_start, mode, goal, rng, deadline)
        elif direction in {"placement_to_random", "random_to_random"}:
            q_goal = self._random_endpoint(q_start, mode, rng, deadline)
        else:
            raise ValueError(f"unsupported direction: {direction}")
        if q_goal is None:
            return None
        if any(
            np.linalg.norm(q_goal[ARM_SLICES[arm]] - q_start[ARM_SLICES[arm]])
            < float(self.config.get("min_active_joint_delta", 0.08))
            for arm in active_arms(mode)
        ):
            return None
        return q_start, q_goal

    def ee_goal_pose(self, q_goal):
        poses = []
        for arm in ("left", "right"):
            pose = self.pose(q_goal, arm)
            poses.append(np.c_[pose.rotation, pose.translation])
        return np.asarray(poses, dtype=np.float32)

