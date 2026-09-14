"""PyBullet-only mesh gate for the Marvin GPU generation pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p
from pybullet_utils import bullet_client
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
import torch
import yaml

from torch_robotics.environments.env_warehouse_marvin_bimanual import (
    EnvWarehouseMarvinBimanual,
)
from torch_robotics.environments.primitives import MultiBoxField, MultiSphereField


JOINT_NAMES = tuple(
    [f"Joint{index}_L" for index in range(1, 8)]
    + [f"Joint{index}_R" for index in range(1, 8)]
)


def _numpy(value):
    return torch.as_tensor(value).detach().cpu().numpy()


def _densify(path, max_joint_step):
    path = np.asarray(path, dtype=float)
    samples = [path[:1]]
    for start, goal in zip(path[:-1], path[1:]):
        count = max(
            1, int(np.ceil(np.max(np.abs(goal - start)) / max_joint_step))
        )
        samples.append(
            start + np.linspace(0.0, 1.0, count + 1)[1:, None] * (goal - start)
        )
    return np.concatenate(samples)


class MarvinPyBulletAuditor:
    """Own exactly one private DIRECT client and no OMPL objects."""

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
        self.client = bullet_client.BulletClient(connection_mode=p.DIRECT, options="")
        self.client.setGravity(0, 0, 0)
        self.robot_id = self.client.loadURDF(
            str(robot_root / "marvin_pika_bimanual_mpd.urdf"),
            (0, 0, 0),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        all_joints = range(self.client.getNumJoints(self.robot_id))
        self.joint_indices = [
            index
            for index in all_joints
            if self.client.getJointInfo(self.robot_id, index)[2] != p.JOINT_FIXED
        ]
        names = tuple(
            self.client.getJointInfo(self.robot_id, index)[1].decode()
            for index in self.joint_indices
        )
        if names != JOINT_NAMES:
            raise ValueError("PyBullet joint order differs from the dataset contract")
        self.bounds = np.asarray(
            [
                (
                    self.client.getJointInfo(self.robot_id, index)[8],
                    self.client.getJointInfo(self.robot_id, index)[9],
                )
                for index in self.joint_indices
            ]
        )
        link_indices = {
            self.client.getBodyInfo(self.robot_id)[0].decode(): -1
        }
        for index in range(self.client.getNumJoints(self.robot_id)):
            link_indices[self.client.getJointInfo(self.robot_id, index)[12].decode()] = index
        pairs = yaml.safe_load((config_root / "self_collision_pairs.yaml").read_text())[
            "pairs"
        ]
        disabled = {
            tuple(pair)
            for pair in self.config.get("disabled_self_collision_pairs", [])
        }
        self.self_pairs = [
            (link_indices[first], link_indices[second])
            for first, second in pairs
            if (first, second) not in disabled and (second, first) not in disabled
        ]
        self.obstacles = []
        self._build_obstacles()

    def _build_obstacles(self):
        tensor_args = {"device": "cpu", "dtype": torch.float32}
        environment = EnvWarehouseMarvinBimanual(
            precompute_sdf_obj_fixed=False,
            precompute_sdf_obj_extra=False,
            tensor_args=tensor_args,
        )
        for objects in (
            environment.get_obj_fixed_list(),
            environment.get_obj_extra_list(),
        ):
            for obj in objects:
                position, orientation_matrix = obj.get_position_orientation()
                position = _numpy(position)
                orientation_matrix = _numpy(orientation_matrix)
                orientation = Rotation.from_matrix(orientation_matrix).as_quat()
                for primitive in obj.get_all_single_primitives():
                    if isinstance(primitive, MultiBoxField):
                        centers = np.atleast_2d(_numpy(primitive.centers))
                        sizes = np.atleast_2d(_numpy(primitive.sizes))
                        for center, size in zip(centers, sizes):
                            center_world = orientation_matrix @ center + position
                            shape = self.client.createCollisionShape(
                                p.GEOM_BOX, halfExtents=(size / 2).tolist()
                            )
                            self.obstacles.append(
                                self.client.createMultiBody(
                                    baseMass=0,
                                    baseCollisionShapeIndex=shape,
                                    basePosition=center_world,
                                    baseOrientation=orientation,
                                )
                            )
                    elif isinstance(primitive, MultiSphereField):
                        centers = np.atleast_2d(_numpy(primitive.centers))
                        radii = np.atleast_1d(_numpy(primitive.radii))
                        if len(radii) == 1:
                            radii = np.repeat(radii, len(centers))
                        for center, radius in zip(centers, radii):
                            center_world = orientation_matrix @ center + position
                            shape = self.client.createCollisionShape(
                                p.GEOM_SPHERE, radius=float(radius)
                            )
                            self.obstacles.append(
                                self.client.createMultiBody(
                                    baseMass=0,
                                    baseCollisionShapeIndex=shape,
                                    basePosition=center_world,
                                )
                            )
                    else:
                        raise NotImplementedError(type(primitive).__name__)

    def state_valid(self, state):
        state = np.asarray(state, dtype=float)
        if state.shape != (14,) or not np.isfinite(state).all():
            return False
        if np.any((state < self.bounds[:, 0]) | (state > self.bounds[:, 1])):
            return False
        for index, value in zip(self.joint_indices, state):
            self.client.resetJointState(self.robot_id, index, float(value))
        for first, second in self.self_pairs:
            if self.client.getClosestPoints(
                self.robot_id,
                self.robot_id,
                distance=0.0,
                linkIndexA=first,
                linkIndexB=second,
            ):
                return False
        margin = float(self.config.get("min_distance_robot_env", 0.02))
        return not any(
            self.client.getClosestPoints(self.robot_id, obstacle, distance=margin)
            for obstacle in self.obstacles
        )

    def states_valid(self, states):
        return all(self.state_valid(state) for state in np.asarray(states))

    def endpoints_valid(self, q_start, q_goal):
        return self.state_valid(q_start) and self.state_valid(q_goal)

    def trajectory_valid(self, path, spline):
        step = float(self.config.get("collision_max_joint_step", 0.025))
        if not self.states_valid(_densify(path, step)):
            return False
        knots, coefficients, degree = spline
        evaluated = BSpline(
            np.asarray(knots), np.asarray(coefficients).T, int(degree)
        )(np.linspace(0.0, 1.0, int(self.config.get("spline_validation_points", 512))))
        return self.states_valid(_densify(evaluated, step))

    def close(self):
        client = getattr(self, "client", None)
        if client is not None:
            self.client = None
            client.disconnect()

