"""Warehouse scene configured for Marvin's two 7-DoF arms.

The original :class:`EnvWarehouse` remains Panda-only. This scene uses box
primitives for two open cabinets and a central table, with larger shelf gaps
and a lower tabletop for Marvin/Pika's collision-aware workspace.
"""

from __future__ import annotations

import numpy as np
import torch

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
import torch_robotics.robots as tr_robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch


class EnvWarehouseMarvinBimanual(EnvBase):
    """Static table plus mirrored left/right cabinet shelves for Marvin."""

    scene_version = "marvin_warehouse_v2"

    @staticmethod
    def _shelf(tensor_args):
        # A real open cabinet: 80 cm wide, 30 cm deep; first accessible
        # compartment is z=[0.015, 0.40] in the robot base frame.
        centers = [[0.01, 0.15, 0.9], [0.79, 0.15, 0.9], [0.4, 0.31, 0.9]]
        sizes = [[0.02, 0.3, 1.8], [0.02, 0.3, 1.8], [0.76, 0.02, 1.8]]
        for z in (0.0, 0.8, 1.2, 1.6, 1.785):
            centers.append([0.4, 0.15, z + 0.0075])
            sizes.append([0.76, 0.3, 0.015])
        return ObjectField([MultiBoxField(np.array(centers), np.array(sizes), tensor_args=tensor_args)], "cabinet")

    def __init__(self, rotation_z_axis_deg: float = 0.0, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        rotation_z_axis_rad = np.deg2rad(rotation_z_axis_deg)
        perturbation = np.eye(4)
        perturbation[:2, :2] = np.array(
            [
                [np.cos(rotation_z_axis_rad), -np.sin(rotation_z_axis_rad)],
                [np.sin(rotation_z_axis_rad), np.cos(rotation_z_axis_rad)],
            ]
        )
        perturbation_th = to_torch(perturbation, **tensor_args)

        # Tabletop at z=-0.16; keep the shoulder/elbow sweep clear of its
        # rear edge. Regions are TCP volumes above this physical surface.
        table = ObjectField(
            [MultiBoxField(np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 0.5, 0.64]]), tensor_args=tensor_args)], "table"
        )
        table_transform = np.eye(4)
        table_transform[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        table_transform[:3, 3] = np.array([0.55, 0.0, -0.48])
        table_transform = perturbation @ table_transform
        table.set_position_orientation(table_transform[:3, 3], table_transform[:3, :3])

        # The shelves face the table.  Mirroring the right shelf keeps the
        # cabinet openings in the two arms' natural lateral workspaces.
        left_shelf = self._shelf(tensor_args)
        left_transform = np.eye(4)
        left_transform[:3, 3] = np.array([0.15, 0.62, -0.80])
        left_transform = perturbation @ left_transform
        left_shelf.set_position_orientation(left_transform[:3, 3], left_transform[:3, :3])

        right_shelf = self._shelf(tensor_args)
        right_transform = np.eye(4)
        right_transform[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        right_transform[:3, 3] = np.array([0.95, -0.62, -0.80])
        right_transform = perturbation @ right_transform
        right_shelf.set_position_orientation(right_transform[:3, 3], right_transform[:3, :3])

        left_shelf.name = "left_cabinet"
        right_shelf.name = "right_cabinet"
        obj_list = [table, left_shelf, right_shelf]
        if "obj_extra_list" in kwargs:
            for obj in kwargs["obj_extra_list"]:
                transform = perturbation_th @ obj.get_transformation_matrix()
                obj.set_position_orientation(transform[:3, 3], transform[:3, :3])

        super().__init__(
            limits=torch.tensor([[-1.2, -1.6, -1.1], [1.8, 1.6, 1.3]], **tensor_args),
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs,
        )

    @staticmethod
    def _planner_params():
        return dict(
            opt_iters=250,
            num_samples=64,
            sigma_start=1e-3,
            sigma_gp=1e-1,
            sigma_goal_prior=1e-3,
            sigma_coll=1e-4,
            step_size=5e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.1,
            sigma_start_sample=1e-3,
            sigma_goal_sample=1e-3,
            solver_params={"delta": 1e-2, "trust_region": True, "method": "cholesky"},
        )

    def get_gpmp2_params(self, robot=None):
        if not isinstance(robot, tr_robots.RobotMarvinBimanual):
            raise NotImplementedError("EnvWarehouseMarvinBimanual requires RobotMarvinBimanual")
        return self._planner_params()

    def get_rrt_connect_params(self, robot=None):
        if not isinstance(robot, tr_robots.RobotMarvinBimanual):
            raise NotImplementedError("EnvWarehouseMarvinBimanual requires RobotMarvinBimanual")
        return dict(n_iters=20000, step_size=torch.pi / 100, n_radius=torch.pi / 3, n_pre_samples=60000, max_time=20)
