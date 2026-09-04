"""Warehouse scene configured for Marvin's two 7-DoF arms.

The original :class:`EnvWarehouse` remains Panda-only.  This scene keeps the
same table/shelf primitives but places two shelves on the left and right of a
central table so both Marvin end-effectors have a useful, collision-aware
workspace.
"""

from __future__ import annotations

import numpy as np
import torch

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.env_table_shelf import create_shelf_field, create_table_object_field
from torch_robotics.environments.primitives import ObjectField
import torch_robotics.robots as tr_robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch


class EnvWarehouseMarvinBimanual(EnvBase):
    """Static table plus mirrored left/right cabinet shelves for Marvin."""

    placement_regions = {
        "table": {
            "translation": {"x": [[0.25, 0.85]], "y": [[-0.22, 0.22]], "z": [[0.08, 0.18]]},
        },
        "left_cabinet": {
            "translation": {"x": [[0.22, 0.82]], "y": [[0.50, 0.72]], "z": [[0.10, 0.72]]},
        },
        "right_cabinet": {
            "translation": {"x": [[0.22, 0.82]], "y": [[-0.72, -0.50]], "z": [[0.10, 0.72]]},
        },
    }

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

        table = create_table_object_field(tensor_args=tensor_args)
        table_sizes = table.fields[0].sizes[0]
        table_transform = np.eye(4)
        table_transform[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        table_transform[:3, 3] = np.array([0.55, 0.0, -table_sizes[2].item() / 2.0])
        table_transform = perturbation @ table_transform
        table.set_position_orientation(table_transform[:3, 3], table_transform[:3, :3])

        # The shelves face the table.  Mirroring the right shelf keeps the
        # cabinet openings in the two arms' natural lateral workspaces.
        left_shelf = create_shelf_field(tensor_args=tensor_args)
        left_transform = np.eye(4)
        left_transform[:3, 3] = np.array([0.15, 0.50, -0.80])
        left_transform = perturbation @ left_transform
        left_shelf.set_position_orientation(left_transform[:3, 3], left_transform[:3, :3])

        right_shelf = create_shelf_field(tensor_args=tensor_args)
        right_transform = np.eye(4)
        right_transform[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        right_transform[:3, 3] = np.array([0.95, -0.50, -0.80])
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
            limits=torch.tensor([[-1.0, -1.6, -0.35], [1.8, 1.6, 1.8]], **tensor_args),
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
