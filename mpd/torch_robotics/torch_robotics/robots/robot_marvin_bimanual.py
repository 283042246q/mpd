"""Marvin's fixed-order, differentiable 14-DoF bimanual model.

The model is deliberately additive: Franka/Panda classes and their assets are
not modified.  The geometry files are a flattened MPD copy of the ROS model;
the ``model_hash`` field lets deployment code reject stale copies.
"""

from __future__ import annotations

import os
import hashlib
from typing import Iterable

import torch

from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.utils.files import get_configs_path, get_robot_path
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class RobotMarvinBimanual(RobotBase):
    JOINT_NAMES = tuple([f"Joint{index}_L" for index in range(1, 8)] + [f"Joint{index}_R" for index in range(1, 8)])
    LEFT_SLICE = slice(0, 7)
    RIGHT_SLICE = slice(7, 14)
    link_name_ee = "flange_L"  # compatibility for generic single-EE utilities
    link_name_ee_left = "flange_L"
    link_name_ee_right = "flange_R"

    def __init__(self, *, tensor_args=DEFAULT_TENSOR_ARGS, grasped_object=None, **kwargs):
        robot_dir = os.path.join(get_robot_path(), "marvin")
        config_dir = os.path.join(get_configs_path(), "marvin")
        super().__init__(
            urdf_robot_file=os.path.join(robot_dir, "marvin_bimanual_mpd.urdf"),
            collision_spheres_file_path=os.path.join(config_dir, "collision_spheres.yaml"),
            collision_parent_bounds_file_path=os.path.join(config_dir, "collision_parent_bounds.yaml"),
            joint_limits_file_path=os.path.join(config_dir, "joint_limits.yaml"),
            link_name_ee=self.link_name_ee,
            grasped_object=grasped_object,
            tensor_args=tensor_args,
            **kwargs,
        )
        with open(os.path.join(robot_dir, "marvin_bimanual_mpd.urdf"), "rb") as model_file:
            self.model_hash = hashlib.sha256(model_file.read()).hexdigest()
        self.joint_names = tuple(
            joint.name for joint in self.robot_urdf.joints if joint.joint_type != "fixed"
        )
        if self.q_dim != 14 or self.joint_names != self.JOINT_NAMES:
            raise ValueError("Marvin model must expose exactly the canonical 14-joint order")

        self._base_offsets = torch.as_tensor(
            [[0.0, 0.32, 0.55], [0.0, -0.32, 0.55]], **tensor_args
        )
        self._link_lengths = torch.as_tensor([0.12, 0.28, 0.25, 0.22, 0.18, 0.12], **tensor_args)

    @staticmethod
    def split_q(q14: torch.Tensor):
        q14 = torch.as_tensor(q14)
        if q14.shape[-1] != 14:
            raise ValueError(f"q14 must end in 14 values, got {tuple(q14.shape)}")
        return q14[..., :7], q14[..., 7:]

    @classmethod
    def merge_q(cls, q_left: torch.Tensor, q_right: torch.Tensor) -> torch.Tensor:
        if q_left.shape[-1] != 7 or q_right.shape[-1] != 7 or q_left.shape[:-1] != q_right.shape[:-1]:
            raise ValueError("left and right joint arrays must have matching leading shape and seven joints")
        return torch.cat((q_left, q_right), dim=-1)

    @staticmethod
    def _rot_z(angle: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(angle), torch.sin(angle)
        matrix = torch.zeros((*angle.shape, 4, 4), dtype=angle.dtype, device=angle.device)
        matrix[..., 0, 0] = c
        matrix[..., 0, 1] = -s
        matrix[..., 1, 0] = s
        matrix[..., 1, 1] = c
        matrix[..., 2, 2] = 1.0
        matrix[..., 3, 3] = 1.0
        return matrix

    @staticmethod
    def _trans_x(distance: torch.Tensor) -> torch.Tensor:
        matrix = torch.eye(4, dtype=distance.dtype, device=distance.device).expand(*distance.shape, 4, 4).clone()
        matrix[..., 0, 3] = distance
        return matrix

    def _fk_arm(self, q: torch.Tensor, side: int) -> torch.Tensor:
        if q.shape[-1] != 7:
            raise ValueError("an arm configuration must have seven joints")
        original_shape = q.shape[:-1]
        q_flat = q.reshape(-1, 7)
        transform = torch.eye(4, dtype=q.dtype, device=q.device).expand(q_flat.shape[0], 4, 4).clone()
        transform[..., :3, 3] = self._base_offsets[side].to(dtype=q.dtype, device=q.device)
        for index in range(7):
            transform = transform @ self._rot_z(q_flat[:, index])
            if index < 6:
                transform = transform @ self._trans_x(self._link_lengths[index].to(dtype=q.dtype, device=q.device))
        return transform.reshape(*original_shape, 4, 4)

    def _fk_arm_with_axes(self, q: torch.Tensor, side: int):
        transform = torch.eye(4, dtype=q.dtype, device=q.device)
        transform = transform.clone()
        transform[:3, 3] = self._base_offsets[side].to(dtype=q.dtype, device=q.device)
        axes, origins = [], []
        for index in range(7):
            axes.append(transform[:3, :3] @ torch.tensor([0.0, 0.0, 1.0], dtype=q.dtype, device=q.device))
            origins.append(transform[:3, 3].clone())
            transform = transform @ self._rot_z(q[index])
            if index < 6:
                transform = transform @ self._trans_x(self._link_lengths[index].to(dtype=q.dtype, device=q.device))
        return transform, torch.stack(axes), torch.stack(origins)

    def fk_left(self, q14: torch.Tensor) -> torch.Tensor:
        return self._fk_arm(self.split_q(q14)[0], 0)

    def fk_right(self, q14: torch.Tensor) -> torch.Tensor:
        return self._fk_arm(self.split_q(q14)[1], 1)

    def _jfk_arm(self, q14: torch.Tensor, side: int):
        q = q14[..., self.LEFT_SLICE if side == 0 else self.RIGHT_SLICE]
        q_flat = q.reshape(-1, 7)
        poses = []
        jacobians = []
        for row in q_flat:
            row = row.detach().requires_grad_(True)
            pose, axes, origins = self._fk_arm_with_axes(row, side)
            position = pose[:3, 3]
            linear = torch.linalg.cross(axes, position.unsqueeze(0) - origins, dim=-1).transpose(0, 1)
            rotational = axes.transpose(0, 1)
            jacobian = torch.cat((linear, rotational), dim=0)
            poses.append(pose)
            jacobians.append(jacobian)
        shape = (*q.shape[:-1],)
        return torch.stack(jacobians).reshape(*shape, 6, 7), torch.stack(poses).reshape(*shape, 4, 4)

    def jfk_left(self, q14: torch.Tensor):
        return self._jfk_arm(q14, 0)

    def jfk_right(self, q14: torch.Tensor):
        return self._jfk_arm(q14, 1)

    def jfk_bimanual(self, q14: torch.Tensor):
        return self.jfk_left(q14), self.jfk_right(q14)

    def render(self, ax, q=None, **kwargs):
        # Rendering is intentionally optional for headless runtime use.
        if q is None:
            return
        for pose in (self.fk_left(q), self.fk_right(q)):
            position = pose[..., :3, 3].detach().cpu().reshape(-1, 3)[0]
            ax.scatter([position[0]], [position[1]], [position[2]], **kwargs)

    def render_trajectories(self, ax, q_pos_trajs=None, **kwargs):
        if q_pos_trajs is None:
            return
        for trajectory in q_pos_trajs:
            self.render(ax, trajectory[-1], **kwargs)
