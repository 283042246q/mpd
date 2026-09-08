"""Marvin's fixed-order, differentiable 14-DoF bimanual model."""

from __future__ import annotations

import os
import hashlib
import yaml

import torch
import torchkin

from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.utils.files import get_configs_path, get_robot_path
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class RobotMarvinBimanual(RobotBase):
    JOINT_NAMES = tuple([f"Joint{index}_L" for index in range(1, 8)] + [f"Joint{index}_R" for index in range(1, 8)])
    LEFT_SLICE = slice(0, 7)
    RIGHT_SLICE = slice(7, 14)
    link_name_ee = "left_pika_gripper_tcp"
    link_name_ee_left = "left_pika_gripper_tcp"
    link_name_ee_right = "right_pika_gripper_tcp"
    link_name_flange_left = "flange_L"
    link_name_flange_right = "flange_R"

    def __init__(self, *, with_pika=True, tensor_args=DEFAULT_TENSOR_ARGS, grasped_object=None, **kwargs):
        robot_dir = os.path.join(get_robot_path(), "marvin")
        config_dir = os.path.join(get_configs_path(), "marvin")
        self.with_pika = with_pika
        if with_pika:
            config_dir = os.path.join(config_dir, "pika")
        self.link_name_ee_left = "left_pika_gripper_tcp" if with_pika else "flange_L"
        self.link_name_ee_right = "right_pika_gripper_tcp" if with_pika else "flange_R"
        self.link_name_ee = self.link_name_ee_left
        model_name = "marvin_pika_bimanual_mpd.urdf" if with_pika else "marvin_bimanual_mpd.urdf"
        self.self_collision_pairs_file_path = os.path.join(config_dir, "self_collision_pairs.yaml")
        super().__init__(
            urdf_robot_file=os.path.join(robot_dir, model_name),
            collision_spheres_file_path=os.path.join(config_dir, "collision_spheres.yaml"),
            collision_parent_bounds_file_path=os.path.join(config_dir, "collision_parent_bounds.yaml"),
            joint_limits_file_path=os.path.join(config_dir, "joint_limits.yaml"),
            link_name_ee=self.link_name_ee,
            grasped_object=grasped_object,
            tensor_args=tensor_args,
            **kwargs,
        )
        with open(os.path.join(robot_dir, model_name), "rb") as model_file:
            self.model_hash = hashlib.sha256(model_file.read()).hexdigest()
        if with_pika:
            with open(os.path.join(robot_dir, "pika_assets.lock.yaml")) as file:
                self.asset_manifest = yaml.safe_load(file)
            self.asset_hash = self.asset_manifest["asset_sha256"]
            self.tcp_calibrated = self.asset_manifest["tcp_calibrated"]
        self.joint_names = tuple(
            joint.name for joint in self.robot_urdf.joints if joint.joint_type != "fixed"
        )
        if self.q_dim != 14 or self.joint_names != self.JOINT_NAMES:
            raise ValueError("Marvin model must expose exactly the canonical 14-joint order")

        self._fk_left, _, _ = torchkin.get_forward_kinematics_fns(
            robot=self.robot_torchkin, link_names=[self.link_name_ee_left]
        )
        self._fk_right, _, _ = torchkin.get_forward_kinematics_fns(
            robot=self.robot_torchkin, link_names=[self.link_name_ee_right]
        )
        self._torchkin_to_canonical = torch.tensor(
            [self.JOINT_NAMES.index(joint.name) for joint in self.robot_torchkin.joints[: self.q_dim]],
            dtype=torch.long,
            device=self.q_pos_min.device,
        )
        self._canonical_to_torchkin = torch.argsort(self._torchkin_to_canonical)
        self.fk_ee = self._canonical_fk_list(self.fk_ee)
        self.jfk_s_ee = self._canonical_jacobian_fk(self.jfk_s_ee)
        self.jfk_b_ee = self._canonical_jacobian_fk(self.jfk_b_ee)

        # RobotBase installs raw TorchKin functions for collision spheres.  A
        # branched tree is traversed in an interleaved order, while MPD's
        # public contract is always [left seven, right seven].  Wrap every
        # inherited collision FK entry point so collision fields use the same
        # canonical ordering as fk_left/fk_right.
        self._fk_collision_spheres_torchkin = self.fk_collision_spheres
        self.fk_collision_spheres = self._canonical_fk_list(self._fk_collision_spheres_torchkin)
        self._fk_collision_parent_links_torchkin = self.fk_collision_sphere_parent_links
        self.fk_collision_sphere_parent_links = self._canonical_fk_list(self._fk_collision_parent_links_torchkin)
        self._jfk_collision_spheres_torchkin = self.jfk_s_collision_spheres
        self.jfk_s_collision_spheres = self._canonical_jacobian_fk(self._jfk_collision_spheres_torchkin)
        self._jfk_collision_parent_links_torchkin = self.jfk_s_collision_sphere_parent_links
        self.jfk_s_collision_sphere_parent_links = self._canonical_jacobian_fk(
            self._jfk_collision_parent_links_torchkin
        )
        self._fk_collision_parent_pose_cache_torchkin = self.fk_collision_parent_pose_cache
        self.fk_collision_parent_pose_cache = self._canonical_fk_list(
            self._fk_collision_parent_pose_cache_torchkin
        )
        self.jfk_b_collision_spheres = self._canonical_jacobian_fk(self.jfk_b_collision_spheres)
        self.jfk_b_collision_sphere_parent_links = self._canonical_jacobian_fk(
            self.jfk_b_collision_sphere_parent_links
        )

    def _canonical_q(self, q):
        q = torch.as_tensor(q)
        if q.shape[-1] != self.q_dim:
            raise ValueError(f"q must end in {self.q_dim} values, got {tuple(q.shape)}")
        return q.reshape(-1, self.q_dim)[..., self._torchkin_to_canonical]

    def q_to_torchkin(self, q):
        """Adapter for consumers that create their own subset FK functions."""
        return q[..., self._torchkin_to_canonical]

    def jacobian_from_torchkin(self, jacobian):
        """Map the last (joint) axis of a raw TorchKin Jacobian to MPD order."""
        return jacobian[..., self._canonical_to_torchkin]

    def _canonical_fk_list(self, fk_fn):
        def wrapped(q):
            q = torch.as_tensor(q)
            leading_shape = q.shape[:-1]
            poses = fk_fn(self._canonical_q(q))
            return [pose.reshape(*leading_shape, *pose.shape[-2:]) for pose in poses]

        return wrapped

    def _canonical_jacobian_fk(self, fk_fn):
        def wrapped(q):
            q = torch.as_tensor(q)
            leading_shape = q.shape[:-1]
            jacobians, poses = fk_fn(self._canonical_q(q))
            jacobians = [jacobian[..., self._canonical_to_torchkin].reshape(
                *leading_shape, *jacobian.shape[-2:]) for jacobian in jacobians]
            poses = [pose.reshape(*leading_shape, *pose.shape[-2:]) for pose in poses]
            return jacobians, poses

        return wrapped

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

    def _flatten_q(self, q14: torch.Tensor):
        q14 = torch.as_tensor(q14)
        if q14.shape[-1] != 14:
            raise ValueError(f"q14 must end in 14 values, got {tuple(q14.shape)}")
        return q14.shape[:-1], q14.reshape(-1, 14)[..., self._torchkin_to_canonical]

    def _pose(self, q14: torch.Tensor, fk_fn):
        leading_shape, q_flat = self._flatten_q(q14)
        pose = fk_fn(q_flat)[0]
        return pose.reshape(*leading_shape, 3, 4)

    def _spatial_jacobian(self, q_flat: torch.Tensor, end_pose: torch.Tensor, joint_slice):
        all_poses = self.fk_all(q_flat)
        end_position = end_pose[..., :3, 3]
        columns = []
        for joint in self.robot_torchkin.joints[: self.q_dim]:
            parent_pose = all_poses[joint.parent_link.id]
            parent_rotation = parent_pose[..., :3, :3]
            parent_position = parent_pose[..., :3, 3]
            origin = joint.origin[0].to(dtype=q_flat.dtype, device=q_flat.device)
            joint_rotation = parent_rotation @ origin[:3, :3]
            joint_position = parent_rotation @ origin[:3, 3] + parent_position
            axis = joint_rotation @ joint.axis[3:, 0].to(dtype=q_flat.dtype, device=q_flat.device)
            linear = torch.linalg.cross(axis, end_position - joint_position, dim=-1)
            columns.append(torch.cat((linear, axis), dim=-1))
        jacobian_torchkin = torch.stack(columns, dim=-1)
        jacobian_canonical = jacobian_torchkin[..., self._canonical_to_torchkin]
        return jacobian_canonical[..., joint_slice]

    def _pose_and_jacobian(self, q14: torch.Tensor, fk_fn, joint_slice):
        leading_shape, q_flat = self._flatten_q(q14)
        pose = fk_fn(q_flat)[0]
        jacobian = self._spatial_jacobian(q_flat, pose, joint_slice)
        jacobian = jacobian.reshape(*leading_shape, 6, 7)
        pose = pose.reshape(*leading_shape, 3, 4)
        return jacobian, pose

    def fk_left(self, q14: torch.Tensor) -> torch.Tensor:
        return self._pose(q14, self._fk_left)

    def fk_right(self, q14: torch.Tensor) -> torch.Tensor:
        return self._pose(q14, self._fk_right)

    def jfk_left(self, q14: torch.Tensor):
        return self._pose_and_jacobian(q14, self._fk_left, self.LEFT_SLICE)

    def jfk_right(self, q14: torch.Tensor):
        return self._pose_and_jacobian(q14, self._fk_right, self.RIGHT_SLICE)

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
