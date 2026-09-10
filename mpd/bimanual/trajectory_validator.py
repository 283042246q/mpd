"""Hard, dense bimanual trajectory validation with structured failures."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from mpd.inference.dense_trajectory_validator import DenseTrajectoryValidator
from torch_robotics.trajectory.metrics import compute_ee_pose_errors
from torch_robotics.torch_kinematics_tree.geometrics.utils import (
    link_pos_from_link_tensor,
)

from .costs import closed_chain_residual


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    failure_code: str | None = None
    min_clearance_m: float = math.inf
    max_joint_violation: float = 0.0
    max_velocity_violation: float = 0.0
    max_acceleration_violation: float = 0.0
    max_closure_translation_error_m: float = 0.0
    max_closure_rotation_error_rad: float = 0.0
    details: dict = field(default_factory=dict)


class BimanualTrajectoryValidator:
    def __init__(self, robot, *, closure_translation_tolerance_m=0.002, closure_rotation_tolerance_rad=0.01745, min_clearance_m=0.0):
        self.robot = robot
        self.closure_translation_tolerance_m = float(closure_translation_tolerance_m)
        self.closure_rotation_tolerance_rad = float(closure_rotation_tolerance_rad)
        self.min_clearance_m = float(min_clearance_m)

    def validate(self, q, *, dt=None, task_mode="dual_independent", object_to_left_grasp=None, object_to_right_grasp=None):
        q = torch.as_tensor(q, dtype=torch.float32)
        if q.ndim != 2 or q.shape[-1] != 14 or not torch.isfinite(q).all():
            return ValidationReport(False, "INVALID_CONFIGURATION")
        if hasattr(self.robot, "q_pos_min") and torch.any(q < self.robot.q_pos_min - 1e-6) or hasattr(self.robot, "q_pos_max") and torch.any(q > self.robot.q_pos_max + 1e-6):
            return ValidationReport(False, "JOINT_LIMIT")
        max_velocity = max_acceleration = 0.0
        if dt is not None:
            dt = float(dt)
            if dt <= 0:
                return ValidationReport(False, "INVALID_TIMESTEP")
            velocity = torch.diff(q, dim=0) / dt
            if getattr(self.robot, "dq_max", None) is not None:
                max_velocity = float(torch.relu(velocity.abs() - self.robot.dq_max).max())
            if velocity.shape[0] > 1 and getattr(self.robot, "ddq_max", None) is not None:
                acceleration = torch.diff(velocity, dim=0) / dt
                max_acceleration = float(torch.relu(acceleration.abs() - self.robot.ddq_max).max())
        if max_velocity > 1e-6:
            return ValidationReport(False, "VELOCITY_LIMIT", max_velocity_violation=max_velocity)
        if max_acceleration > 1e-6:
            return ValidationReport(False, "ACCELERATION_LIMIT", max_acceleration_violation=max_acceleration)
        if task_mode == "cooperative_rigid":
            if object_to_left_grasp is None or object_to_right_grasp is None:
                return ValidationReport(False, "MISSING_GRASP_PROFILE")
            translation, rotation = closed_chain_residual(self.robot.fk_left(q), self.robot.fk_right(q), object_to_left_grasp, object_to_right_grasp)
            max_translation = float(translation.max())
            max_rotation = float(rotation.max())
            if max_translation > self.closure_translation_tolerance_m:
                return ValidationReport(False, "CLOSED_CHAIN_TRANSLATION", max_closure_translation_error_m=max_translation, max_closure_rotation_error_rad=max_rotation)
            if max_rotation > self.closure_rotation_tolerance_rad:
                return ValidationReport(False, "CLOSED_CHAIN_ROTATION", max_closure_translation_error_m=max_translation, max_closure_rotation_error_rad=max_rotation)
            return ValidationReport(True, max_closure_translation_error_m=max_translation, max_closure_rotation_error_rad=max_rotation)
        return ValidationReport(True, max_velocity_violation=max_velocity, max_acceleration_violation=max_acceleration)


def _collision_sphere_side(name: str) -> str:
    if name.startswith("left_") or "_L" in name:
        return "left"
    if name.startswith("right_") or "_R" in name:
        return "right"
    return "base"


class BimanualDenseTrajectoryValidator(DenseTrajectoryValidator):
    """Independent dense oracle with pair-aware and dual-TCP diagnostics."""

    def __init__(self, planning_task, config=None):
        super().__init__(planning_task, config=config)
        config = dict(config or {})
        self.ee_position_tolerance_m = float(
            config.get("ee_position_tolerance_m", 0.03)
        )
        self.ee_orientation_tolerance_rad = float(
            config.get("ee_orientation_tolerance_rad", 0.0872664626)
        )
        category_names = ("left", "right", "interarm", "base")
        pair_categories = []
        names = self.robot.link_collision_spheres_names
        for pair in self.robot.link_self_collision_tuples:
            side_a = _collision_sphere_side(names[pair[0]])
            side_b = _collision_sphere_side(names[pair[1]])
            if {side_a, side_b} == {"left", "right"}:
                category = "interarm"
            elif "left" in (side_a, side_b):
                category = "left"
            elif "right" in (side_a, side_b):
                category = "right"
            else:
                category = "base"
            pair_categories.append(category_names.index(category))
        self._self_pair_category_names = category_names
        self._self_pair_category_ids = torch.as_tensor(
            pair_categories, dtype=torch.long
        )

    def _self_clearance_block(self, positions, pair_chunk_size):
        self_field = self.planning_task.get_collision_self_field()
        if self_field is None:
            return super()._self_clearance_block(positions, pair_chunk_size)
        overall, grouped = self_field.compute_minimum_signed_distances_by_group(
            positions,
            self._self_pair_category_ids,
            len(self._self_pair_category_names),
            pair_chunk_size=pair_chunk_size,
        )
        return overall, {
            name: grouped[..., index]
            for index, name in enumerate(self._self_pair_category_names)
        }

    def _pair_masks(self, device):
        categories = {key: [] for key in ("left", "right", "interarm", "base")}
        names = self.robot.link_collision_spheres_names
        for pair_index, pair in enumerate(self.robot.link_self_collision_tuples):
            side_a = _collision_sphere_side(names[pair[0]])
            side_b = _collision_sphere_side(names[pair[1]])
            if {side_a, side_b} == {"left", "right"}:
                category = "interarm"
            elif "left" in (side_a, side_b):
                category = "left"
            elif "right" in (side_a, side_b):
                category = "right"
            else:
                category = "base"
            categories[category].append(pair_index)
        return {
            key: torch.as_tensor(value, dtype=torch.long, device=device)
            for key, value in categories.items()
        }

    @staticmethod
    def _category_minimum(pair_clearance, indices):
        if indices.numel() == 0 or pair_clearance.shape[-1] == 0:
            return torch.full(
                pair_clearance.shape[0:1],
                torch.inf,
                dtype=pair_clearance.dtype,
                device=pair_clearance.device,
            )
        return pair_clearance.index_select(-1, indices).amin(dim=(-2, -1))

    def _annotate(self, result):
        q = result.q_position
        batch, horizon, _ = q.shape
        category_minimum = result.self_collision_category_minimum
        if category_minimum is not None:
            result.minimum_left_self_clearance = category_minimum["left"]
            result.minimum_right_self_clearance = category_minimum["right"]
            result.minimum_interarm_clearance = category_minimum["interarm"]
            result.minimum_base_self_clearance = category_minimum["base"]
        else:
            # Legacy/default path retained for bitwise-compatible entry points.
            poses = self.robot.fk_collision_spheres(q.reshape(batch * horizon, -1))
            poses = torch.stack(poses).transpose(0, 1).reshape(
                batch, horizon, -1, 3, 4
            )
            positions = link_pos_from_link_tensor(poses)[
                ..., : self.robot.task_space_dim
            ]
            self_field = self.planning_task.get_collision_self_field()
            if self_field is None:
                pair_clearance = torch.empty(
                    batch, horizon, 0, dtype=q.dtype, device=q.device
                )
            else:
                pair_clearance = self_field.compute_embodiment_signed_distances(
                    None, positions
                )
            masks = self._pair_masks(q.device)
            result.minimum_left_self_clearance = self._category_minimum(
                pair_clearance, masks["left"]
            )
            result.minimum_right_self_clearance = self._category_minimum(
                pair_clearance, masks["right"]
            )
            result.minimum_interarm_clearance = self._category_minimum(
                pair_clearance, masks["interarm"]
            )
            result.minimum_base_self_clearance = self._category_minimum(
                pair_clearance, masks["base"]
            )

        terminal = q[:, -1]
        achieved = torch.stack(
            (self.robot.fk_left(terminal), self.robot.fk_right(terminal)), dim=-3
        )
        error_position, error_orientation = compute_ee_pose_errors(
            self.planning_task.ee_pose_goal, achieved
        )
        result.ee_position_error_m = torch.linalg.norm(error_position, dim=-1)
        result.ee_orientation_error_rad = torch.linalg.norm(
            error_orientation, dim=-1
        )
        active = self.planning_task.active_ee_mask.to(
            dtype=torch.bool, device=q.device
        )
        goal_invalid = (
            (
                result.ee_position_error_m > self.ee_position_tolerance_m
            )
            | (
                result.ee_orientation_error_rad
                > self.ee_orientation_tolerance_rad
            )
        )[..., active].any(dim=-1)
        checked = result.trajectory_checked_mask
        if checked is None:
            checked = torch.ones_like(result.trajectory_valid_mask)
        goal_invalid &= checked
        was_valid = result.trajectory_valid_mask.clone()
        result.trajectory_valid_mask &= ~goal_invalid
        result.first_invalid_index = torch.where(
            was_valid & goal_invalid,
            torch.full_like(result.first_invalid_index, horizon - 1),
            result.first_invalid_index,
        )
        result.goal_violation_mask = goal_invalid
        failure_codes = []
        for index in range(batch):
            if not bool(checked[index].item()):
                failure_codes.append("NOT_CHECKED")
            elif bool(result.environment_collision_mask[index].any().item()):
                failure_codes.append("ENVIRONMENT_COLLISION")
            elif bool((result.minimum_interarm_clearance[index] < 0).item()):
                failure_codes.append("INTERARM_COLLISION")
            elif bool(result.self_collision_mask[index].any().item()):
                failure_codes.append("SELF_COLLISION")
            elif bool(result.joint_position_violation_mask[index].item()):
                failure_codes.append("JOINT_POSITION_LIMIT")
            elif bool(result.joint_velocity_violation_mask[index].item()):
                failure_codes.append("JOINT_VELOCITY_LIMIT")
            elif bool(result.joint_acceleration_violation_mask[index].item()):
                failure_codes.append("JOINT_ACCELERATION_LIMIT")
            elif bool(goal_invalid[index].item()):
                failure_codes.append("DUAL_EE_GOAL")
            else:
                failure_codes.append(None)
        result.failure_codes = failure_codes
        return result

    def validate(self, *args, **kwargs):
        return self._annotate(super().validate(*args, **kwargs))

    def validate_ranked_batches(self, *args, **kwargs):
        # The base merger intentionally copies only common fields. Recompute
        # bimanual diagnostics from the merged dense samples so artifact output
        # remains complete when ranked early exit is enabled later.
        return self._annotate(super().validate_ranked_batches(*args, **kwargs))
