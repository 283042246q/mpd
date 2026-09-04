"""Hard, dense bimanual trajectory validation with structured failures."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

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

