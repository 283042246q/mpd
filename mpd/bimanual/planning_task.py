"""Task container shared by independent and cooperative bimanual planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .costs import closed_chain_cost, project_inactive_arm


@dataclass
class BimanualPlanningTask:
    robot: Any
    task_mode: str = "dual_independent"
    q_start: torch.Tensor | None = None
    q_goal: torch.Tensor | None = None
    left_goal_pose: torch.Tensor | None = None
    right_goal_pose: torch.Tensor | None = None
    object_goal_pose: torch.Tensor | None = None
    object_to_left_grasp: torch.Tensor | None = None
    object_to_right_grasp: torch.Tensor | None = None
    payload: Any = None
    active_joint_mask: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    VALID_MODES = frozenset({"left_only", "right_only", "dual_independent", "cooperative_rigid"})

    def __post_init__(self):
        if self.task_mode not in self.VALID_MODES:
            raise ValueError(f"unknown bimanual task mode: {self.task_mode}")
        if self.q_start is not None and self.q_start.shape[-1] != 14:
            raise ValueError("q_start must be 14-dimensional")
        if self.q_goal is not None and self.q_goal.shape[-1] != 14:
            raise ValueError("q_goal must be 14-dimensional")
        if self.active_joint_mask is None:
            self.active_joint_mask = torch.ones(14, dtype=torch.bool, device=self.q_start.device if self.q_start is not None else None)
        if self.active_joint_mask.shape != (14,):
            raise ValueError("active_joint_mask must have shape [14]")
        if self.task_mode == "left_only":
            self.active_joint_mask[7:] = False
        elif self.task_mode == "right_only":
            self.active_joint_mask[:7] = False
        if self.task_mode == "cooperative_rigid":
            if self.object_to_left_grasp is None or self.object_to_right_grasp is None:
                raise ValueError("cooperative_rigid requires both grasp transforms")

    @property
    def joint_order(self):
        return tuple([f"Joint{i}_L" for i in range(1, 8)] + [f"Joint{i}_R" for i in range(1, 8)])

    def project(self, q: torch.Tensor) -> torch.Tensor:
        if self.q_start is None:
            return q
        return project_inactive_arm(q, self.q_start, self.task_mode)

    def closure_cost(self, q: torch.Tensor) -> torch.Tensor:
        if self.task_mode != "cooperative_rigid":
            return torch.zeros(q.shape[:-2], dtype=q.dtype, device=q.device)
        left = self.robot.fk_left(q)
        right = self.robot.fk_right(q)
        return closed_chain_cost(left, right, self.object_to_left_grasp, self.object_to_right_grasp)

