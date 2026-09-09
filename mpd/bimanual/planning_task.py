"""Production planning task for Marvin's fixed-order bimanual model."""

from __future__ import annotations

from typing import Any

import torch

from torch_robotics.tasks.tasks import PlanningTask

from .costs import closed_chain_cost, project_inactive_arm


class BimanualPlanningTask(PlanningTask):
    VALID_MODES = frozenset(
        {"left_only", "right_only", "dual_independent", "cooperative_rigid"}
    )

    def __init__(
        self,
        *args,
        task_mode: str = "dual_independent",
        object_to_left_grasp: torch.Tensor | None = None,
        object_to_right_grasp: torch.Tensor | None = None,
        payload: Any = None,
        **kwargs,
    ):
        if task_mode not in self.VALID_MODES:
            raise ValueError(f"unknown bimanual task mode: {task_mode}")
        self.task_mode = task_mode
        self.object_to_left_grasp = object_to_left_grasp
        self.object_to_right_grasp = object_to_right_grasp
        self.payload = payload
        self.object_goal_pose = None
        super().__init__(*args, **kwargs)
        if self.robot.q_dim != 14:
            raise ValueError("BimanualPlanningTask requires a 14-DoF robot")
        self.active_joint_mask = torch.ones(
            14, dtype=torch.bool, device=self.q_pos_start.device
        )
        if task_mode == "left_only":
            self.active_joint_mask[7:] = False
        elif task_mode == "right_only":
            self.active_joint_mask[:7] = False
        self.active_ee_mask = torch.tensor(
            [task_mode != "right_only", task_mode != "left_only"],
            dtype=self.q_pos_start.dtype,
            device=self.q_pos_start.device,
        )
        identity = torch.eye(4, **self.tensor_args)[:3, :]
        self.ee_pose_goal = torch.stack((identity, identity))
        self.left_goal_pose = self.ee_pose_goal[0]
        self.right_goal_pose = self.ee_pose_goal[1]

    @property
    def joint_order(self):
        return tuple(self.robot.JOINT_NAMES)

    def set_q_pos_start_goal(self, q_pos_start, q_pos_goal, **kwargs):
        if q_pos_start.shape[-1] != 14 or q_pos_goal.shape[-1] != 14:
            raise ValueError("Marvin start and goal states must be 14-dimensional")
        super().set_q_pos_start_goal(q_pos_start, q_pos_goal, **kwargs)

    def set_ee_pose_goal(self, ee_pose_goal, active_ee_mask=None, **kwargs):
        ee_pose_goal = torch.as_tensor(
            ee_pose_goal, dtype=self.q_pos_start.dtype, device=self.q_pos_start.device
        )
        if ee_pose_goal.shape != (2, 3, 4):
            raise ValueError("bimanual ee_pose_goal must have shape [2, 3, 4]")
        if active_ee_mask is None:
            active_ee_mask = torch.ones(
                2, dtype=ee_pose_goal.dtype, device=ee_pose_goal.device
            )
        active_ee_mask = torch.as_tensor(
            active_ee_mask, dtype=ee_pose_goal.dtype, device=ee_pose_goal.device
        )
        if active_ee_mask.shape != (2,) or not torch.all(
            (active_ee_mask == 0) | (active_ee_mask == 1)
        ):
            raise ValueError("active_ee_mask must have shape [2] and contain only zero/one")
        self.ee_pose_goal = ee_pose_goal
        self.left_goal_pose = ee_pose_goal[0]
        self.right_goal_pose = ee_pose_goal[1]
        self.active_ee_mask = active_ee_mask

    def set_object_goal(self, object_goal_pose):
        self.object_goal_pose = object_goal_pose

    def jfk_s_ee(self, q):
        return self.robot.jfk_s_ee_bimanual(q)

    def project(self, q: torch.Tensor) -> torch.Tensor:
        return project_inactive_arm(q, self.q_pos_start, self.task_mode)

    def closure_cost(self, q: torch.Tensor) -> torch.Tensor:
        if self.task_mode != "cooperative_rigid":
            return torch.zeros(q.shape[:-2], dtype=q.dtype, device=q.device)
        if self.object_to_left_grasp is None or self.object_to_right_grasp is None:
            raise ValueError("cooperative_rigid requires both grasp transforms")
        return closed_chain_cost(
            self.robot.fk_left(q),
            self.robot.fk_right(q),
            self.object_to_left_grasp,
            self.object_to_right_grasp,
        )
