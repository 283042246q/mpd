"""Goal resolution helpers.  Pose-to-joint solving is injected by deployment."""

from __future__ import annotations

from typing import Callable

import torch


def resolve_goal(q_start, *, q_goal=None, task_mode="dual_independent", left_goal_pose=None, right_goal_pose=None, object_goal_pose=None, solver: Callable | None = None):
    if q_goal is not None:
        q_goal = torch.as_tensor(q_goal, dtype=torch.float32)
        if q_goal.shape != (14,):
            raise ValueError("q_goal must have shape [14]")
        return q_goal
    if solver is None:
        raise ValueError("a goal IK solver is required when q_goal is not supplied")
    result = solver(q_start, task_mode=task_mode, left_goal_pose=left_goal_pose, right_goal_pose=right_goal_pose, object_goal_pose=object_goal_pose)
    result = torch.as_tensor(result, dtype=torch.float32)
    if result.shape != (14,):
        raise ValueError("goal IK solver must return shape [14]")
    return result

