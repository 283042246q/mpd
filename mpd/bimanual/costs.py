"""Differentiable dual-arm costs used by guidance and dense validation."""

from __future__ import annotations

import torch

from torchlie.functional import SE3 as SE3_Func


def inactive_arm_hold_cost(q: torch.Tensor, q_start: torch.Tensor, task_mode: str) -> torch.Tensor:
    """Quadratic hold cost; callers should also apply hard projection."""
    if task_mode not in {"left_only", "right_only"}:
        return torch.zeros(q.shape[:-2], dtype=q.dtype, device=q.device)
    inactive = q[..., 7:] if task_mode == "left_only" else q[..., :7]
    start = q_start[..., 7:] if task_mode == "left_only" else q_start[..., :7]
    return (inactive - start.unsqueeze(-2)).square().mean(dim=(-1, -2))


def project_inactive_arm(q: torch.Tensor, q_start: torch.Tensor, task_mode: str) -> torch.Tensor:
    if task_mode not in {"left_only", "right_only"}:
        return q
    projected = q.clone()
    if task_mode == "left_only":
        projected[..., 7:] = q_start[..., 7:].unsqueeze(-2)
    else:
        projected[..., :7] = q_start[..., :7].unsqueeze(-2)
    return projected


def _rotation_log_angle(relative_rotation: torch.Tensor) -> torch.Tensor:
    cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def closed_chain_residual(
    left_pose: torch.Tensor,
    right_pose: torch.Tensor,
    object_to_left: torch.Tensor,
    object_to_right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-waypoint translation and rotation closure residuals."""
    left_object = left_pose @ torch.linalg.inv(object_to_left)
    right_object = right_pose @ torch.linalg.inv(object_to_right)
    translation = torch.linalg.vector_norm(left_object[..., :3, 3] - right_object[..., :3, 3], dim=-1)
    relative = left_object[..., :3, :3].transpose(-1, -2) @ right_object[..., :3, :3]
    return translation, _rotation_log_angle(relative)


def closed_chain_cost(*args, translation_weight: float = 1.0, rotation_weight: float = 0.2, **kwargs) -> torch.Tensor:
    translation, rotation = closed_chain_residual(*args, **kwargs)
    return translation_weight * translation.square().mean(dim=-1) + rotation_weight * rotation.square().mean(dim=-1)


def interarm_clearance_cost(left_points: torch.Tensor, right_points: torch.Tensor, margin: float = 0.08) -> torch.Tensor:
    distances = torch.cdist(left_points, right_points)
    return torch.relu(margin - distances).square().mean(dim=(-1, -2))


def dual_ee_goal_cost_gradient(
    current_poses: torch.Tensor,
    spatial_jacobians: torch.Tensor,
    goal_poses: torch.Tensor,
    active_ee_mask: torch.Tensor,
    *,
    component: str = "pose",
    error_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a masked dual-EE SE(3) cost and its 14D joint gradient.

    ``current_poses`` and ``spatial_jacobians`` use fixed ``[left, right]``
    slots.  Averaging by the number of active end effectors keeps the guide
    magnitude stable when a future single-arm mode uses the same checkpoint.
    """
    if current_poses.shape[-3:] != (2, 3, 4):
        raise ValueError("current_poses must end in [2, 3, 4]")
    if spatial_jacobians.shape[-3:] != (2, 6, 14):
        raise ValueError("spatial_jacobians must end in [2, 6, 14]")
    if goal_poses.shape != (2, 3, 4):
        raise ValueError("goal_poses must have shape [2, 3, 4]")
    mask = torch.as_tensor(
        active_ee_mask,
        dtype=current_poses.dtype,
        device=current_poses.device,
    )
    if mask.shape != (2,) or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("active_ee_mask must have shape [2] and contain zero/one")
    if not bool(mask.any().item()):
        raise ValueError("at least one end effector must be active")
    scale = float(error_scale)
    if scale <= 0.0:
        raise ValueError("error_scale must be positive")

    component_slices = {
        "position": slice(0, 3),
        "orientation": slice(3, 6),
        "pose": slice(0, 6),
    }
    try:
        component_slice = component_slices[component]
    except KeyError as error:
        raise ValueError("component must be position, orientation, or pose") from error

    current_inverse = SE3_Func.inv(current_poses)
    error_se3 = SE3_Func.log(SE3_Func.compose(goal_poses, current_inverse))
    error_component = error_se3[..., component_slice]
    jacobian_component = spatial_jacobians[..., component_slice, :]
    denominator = mask.sum()
    scaled_square = error_component.square() / (scale * scale)
    per_arm_cost = 0.5 * scaled_square.sum(dim=-1)
    cost = (per_arm_cost * mask).sum(dim=-1) / denominator
    gradient_task = -error_component / (scale * scale)
    gradient_joint_per_arm = torch.einsum(
        "...adj,...ad->...aj", jacobian_component, gradient_task
    )
    gradient_joint = (
        gradient_joint_per_arm * mask[..., None]
    ).sum(dim=-2) / denominator
    return cost, gradient_joint, error_se3
