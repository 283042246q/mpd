"""Differentiable dual-arm costs used by guidance and dense validation."""

from __future__ import annotations

import torch


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
