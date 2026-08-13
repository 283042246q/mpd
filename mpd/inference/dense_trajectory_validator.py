"""Independent dense trajectory safety oracle.

This module intentionally uses only FK and distance queries. It never consumes
the guidance active set and never computes collision Jacobians.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as torch_functional

from mpd.parametric_trajectory.trajectory_bspline import BSpline, ParametricTrajectoryBspline
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor


@dataclass
class DenseValidationResult:
    trajectory_valid_mask: torch.Tensor
    environment_collision_mask: torch.Tensor
    self_collision_mask: torch.Tensor
    joint_limit_mask: torch.Tensor
    joint_position_violation_mask: torch.Tensor
    joint_velocity_violation_mask: torch.Tensor
    joint_acceleration_violation_mask: torch.Tensor
    minimum_environment_clearance: torch.Tensor
    minimum_self_clearance: torch.Tensor
    first_invalid_index: torch.Tensor
    q_position: torch.Tensor
    q_velocity: torch.Tensor
    q_acceleration: torch.Tensor
    trajectory_checked_mask: torch.Tensor = None
    validated_indices: torch.Tensor = None
    batches_evaluated: int = 1
    complete: bool = True
    bucket_capacities_evaluated: tuple = ()
    padding_slots_evaluated: int = 0
    cuda_graph_replays: int = 0


@dataclass
class _DenseBucketWorkspace:
    """Persistent fixed-address input buffers for one validator batch shape."""

    control_points: torch.Tensor
    slot_valid_mask: torch.Tensor


def _resample(values, num_points):
    if values.shape[-2] == num_points:
        return values
    flat = values.reshape(-1, values.shape[-2], values.shape[-1]).transpose(1, 2)
    dense = torch_functional.interpolate(flat, size=num_points, mode="linear", align_corners=True)
    return dense.transpose(1, 2).reshape(*values.shape[:-2], num_points, values.shape[-1])


class DenseTrajectoryValidator:
    def __init__(self, planning_task, config=None):
        self.planning_task = planning_task
        self.robot = planning_task.robot
        self.parametric_trajectory = planning_task.parametric_trajectory
        self.config = dict(config or {})
        self._bspline_cache = {}
        self._bucket_workspaces = {}

    @staticmethod
    def _bucket_workspace_key(control_points, bucket_size):
        return (
            int(bucket_size),
            tuple(control_points.shape[1:]),
            control_points.dtype,
            control_points.device,
        )

    def _bucket_workspace(self, control_points, bucket_size):
        if not hasattr(self, "_bucket_workspaces"):
            self._bucket_workspaces = {}
        key = self._bucket_workspace_key(control_points, bucket_size)
        workspace = self._bucket_workspaces.get(key)
        if workspace is None:
            workspace = _DenseBucketWorkspace(
                control_points=torch.empty(
                    (int(bucket_size), *control_points.shape[1:]),
                    dtype=control_points.dtype,
                    device=control_points.device,
                ),
                slot_valid_mask=torch.zeros(
                    int(bucket_size),
                    dtype=torch.bool,
                    device=control_points.device,
                ),
            )
            self._bucket_workspaces[key] = workspace
        return workspace

    def prepare_ranked_batch_workspaces(self, control_points, batch_buckets):
        """Allocate every configured fixed input shape before ranked validation."""

        return tuple(
            self._bucket_workspace(control_points, bucket_size)
            for bucket_size in batch_buckets
        )

    def warmup_ranked_batch_workspaces(
        self,
        control_points,
        batch_buckets,
        **validate_kwargs,
    ):
        """Execute every fixed ranked-validation shape once.

        Workspace allocation alone does not initialize the FK and collision-field
        kernels used by :meth:`validate`.  Running each configured bucket here
        keeps that one-time cost out of the first user request.
        """

        if control_points is None or control_points.ndim < 2 or control_points.shape[0] < 1:
            raise ValueError("Dense warmup requires at least one batched control-point trajectory.")

        batch_buckets = tuple(int(value) for value in batch_buckets)
        if (
            not batch_buckets
            or any(value < 1 for value in batch_buckets)
            or tuple(sorted(set(batch_buckets))) != batch_buckets
        ):
            raise ValueError(
                "batch_buckets must contain unique positive integers in ascending order."
            )

        results = []
        for bucket_size in batch_buckets:
            workspace = self._bucket_workspace(control_points, bucket_size)
            workspace.control_points.copy_(
                control_points[:1].expand(bucket_size, *control_points.shape[1:])
            )
            workspace.slot_valid_mask.fill_(True)
            with torch.no_grad():
                results.append(
                    self.validate(
                        control_points=workspace.control_points,
                        **validate_kwargs,
                    )
                )

        if control_points.is_cuda:
            torch.cuda.synchronize(control_points.device)
        return tuple(results)

    def _dense_bspline(self, control_points, num_points):
        trajectory = self.parametric_trajectory
        augmented = trajectory.augment_control_points_fn(control_points, None, None)
        augmented = trajectory.preprocess_control_points(augmented)
        num_control_points = int(augmented.shape[-2])
        dense_points = int(num_points)
        # ``d`` is also used by some downstream code for derivative data.  The
        # knot-vector invariant preserves the constructor degree even if that
        # public attribute is overwritten after the planning trajectory is built.
        degree = int(trajectory.bspline.m - trajectory.bspline.n_pts)
        key = (num_control_points, degree, dense_points, augmented.dtype, augmented.device)
        basis = self._bspline_cache.get(key)
        if basis is None:
            basis = BSpline(
                num_pts=num_control_points,
                degree=degree,
                num_T_pts=dense_points,
                dtype=augmented.dtype,
                device=augmented.device,
            )
            self._bspline_cache[key] = basis

        q_position = torch.einsum("ijk,...km->...jm", basis.N, augmented)
        q_velocity_phase = torch.einsum("ijk,...km->...jm", basis.dN, augmented)
        q_acceleration_phase = torch.einsum("ijk,...km->...jm", basis.ddN, augmented)
        dense_phase = torch.linspace(
            0.0,
            1.0,
            dense_points,
            dtype=augmented.dtype,
            device=augmented.device,
        )
        original_phase = trajectory.phase_time.s.detach()
        original_raw_rate = trajectory.phase_time.rs_fn(original_phase)
        rate_scale = trajectory.phase_time.rs.detach()[0] / original_raw_rate[0]
        rs = trajectory.phase_time.rs_fn(dense_phase) * rate_scale
        dr_ds = trajectory.phase_time.drs_ds_fn(dense_phase) * rate_scale
        q_velocity = q_velocity_phase * rs[..., None]
        q_acceleration = q_acceleration_phase * rs[..., None] ** 2 + q_velocity_phase * dr_ds[..., None] * rs[..., None]
        return q_position, q_velocity, q_acceleration

    def _dense_trajectory(self, control_points, num_points, q_position=None, q_velocity=None, q_acceleration=None):
        if control_points is not None and isinstance(self.parametric_trajectory, ParametricTrajectoryBspline):
            return self._dense_bspline(control_points, num_points)
        if q_position is None:
            trajectory = self.parametric_trajectory.get_q_trajectory(
                control_points,
                None,
                None,
                get_type=("pos", "vel", "acc"),
                get_time_representation=True,
            )
            q_position, q_velocity, q_acceleration = trajectory["pos"], trajectory["vel"], trajectory["acc"]
        return _resample(q_position, num_points), _resample(q_velocity, num_points), _resample(q_acceleration, num_points)

    def _environment_clearance(self, positions):
        clearances = []
        for field in (
            self.planning_task.get_collision_objects_field(),
            self.planning_task.get_collision_ws_boundaries_field(),
        ):
            if field is None:
                continue
            signed = field.object_signed_distances(positions)
            radii = torch.as_tensor(field.collision_margins, dtype=positions.dtype, device=positions.device)
            while radii.ndim < signed.ndim:
                radii = radii.unsqueeze(0)
            clearance = signed - radii
            while clearance.ndim > 3:
                clearance = clearance.amin(dim=-2)
            clearances.append(clearance.amin(dim=-1))
        if not clearances:
            return torch.full(positions.shape[:2], torch.inf, dtype=positions.dtype, device=positions.device)
        return torch.stack(clearances).amin(dim=0)

    def validate(
        self,
        control_points=None,
        num_points=128,
        q_position=None,
        q_velocity=None,
        q_acceleration=None,
        check_environment=True,
        check_self_collision=True,
        check_joint_position=True,
        check_joint_velocity=True,
        check_joint_acceleration=True,
        check_joint_limits=None,
    ):
        if check_joint_limits is not None:
            check_joint_position = bool(check_joint_limits)
            check_joint_velocity = bool(check_joint_limits)
            check_joint_acceleration = bool(check_joint_limits)
        q_position, q_velocity, q_acceleration = self._dense_trajectory(
            control_points,
            int(num_points),
            q_position=q_position,
            q_velocity=q_velocity,
            q_acceleration=q_acceleration,
        )
        if q_position.ndim != 3:
            raise ValueError("DenseTrajectoryValidator expects [batch, time, dof] trajectories.")

        batch, horizon, _ = q_position.shape
        poses = self.robot.fk_collision_spheres(q_position.reshape(batch * horizon, -1))
        poses = torch.stack(poses).transpose(0, 1).reshape(batch, horizon, -1, 3, 4)
        positions = link_pos_from_link_tensor(poses)[..., : self.robot.task_space_dim]

        environment_clearance = self._environment_clearance(positions)
        environment_collision_mask = environment_clearance <= 0 if check_environment else torch.zeros_like(
            environment_clearance, dtype=torch.bool
        )

        self_field = self.planning_task.get_collision_self_field()
        if self_field is not None:
            self_clearance = self_field.compute_embodiment_signed_distances(None, positions).amin(dim=-1)
        else:
            self_clearance = torch.full_like(environment_clearance, torch.inf)
        self_collision_mask = self_clearance < 0 if check_self_collision else torch.zeros_like(
            self_clearance, dtype=torch.bool
        )

        joint_position_mask = torch.zeros((batch, horizon), dtype=torch.bool, device=q_position.device)
        if check_joint_position:
            joint_position_mask = ((q_position < self.robot.q_pos_min) | (q_position > self.robot.q_pos_max)).any(-1)
        joint_velocity_mask = torch.zeros_like(joint_position_mask)
        if check_joint_velocity and self.robot.dq_max is not None:
            joint_velocity_mask = (torch.abs(q_velocity) > self.robot.dq_max).any(-1)
        joint_acceleration_mask = torch.zeros_like(joint_position_mask)
        if check_joint_acceleration and self.robot.ddq_max is not None:
            joint_acceleration_mask = (torch.abs(q_acceleration) > self.robot.ddq_max).any(-1)
        joint_limit_mask = joint_position_mask | joint_velocity_mask | joint_acceleration_mask

        invalid_waypoint = environment_collision_mask | self_collision_mask | joint_limit_mask
        valid = ~invalid_waypoint.any(dim=-1)
        invalid_trajectories = ~valid
        first_invalid_candidate = invalid_waypoint.long().argmax(dim=-1)
        first_invalid = torch.where(
            invalid_trajectories,
            first_invalid_candidate,
            torch.full_like(first_invalid_candidate, -1),
        )

        return DenseValidationResult(
            trajectory_valid_mask=valid,
            environment_collision_mask=environment_collision_mask,
            self_collision_mask=self_collision_mask,
            joint_limit_mask=joint_limit_mask,
            joint_position_violation_mask=joint_position_mask.any(dim=-1),
            joint_velocity_violation_mask=joint_velocity_mask.any(dim=-1),
            joint_acceleration_violation_mask=joint_acceleration_mask.any(dim=-1),
            minimum_environment_clearance=environment_clearance.amin(dim=-1),
            minimum_self_clearance=self_clearance.amin(dim=-1),
            first_invalid_index=first_invalid,
            q_position=q_position,
            q_velocity=q_velocity,
            q_acceleration=q_acceleration,
            trajectory_checked_mask=torch.ones(
                batch, dtype=torch.bool, device=q_position.device
            ),
            validated_indices=torch.arange(
                batch, dtype=torch.long, device=q_position.device
            ),
            batches_evaluated=1,
            complete=True,
            bucket_capacities_evaluated=(batch,),
        )

    def validate_ranked_batches(
        self,
        control_points,
        ranked_indices,
        batch_buckets=(8, 16, 32, 64),
        preallocate_buffers=True,
        cuda_graph=False,
        stop_on_first_valid=True,
        **validate_kwargs,
    ):
        """Validate candidates in score order and optionally stop at first success.

        Returned tensors retain the original candidate dimension. Unchecked
        candidates have false boolean diagnostics, NaN dense trajectories and
        clearances, and are explicitly identified by ``trajectory_checked_mask``.
        """

        if control_points is None or control_points.ndim < 2:
            raise ValueError("Ranked dense validation requires batched control points.")
        total = int(control_points.shape[0])
        ranked_indices = torch.as_tensor(
            ranked_indices, dtype=torch.long, device=control_points.device
        ).flatten()
        if ranked_indices.numel() != total or not torch.equal(
            torch.sort(ranked_indices).values,
            torch.arange(total, dtype=torch.long, device=control_points.device),
        ):
            raise ValueError("ranked_indices must be a permutation of all candidates.")
        batch_buckets = tuple(int(value) for value in batch_buckets)
        if (
            not batch_buckets
            or any(value < 1 for value in batch_buckets)
            or tuple(sorted(set(batch_buckets))) != batch_buckets
        ):
            raise ValueError(
                "batch_buckets must contain unique positive integers in ascending order."
            )
        if cuda_graph:
            raise NotImplementedError(
                "CUDA Graph capture for dense validator buckets is not enabled yet; "
                "fixed preallocated eager buckets remain available."
            )
        if preallocate_buffers:
            self.prepare_ranked_batch_workspaces(control_points, batch_buckets)

        evaluated = []
        start = 0
        bucket_index = 0
        while start < total:
            bucket_size = batch_buckets[min(bucket_index, len(batch_buckets) - 1)]
            actual_size = min(bucket_size, total - start)
            indices = ranked_indices[start : start + actual_size]
            workspace = self._bucket_workspace(control_points, bucket_size)
            workspace.control_points[:actual_size].copy_(
                control_points.index_select(0, indices)
            )
            if actual_size < bucket_size:
                workspace.control_points[actual_size:].copy_(
                    workspace.control_points[actual_size - 1].unsqueeze(0).expand(
                        bucket_size - actual_size, *control_points.shape[1:]
                    )
                )
            workspace.slot_valid_mask.zero_()
            workspace.slot_valid_mask[:actual_size] = True
            result = self.validate(
                control_points=workspace.control_points,
                **validate_kwargs,
            )
            result.trajectory_valid_mask &= workspace.slot_valid_mask
            evaluated.append((indices, result, actual_size, bucket_size))
            if stop_on_first_valid and bool(
                result.trajectory_valid_mask[:actual_size].any().item()
            ):
                break
            start += actual_size
            bucket_index += 1

        sample = evaluated[0][1]
        horizon = int(sample.q_position.shape[1])
        dof = int(sample.q_position.shape[2])
        bool_waypoint = lambda: torch.zeros(
            (total, horizon), dtype=torch.bool, device=control_points.device
        )
        bool_trajectory = lambda: torch.zeros(
            total, dtype=torch.bool, device=control_points.device
        )
        float_trajectory = lambda: torch.full(
            (total,),
            torch.nan,
            dtype=sample.minimum_environment_clearance.dtype,
            device=control_points.device,
        )
        dense_trajectory = lambda: torch.full(
            (total, horizon, dof),
            torch.nan,
            dtype=sample.q_position.dtype,
            device=control_points.device,
        )

        merged = DenseValidationResult(
            trajectory_valid_mask=bool_trajectory(),
            environment_collision_mask=bool_waypoint(),
            self_collision_mask=bool_waypoint(),
            joint_limit_mask=bool_waypoint(),
            joint_position_violation_mask=bool_trajectory(),
            joint_velocity_violation_mask=bool_trajectory(),
            joint_acceleration_violation_mask=bool_trajectory(),
            minimum_environment_clearance=float_trajectory(),
            minimum_self_clearance=float_trajectory(),
            first_invalid_index=torch.full(
                (total,), -1, dtype=torch.long, device=control_points.device
            ),
            q_position=dense_trajectory(),
            q_velocity=dense_trajectory(),
            q_acceleration=dense_trajectory(),
            trajectory_checked_mask=bool_trajectory(),
            validated_indices=torch.cat(
                [indices for indices, _, _, _ in evaluated]
            ),
            batches_evaluated=len(evaluated),
            complete=sum(actual_size for _, _, actual_size, _ in evaluated) == total,
            bucket_capacities_evaluated=tuple(
                bucket_size for _, _, _, bucket_size in evaluated
            ),
            padding_slots_evaluated=sum(
                bucket_size - actual_size
                for _, _, actual_size, bucket_size in evaluated
            ),
        )
        for indices, result, actual_size, _ in evaluated:
            merged.trajectory_valid_mask[indices] = result.trajectory_valid_mask[:actual_size]
            merged.environment_collision_mask[indices] = result.environment_collision_mask[:actual_size]
            merged.self_collision_mask[indices] = result.self_collision_mask[:actual_size]
            merged.joint_limit_mask[indices] = result.joint_limit_mask[:actual_size]
            merged.joint_position_violation_mask[indices] = result.joint_position_violation_mask[:actual_size]
            merged.joint_velocity_violation_mask[indices] = result.joint_velocity_violation_mask[:actual_size]
            merged.joint_acceleration_violation_mask[indices] = result.joint_acceleration_violation_mask[:actual_size]
            merged.minimum_environment_clearance[indices] = result.minimum_environment_clearance[:actual_size]
            merged.minimum_self_clearance[indices] = result.minimum_self_clearance[:actual_size]
            merged.first_invalid_index[indices] = result.first_invalid_index[:actual_size]
            merged.q_position[indices] = result.q_position[:actual_size]
            merged.q_velocity[indices] = result.q_velocity[:actual_size]
            merged.q_acceleration[indices] = result.q_acceleration[:actual_size]
            merged.trajectory_checked_mask[indices] = True
        return merged
