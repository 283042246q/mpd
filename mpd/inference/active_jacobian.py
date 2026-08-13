"""Bucketed active-time collision Jacobian evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchkin


@dataclass
class ActiveJacobianBucket:
    candidate_indices: torch.Tensor
    phase_indices: torch.Tensor
    q_active: torch.Tensor
    poses: torch.Tensor
    jacobians: torch.Tensor
    sphere_indices: torch.Tensor = None
    self_pair_indices: torch.Tensor = None
    environment_sdf_values: torch.Tensor = None


@dataclass
class DenseJacobianBatch:
    candidate_indices: torch.Tensor
    q_dense: torch.Tensor
    poses: torch.Tensor
    jacobians: torch.Tensor
    covers_full_batch: bool


class ActiveJacobianComputer:
    def __init__(self, robot, use_parent_link_kinematics=False):
        self.robot = robot
        self.use_parent_link_kinematics = bool(use_parent_link_kinematics)
        self._parent_subset_jfk_cache = {}

    def _jacobian_fn(self):
        if self.use_parent_link_kinematics:
            return self.robot.jfk_s_collision_spheres_parent_links
        return self.robot.jfk_s_collision_spheres

    def _evaluate(self, q):
        jacobians, poses = self._jacobian_fn()(q)
        return torch.stack(jacobians).transpose(0, 1), torch.stack(poses).transpose(0, 1)

    def _evaluate_parent_subset(self, q, parent_indices, sphere_indices):
        """Evaluate only selected physical parents and reconstruct their spheres."""

        parent_key = tuple(int(value) for value in parent_indices)
        if not parent_key:
            raise ValueError("A parent-link broad-phase bucket cannot be empty.")
        jfk_fn = self._parent_subset_jfk_cache.get(parent_key)
        if jfk_fn is None:
            parent_names = [
                self.robot.collision_sphere_unique_parent_links[index]
                for index in parent_key
            ]
            _, _, jfk_fn = torchkin.get_forward_kinematics_fns(
                robot=self.robot.robot_torchkin,
                link_names=parent_names,
            )
            self._parent_subset_jfk_cache[parent_key] = jfk_fn

        parent_jacobians, parent_poses = jfk_fn(q)
        parent_jacobians = torch.stack(parent_jacobians).transpose(0, 1)
        parent_poses = torch.stack(parent_poses).transpose(0, 1)
        parent_lookup = torch.full(
            (len(self.robot.collision_sphere_unique_parent_links),),
            -1,
            dtype=torch.long,
            device=q.device,
        )
        parent_lookup[torch.as_tensor(parent_key, dtype=torch.long, device=q.device)] = torch.arange(
            len(parent_key), dtype=torch.long, device=q.device
        )
        sphere_indices = sphere_indices.to(device=q.device)
        sphere_parents = self.robot.collision_sphere_parent_indices.to(q.device).index_select(
            0, sphere_indices
        )
        local_parent_indices = parent_lookup.index_select(0, sphere_parents)
        selected_parent_poses = parent_poses.index_select(1, local_parent_indices)
        rotations = selected_parent_poses[..., :3, :3]
        translations = selected_parent_poses[..., :3, 3]
        local_positions = self.robot.collision_sphere_local_positions.to(
            dtype=q.dtype, device=q.device
        ).index_select(0, sphere_indices)
        sphere_positions = (
            torch.einsum("...sij,sj->...si", rotations, local_positions) + translations
        )
        sphere_poses = selected_parent_poses.clone()
        sphere_poses[..., :3, 3] = sphere_positions
        sphere_jacobians = parent_jacobians.index_select(1, local_parent_indices)
        return sphere_jacobians, sphere_poses

    def _evaluate_parent_subset_from_pose_cache(
        self,
        scan_cache,
        candidate_indices,
        phase_indices,
        parent_indices,
        sphere_indices,
    ):
        """Construct active-parent Jacobians without rerunning TorchKin FK."""

        parent_key = tuple(int(value) for value in parent_indices)
        cache_key = ("pose_cache",) + parent_key
        cached_factory = self._parent_subset_jfk_cache.get(cache_key)
        if cached_factory is None:
            parent_names = [
                self.robot.collision_sphere_unique_parent_links[index]
                for index in parent_key
            ]
            _, jacobian_from_poses, related_ids, _ = (
                torchkin.get_forward_kinematics_pose_cache_fns(
                    robot=self.robot.robot_torchkin,
                    link_names=parent_names,
                )
            )
            cached_factory = (jacobian_from_poses, related_ids)
            self._parent_subset_jfk_cache[cache_key] = cached_factory
        jacobian_from_poses, related_ids = cached_factory

        scan_related_lookup = {
            link_id: index
            for index, link_id in enumerate(scan_cache.related_link_ids)
        }
        selected_related_poses = tuple(
            scan_cache.related_poses[scan_related_lookup[link_id]][
                candidate_indices[:, None], phase_indices
            ].reshape(-1, 3, 4)
            for link_id in related_ids
        )
        parent_jacobians, _ = jacobian_from_poses(selected_related_poses)
        parent_jacobians = torch.stack(parent_jacobians).transpose(0, 1)

        parent_lookup = torch.full(
            (len(self.robot.collision_sphere_unique_parent_links),),
            -1,
            dtype=torch.long,
            device=candidate_indices.device,
        )
        parent_lookup[
            torch.as_tensor(parent_key, dtype=torch.long, device=candidate_indices.device)
        ] = torch.arange(
            len(parent_key), dtype=torch.long, device=candidate_indices.device
        )
        sphere_parents = self.robot.collision_sphere_parent_indices.to(
            candidate_indices.device
        ).index_select(0, sphere_indices)
        local_parent_indices = parent_lookup.index_select(0, sphere_parents)
        sphere_jacobians = parent_jacobians.index_select(1, local_parent_indices)
        sphere_poses = scan_cache.sphere_poses[
            candidate_indices[:, None], phase_indices
        ].index_select(-3, sphere_indices).reshape(
            -1, sphere_indices.numel(), 3, 4
        )
        return sphere_jacobians, sphere_poses

    def compute_dense(self, q_dense, candidate_indices=None):
        """Evaluate a regular dense batch without temporal gather/scatter."""

        batch, horizon, dof = q_dense.shape
        covers_full_batch = candidate_indices is None
        if covers_full_batch:
            candidate_indices = torch.arange(batch, dtype=torch.long, device=q_dense.device)
            q_selected = q_dense
        else:
            candidate_indices = torch.as_tensor(
                candidate_indices,
                dtype=torch.long,
                device=q_dense.device,
            )
            if candidate_indices.numel() == 0:
                return None
            q_selected = q_dense.index_select(0, candidate_indices)

        jacobians, poses = self._evaluate(q_selected.reshape(-1, dof))
        jacobians = jacobians.reshape(q_selected.shape[0], horizon, *jacobians.shape[1:])
        poses = poses.reshape(q_selected.shape[0], horizon, *poses.shape[1:])
        return DenseJacobianBatch(
            candidate_indices=candidate_indices,
            q_dense=q_selected,
            poses=poses,
            jacobians=jacobians,
            covers_full_batch=covers_full_batch,
        )

    def compute(self, q_dense, active_indices):
        groups = {}
        for candidate_idx, indices in enumerate(active_indices):
            groups.setdefault(int(indices.numel()), []).append(candidate_idx)

        buckets = []
        for bucket_size, candidate_indices_list in sorted(groups.items()):
            if bucket_size == 0:
                continue
            candidate_indices = torch.tensor(
                candidate_indices_list,
                dtype=torch.long,
                device=q_dense.device,
            )
            phase_indices = torch.stack([active_indices[idx] for idx in candidate_indices_list])
            q_active = q_dense[candidate_indices[:, None], phase_indices]
            flat_q = q_active.reshape(-1, q_active.shape[-1])
            jacobians, poses = self._evaluate(flat_q)
            jacobians = jacobians.reshape(
                candidate_indices.numel(), bucket_size, *jacobians.shape[1:]
            )
            poses = poses.reshape(candidate_indices.numel(), bucket_size, *poses.shape[1:])
            buckets.append(
                ActiveJacobianBucket(
                    candidate_indices=candidate_indices,
                    phase_indices=phase_indices,
                    q_active=q_active,
                    poses=poses,
                    jacobians=jacobians,
                )
            )
        return buckets

    def compute_selection(self, q_dense, selection, exclude_full_horizon=False):
        """Evaluate packed temporal buckets without per-candidate Python gather."""

        horizon = q_dense.shape[1]
        buckets = []
        bucket_options = selection.bucket_options or tuple(
            sorted({int(value) for value in selection.bucket_sizes.detach().cpu().tolist() if value > 0})
        )
        for bucket_size in bucket_options:
            if bucket_size == 0 or (exclude_full_horizon and bucket_size == horizon):
                continue
            candidate_indices = torch.nonzero(
                selection.bucket_sizes == bucket_size,
                as_tuple=False,
            ).flatten()
            if candidate_indices.numel() == 0:
                continue
            phase_indices = selection.active_index_matrix.index_select(
                0, candidate_indices
            )[:, :bucket_size]
            q_active = q_dense[candidate_indices[:, None], phase_indices]
            jacobians, poses = self._evaluate(q_active.reshape(-1, q_active.shape[-1]))
            jacobians = jacobians.reshape(
                candidate_indices.numel(), bucket_size, *jacobians.shape[1:]
            )
            poses = poses.reshape(
                candidate_indices.numel(), bucket_size, *poses.shape[1:]
            )
            buckets.append(
                ActiveJacobianBucket(
                    candidate_indices=candidate_indices,
                    phase_indices=phase_indices,
                    q_active=q_active,
                    poses=poses,
                    jacobians=jacobians,
                )
            )
        return buckets

    def compute_selection_link_broad_phase(
        self, q_dense, selection, reuse_scan_cache=False
    ):
        """Evaluate temporal/parent-mask groups with subset parent-link J-FK."""

        if selection.parent_link_mask is None:
            raise ValueError("Link broad phase requires parent_link_mask in TemporalSelection.")
        horizon = q_dense.shape[1]
        n_parents = selection.parent_link_mask.shape[-1]
        bit_values = 1 << torch.arange(
            n_parents, dtype=torch.long, device=q_dense.device
        )
        parent_keys = (selection.parent_link_mask.to(torch.long) * bit_values).sum(dim=-1)
        sphere_parent_indices = self.robot.collision_sphere_parent_indices.to(q_dense.device)
        self_pairs = self.robot.link_self_collision_tuples
        pair_1 = torch.as_tensor(
            [value[0] for value in self_pairs], dtype=torch.long, device=q_dense.device
        )
        pair_2 = torch.as_tensor(
            [value[1] for value in self_pairs], dtype=torch.long, device=q_dense.device
        )
        scan_cache = (
            selection.fine_sphere_scan_cache if reuse_scan_cache else None
        )

        buckets = []
        for bucket_size in selection.bucket_options or (horizon,):
            if bucket_size == 0:
                continue
            for parent_key in selection.parent_group_keys:
                if parent_key == 0:
                    continue
                candidate_indices = torch.nonzero(
                    (selection.bucket_sizes == bucket_size) & (parent_keys == parent_key),
                    as_tuple=False,
                ).flatten()
                if candidate_indices.numel() == 0:
                    continue
                parent_indices = tuple(
                    index for index in range(n_parents) if parent_key & (1 << index)
                )
                parent_tensor = torch.as_tensor(
                    parent_indices, dtype=torch.long, device=q_dense.device
                )
                sphere_indices = torch.nonzero(
                    torch.isin(sphere_parent_indices, parent_tensor), as_tuple=False
                ).flatten()
                self_pair_indices = (
                    torch.nonzero(
                        torch.isin(pair_1, sphere_indices) & torch.isin(pair_2, sphere_indices),
                        as_tuple=False,
                    ).flatten()
                    if pair_1.numel()
                    else torch.empty(0, dtype=torch.long, device=q_dense.device)
                )
                phase_indices = selection.active_index_matrix.index_select(
                    0, candidate_indices
                )[:, :bucket_size]
                q_active = q_dense[candidate_indices[:, None], phase_indices]
                if scan_cache is not None:
                    jacobians, poses = self._evaluate_parent_subset_from_pose_cache(
                        scan_cache,
                        candidate_indices,
                        phase_indices,
                        parent_indices,
                        sphere_indices,
                    )
                else:
                    jacobians, poses = self._evaluate_parent_subset(
                        q_active.reshape(-1, q_active.shape[-1]),
                        parent_indices,
                        sphere_indices,
                    )
                jacobians = jacobians.reshape(
                    candidate_indices.numel(), bucket_size, *jacobians.shape[1:]
                )
                poses = poses.reshape(
                    candidate_indices.numel(), bucket_size, *poses.shape[1:]
                )
                environment_sdf_values = None
                if (
                    scan_cache is not None
                    and scan_cache.environment_sdf_values is not None
                ):
                    environment_sdf_values = scan_cache.environment_sdf_values[
                        candidate_indices[:, None], phase_indices
                    ].index_select(-1, sphere_indices)
                buckets.append(
                    ActiveJacobianBucket(
                        candidate_indices=candidate_indices,
                        phase_indices=phase_indices,
                        q_active=q_active,
                        poses=poses,
                        jacobians=jacobians,
                        sphere_indices=sphere_indices,
                        self_pair_indices=self_pair_indices,
                        environment_sdf_values=environment_sdf_values,
                    )
                )
        return buckets
