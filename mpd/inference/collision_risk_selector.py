"""Deterministic temporal collision-risk selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch

from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor


def fixed_sample_indices(horizon: int, count: int, device=None) -> torch.Tensor:
    count = min(max(int(count), 2), int(horizon))
    indices = torch.linspace(0, horizon - 1, count, device=device).round().long()
    indices[0] = 0
    indices[-1] = horizon - 1
    return torch.unique_consecutive(indices)


def nonuniform_trapezoid_weights(phase: torch.Tensor) -> torch.Tensor:
    """Return weights whose dot product matches torch.trapezoid(y, phase)."""

    if phase.ndim != 1 or phase.numel() < 2:
        raise ValueError("phase must be a one-dimensional tensor with at least two values.")
    delta = torch.diff(phase)
    if torch.any(delta <= 0):
        raise ValueError("phase values must be strictly increasing.")
    weights = torch.empty_like(phase)
    weights[0] = delta[0] / 2
    weights[-1] = delta[-1] / 2
    if phase.numel() > 2:
        weights[1:-1] = (delta[:-1] + delta[1:]) / 2
    return weights


class PackedActiveIndices:
    """Lazy compatibility view over GPU-packed temporal phase indices."""

    def __init__(self, matrix, bucket_sizes):
        self.matrix = matrix
        self.bucket_sizes = bucket_sizes

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, candidate_idx):
        bucket = int(self.bucket_sizes[candidate_idx].detach().cpu().item())
        return self.matrix[candidate_idx, :bucket]

    def __iter__(self):
        for candidate_idx in range(len(self)):
            yield self[candidate_idx]


@dataclass
class FineSphereScanCache:
    """Detached C1 scan products reusable by the first precise guide pass."""

    related_link_ids: tuple
    related_poses: tuple
    sphere_poses: torch.Tensor
    environment_sdf_values: torch.Tensor = None
    self_pair_distances: torch.Tensor = None


@dataclass
class TemporalSelection:
    active_indices: list
    bucket_sizes: torch.Tensor
    risk_mask: torch.Tensor
    environment_clearance: torch.Tensor
    self_clearance: torch.Tensor
    active_index_matrix: torch.Tensor = None
    bucket_options: tuple = ()
    predicted_active_ratio: float = 1.0
    temporal_sparse_applied: bool = False
    parent_link_mask: torch.Tensor = None
    environment_sphere_mask: torch.Tensor = None
    self_pair_mask: torch.Tensor = None
    parent_group_keys: tuple = ()
    fine_sphere_scan_cache: FineSphereScanCache = None
    span_certificate_statistics: dict = None

    def __post_init__(self):
        if self.active_index_matrix is None:
            horizon = self.risk_mask.shape[1]
            self.active_index_matrix = torch.full(
                (self.bucket_sizes.shape[0], horizon),
                -1,
                dtype=torch.long,
                device=self.bucket_sizes.device,
            )
            for candidate_idx, indices in enumerate(self.active_indices):
                self.active_index_matrix[candidate_idx, : indices.numel()] = indices
        if self.active_indices is None:
            self.active_indices = PackedActiveIndices(
                self.active_index_matrix,
                self.bucket_sizes,
            )
        if not self.bucket_options:
            self.bucket_options = tuple(
                sorted({int(value) for value in self.bucket_sizes.detach().cpu().tolist() if value > 0})
            )
        if self.parent_link_mask is not None and not self.parent_group_keys:
            bit_values = 1 << torch.arange(
                self.parent_link_mask.shape[-1],
                dtype=torch.long,
                device=self.parent_link_mask.device,
            )
            keys = (self.parent_link_mask.to(torch.long) * bit_values).sum(dim=-1)
            self.parent_group_keys = tuple(
                int(value) for value in torch.unique(keys).detach().cpu().tolist()
            )

    @property
    def active_counts(self):
        return self.bucket_sizes


class CollisionRiskSelector:
    def __init__(
        self,
        robot,
        config,
        use_parent_link_kinematics=False,
        candidate_pruning_enabled=False,
        link_broad_phase_config=None,
        use_parent_bounds_scan=False,
        span_certificate_config=None,
        guidance_profiler=None,
    ):
        self.robot = robot
        self.config = config
        self.use_parent_link_kinematics = bool(use_parent_link_kinematics)
        self.candidate_pruning_enabled = bool(candidate_pruning_enabled)
        self.link_broad_phase_config = dict(link_broad_phase_config or {})
        self.use_parent_bounds_scan = bool(use_parent_bounds_scan)
        self.span_certificate_config = dict(span_certificate_config or {})
        self.guidance_profiler = guidance_profiler
        self._sphere_jacobian_column_bounds_cache = None
        self._span_metadata_cache = {}

    def _profiled(self, name, fn):
        """Run a GPU stage and return (value, elapsed seconds).

        Synchronizing each stage is intentionally restricted to explicit
        diagnostic runs. Normal latency benchmarks leave profile_stages off.
        """

        if not bool(self.span_certificate_config.get("profile_stages", False)):
            return fn(), 0.0
        if self.guidance_profiler is not None:
            with self.guidance_profiler.section(name):
                value = fn()
            return value, 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return value, time.perf_counter() - start

    @staticmethod
    def _joint_origin_norm(joint):
        origin = getattr(joint, "origin", None)
        xyz = getattr(origin, "xyz", None) if origin is not None else None
        if xyz is None:
            return 0.0
        return math.sqrt(sum(float(value) ** 2 for value in xyz))

    def _sphere_jacobian_column_bounds(self, dtype, device):
        """Configuration-independent translational Jacobian-column bounds.

        For a revolute ancestor j, ||J[:,j]|| is bounded by the sum of all
        downstream joint-origin translation lengths plus the fine sphere's
        local offset. This triangle-inequality reach bound is conservative for
        every robot configuration and requires no runtime Jacobian.
        """

        cached = self._sphere_jacobian_column_bounds_cache
        if cached is not None:
            return cached.to(dtype=dtype, device=device)

        robot_urdf = self.robot.robot_urdf
        child_to_joint = {joint.child: joint for joint in robot_urdf.joints}
        actuated = [joint for joint in robot_urdf.joints if joint.joint_type != "fixed"]
        actuated_index = {joint.name: index for index, joint in enumerate(actuated)}
        bounds = torch.zeros(
            (len(self.robot.collision_sphere_parent_links), len(actuated)),
            dtype=torch.float64,
        )
        local_positions = self.robot.collision_sphere_local_positions.detach().cpu().double()
        for sphere_index, parent_link in enumerate(self.robot.collision_sphere_parent_links):
            path = []
            current_link = parent_link
            while current_link in child_to_joint:
                joint = child_to_joint[current_link]
                path.append(joint)
                current_link = joint.parent
            path.reverse()
            local_reach = float(torch.linalg.norm(local_positions[sphere_index]).item())
            downstream_origin_norms = [self._joint_origin_norm(joint) for joint in path]
            for path_index, joint in enumerate(path):
                if joint.joint_type == "fixed":
                    continue
                column_index = actuated_index[joint.name]
                if joint.joint_type in {"revolute", "continuous"}:
                    bounds[sphere_index, column_index] = local_reach + sum(
                        downstream_origin_norms[path_index + 1 :]
                    )
                elif joint.joint_type == "prismatic":
                    bounds[sphere_index, column_index] = 1.0
                else:
                    raise NotImplementedError(
                        f"Unsupported joint type for span certificate: {joint.joint_type}."
                    )
        self._sphere_jacobian_column_bounds_cache = bounds
        return bounds.to(dtype=dtype, device=device)

    def _span_metadata(self, parametric_trajectory, dtype, device):
        bspline = parametric_trajectory.bspline
        key = (id(bspline), str(dtype), str(device))
        cached = self._span_metadata_cache.get(key)
        if cached is not None:
            return cached
        knots = torch.as_tensor(bspline.u, dtype=dtype, device=device)
        unique_knots = torch.unique_consecutive(knots)
        left = unique_knots[:-1]
        right = unique_knots[1:]
        nonempty = right > left
        left = left[nonempty]
        right = right[nonempty]
        # The derivative spline has degree p-1 and knot vector U[1:-1].
        derivative_left = knots[1 : bspline.n_pts]
        derivative_right = knots[bspline.d + 1 : bspline.n_pts + bspline.d]
        active = (derivative_left[None, :] < right[:, None]) & (
            derivative_right[None, :] > left[:, None]
        )
        cached = (left, right, active)
        self._span_metadata_cache[key] = cached
        return cached

    @staticmethod
    def _grid_error_and_in_bounds(environment_field, positions, scale):
        if environment_field is None:
            return 0.0, torch.ones(
                positions.shape[:-2], dtype=torch.bool, device=positions.device
            )
        df_fn = getattr(environment_field, "df_obj_list_fn", None)
        dfs = [] if df_fn is None else list(df_fn())
        max_error = 0.0
        in_bounds = torch.ones(
            positions.shape[:-2], dtype=torch.bool, device=positions.device
        )
        for df in dfs:
            if not all(hasattr(df, name) for name in ("limits", "map_dim", "cmap_dim")):
                continue
            limits = df.limits.to(dtype=positions.dtype, device=positions.device)
            inside = ((positions >= limits[0]) & (positions <= limits[1])).all(dim=-1).all(dim=-1)
            in_bounds &= inside
            cmap_dim = df.cmap_dim.to(dtype=positions.dtype, device=positions.device)
            spacing = df.map_dim.to(dtype=positions.dtype, device=positions.device) / torch.clamp(
                cmap_dim - 1, min=1
            )
            # The current nearest-grid projection uses cmap_dim rather than
            # cmap_dim-1. One full diagonal cell is a conservative allowance.
            max_error = max(max_error, float(torch.linalg.norm(spacing).detach().cpu().item()))
        return float(scale) * max_error, in_bounds

    @staticmethod
    def _exact_environment_clearance_by_sphere(field, positions):
        """Evaluate GridMapSDF source primitives for a continuous certificate."""

        if field is None:
            return torch.full(
                positions.shape[:-1],
                torch.inf,
                dtype=positions.dtype,
                device=positions.device,
            )
        df_fn = getattr(field, "df_obj_list_fn", None)
        dfs = [] if df_fn is None else list(df_fn())
        if not dfs or not all(hasattr(df, "compute_signed_distance_raw") for df in dfs):
            return None
        signed_distances = torch.stack(
            [df.compute_signed_distance_raw(positions) for df in dfs], dim=-2
        )
        radii = torch.as_tensor(
            field.collision_margins,
            dtype=positions.dtype,
            device=positions.device,
        )
        clearance = signed_distances - radii
        return clearance.amin(dim=-2)

    @staticmethod
    def _dilate(mask, radius):
        if radius <= 0:
            return mask
        result = mask.clone()
        for offset in range(1, radius + 1):
            result[:, offset:] |= mask[:, :-offset]
            result[:, :-offset] |= mask[:, offset:]
        return result

    @staticmethod
    def _environment_clearance_by_sphere(field, positions):
        if field is None:
            return torch.full(
                positions.shape[:-1],
                torch.inf,
                dtype=positions.dtype,
                device=positions.device,
            )
        signed_distance = field.object_signed_distances(positions)
        return CollisionRiskSelector._environment_clearance_from_signed_distance(
            field, signed_distance
        )

    @staticmethod
    def _environment_clearance_from_signed_distance(field, signed_distance):
        radii = torch.as_tensor(
            field.collision_margins,
            dtype=signed_distance.dtype,
            device=signed_distance.device,
        )
        while radii.ndim < signed_distance.ndim:
            radii = radii.unsqueeze(0)
        clearance = signed_distance - radii
        # Object fields add an object axis immediately before the sphere axis.
        while clearance.ndim > 3:
            clearance = clearance.amin(dim=-2)
        return clearance

    @classmethod
    def _minimum_environment_clearance(cls, field, positions):
        return cls._environment_clearance_by_sphere(field, positions).amin(dim=-1)

    @staticmethod
    def _minimum_self_clearance(field, positions):
        if field is None:
            return torch.full(positions.shape[:2], torch.inf, dtype=positions.dtype, device=positions.device)
        return field.compute_embodiment_signed_distances(None, positions).amin(dim=-1)

    @staticmethod
    def _self_clearance_by_pair(field, positions):
        if field is None:
            return torch.empty(
                (*positions.shape[:2], 0),
                dtype=positions.dtype,
                device=positions.device,
            )
        return field.compute_embodiment_signed_distances(None, positions)

    def compute_clearances(
        self,
        q_dense,
        environment_field=None,
        self_field=None,
        return_details=False,
        return_scan_cache=False,
    ):
        batch, horizon, _ = q_dense.shape
        related_poses = None
        if return_scan_cache and self.use_parent_link_kinematics:
            related_poses = self.robot.fk_collision_parent_pose_cache(
                q_dense.reshape(batch * horizon, -1)
            )
            poses = self.robot.collision_sphere_poses_from_related_pose_cache(
                related_poses
            )
        else:
            fk_fn = self.robot.fk_collision_spheres
            if self.use_parent_link_kinematics:
                fk_fn = self.robot.fk_collision_spheres_parent_links
            poses = fk_fn(q_dense.reshape(batch * horizon, -1))
        poses = torch.stack(poses).transpose(0, 1).reshape(batch, horizon, -1, 3, 4)
        positions = link_pos_from_link_tensor(poses)[..., : self.robot.task_space_dim]
        environment_sdf_values = None
        if environment_field is None:
            environment_by_sphere = torch.full(
                positions.shape[:-1],
                torch.inf,
                dtype=positions.dtype,
                device=positions.device,
            )
        else:
            environment_sdf_values = environment_field.object_signed_distances(
                positions
            )
            environment_by_sphere = self._environment_clearance_from_signed_distance(
                environment_field, environment_sdf_values
            )
        self_by_pair = self._self_clearance_by_pair(self_field, positions)
        environment = environment_by_sphere.amin(dim=-1)
        self_clearance = (
            self_by_pair.amin(dim=-1)
            if self_by_pair.shape[-1]
            else torch.full_like(environment, torch.inf)
        )
        if return_details or return_scan_cache:
            scan_cache = None
            if return_scan_cache:
                reshaped_related_poses = tuple(
                    pose.reshape(batch, horizon, 3, 4) for pose in related_poses
                )
                scan_cache = FineSphereScanCache(
                    related_link_ids=tuple(
                        self.robot.collision_parent_related_link_ids
                    ),
                    related_poses=reshaped_related_poses,
                    sphere_poses=poses,
                    environment_sdf_values=environment_sdf_values,
                    self_pair_distances=self_by_pair,
                )
            return (
                environment,
                self_clearance,
                environment_by_sphere,
                self_by_pair,
                scan_cache,
            )
        return environment, self_clearance

    def compute_parent_bound_clearances(
        self,
        q_dense,
        environment_field=None,
        self_field=None,
    ):
        """Scan conservative physical-parent bounds without expanding fine spheres."""

        batch, horizon, _ = q_dense.shape
        positions = self.robot.fk_collision_parent_bounds(
            q_dense.reshape(batch * horizon, -1)
        ).reshape(batch, horizon, -1, 3)
        positions = positions[..., : self.robot.task_space_dim]
        if environment_field is None:
            environment_by_bound = torch.full(
                positions.shape[:-1],
                torch.inf,
                dtype=positions.dtype,
                device=positions.device,
            )
        else:
            signed_distance = environment_field.object_signed_distances(positions)
            radii = self.robot.collision_parent_bound_radii.to(
                dtype=positions.dtype, device=positions.device
            )
            while radii.ndim < signed_distance.ndim:
                radii = radii.unsqueeze(0)
            environment_by_bound = signed_distance - radii
            # Object fields insert an object axis immediately before bounds.
            while environment_by_bound.ndim > 3:
                environment_by_bound = environment_by_bound.amin(dim=-2)

        n_parents = len(self.robot.collision_sphere_unique_parent_links)
        bound_parents = self.robot.collision_parent_bound_parent_indices.to(
            positions.device
        )
        environment_by_parent = torch.full(
            (batch, horizon, n_parents),
            torch.inf,
            dtype=positions.dtype,
            device=positions.device,
        )
        for parent_idx in range(n_parents):
            parent_bounds = torch.nonzero(
                bound_parents == parent_idx, as_tuple=False
            ).flatten()
            environment_by_parent[..., parent_idx] = environment_by_bound.index_select(
                -1, parent_bounds
            ).amin(dim=-1)

        parent_pairs = self.robot.collision_parent_self_pairs.to(positions.device)
        if self_field is None or parent_pairs.numel() == 0:
            self_by_parent_pair = torch.empty(
                (batch, horizon, 0),
                dtype=positions.dtype,
                device=positions.device,
            )
        else:
            radii = self.robot.collision_parent_bound_radii.to(
                dtype=positions.dtype, device=positions.device
            )
            bound_pairs = self.robot.collision_parent_bound_self_pairs.to(
                positions.device
            )
            pair_1 = bound_pairs[:, 0]
            pair_2 = bound_pairs[:, 1]
            clearance_by_bound_pair = (
                torch.linalg.norm(
                    positions.index_select(-2, pair_1)
                    - positions.index_select(-2, pair_2),
                    dim=-1,
                )
                - radii.index_select(0, pair_1)
                - radii.index_select(0, pair_2)
            )
            pair_groups = self.robot.collision_parent_bound_self_pair_groups.to(
                positions.device
            )
            self_by_parent_pair = torch.full(
                (batch, horizon, parent_pairs.shape[0]),
                torch.inf,
                dtype=positions.dtype,
                device=positions.device,
            )
            for pair_group in range(parent_pairs.shape[0]):
                group_indices = torch.nonzero(
                    pair_groups == pair_group, as_tuple=False
                ).flatten()
                self_by_parent_pair[..., pair_group] = clearance_by_bound_pair.index_select(
                    -1, group_indices
                ).amin(dim=-1)

        environment = environment_by_parent.amin(dim=-1)
        self_clearance = (
            self_by_parent_pair.amin(dim=-1)
            if self_by_parent_pair.shape[-1]
            else torch.full_like(environment, torch.inf)
        )
        return environment, self_clearance, environment_by_parent, self_by_parent_pair

    def _expand_parent_link_mask(self, parent_link_mask):
        sphere_parent = self.robot.collision_sphere_parent_indices.to(
            device=parent_link_mask.device
        )
        sphere_mask = parent_link_mask[:, sphere_parent]
        self_pairs = self.robot.df_collision_self
        if self_pairs is not None and len(self_pairs.link_idx_1):
            pair_1 = torch.as_tensor(
                self_pairs.link_idx_1,
                dtype=torch.long,
                device=parent_link_mask.device,
            )
            pair_2 = torch.as_tensor(
                self_pairs.link_idx_2,
                dtype=torch.long,
                device=parent_link_mask.device,
            )
            self_pair_mask = sphere_mask[:, pair_1] & sphere_mask[:, pair_2]
        else:
            self_pair_mask = torch.zeros(
                (parent_link_mask.shape[0], 0),
                dtype=torch.bool,
                device=parent_link_mask.device,
            )
        return parent_link_mask, sphere_mask, self_pair_mask

    def _link_broad_phase_masks_from_parent_bounds(
        self, environment_by_parent, self_by_parent_pair
    ):
        """Build conservative candidate-level masks from coarse parent bounds."""

        parent_link_mask = environment_by_parent.amin(dim=1) < float(
            self.link_broad_phase_config.get("environment_margin", 0.20)
        )
        if self_by_parent_pair.shape[-1]:
            active_pairs = self_by_parent_pair.amin(dim=1) < float(
                self.link_broad_phase_config.get("self_margin", 0.10)
            )
            parent_pairs = self.robot.collision_parent_self_pairs.to(
                environment_by_parent.device
            )
            counts = parent_link_mask.to(torch.long)
            counts.scatter_add_(
                1,
                parent_pairs[:, 0][None, :].expand(parent_link_mask.shape[0], -1),
                active_pairs.to(torch.long),
            )
            counts.scatter_add_(
                1,
                parent_pairs[:, 1][None, :].expand(parent_link_mask.shape[0], -1),
                active_pairs.to(torch.long),
            )
            parent_link_mask = counts > 0
        return self._expand_parent_link_mask(parent_link_mask)

    def _link_broad_phase_masks(self, environment_by_sphere, self_by_pair):
        """Build candidate-level conservative parent-link and pair masks."""

        batch = environment_by_sphere.shape[0]
        sphere_parent = self.robot.collision_sphere_parent_indices.to(
            device=environment_by_sphere.device
        )
        n_parents = len(self.robot.collision_sphere_unique_parent_links)
        environment_sphere_mask = environment_by_sphere.amin(dim=1) < float(
            self.link_broad_phase_config.get("environment_margin", 0.20)
        )
        parent_link_counts = torch.zeros(
            (batch, n_parents), dtype=torch.long, device=environment_by_sphere.device
        )
        parent_link_counts.scatter_add_(
            1,
            sphere_parent[None, :].expand(batch, -1),
            environment_sphere_mask.to(torch.long),
        )

        if self_by_pair.shape[-1]:
            active_self_pairs = self_by_pair.amin(dim=1) < float(
                self.link_broad_phase_config.get("self_margin", 0.10)
            )
            pair_1 = torch.as_tensor(
                self.robot.df_collision_self.link_idx_1,
                dtype=torch.long,
                device=environment_by_sphere.device,
            )
            pair_2 = torch.as_tensor(
                self.robot.df_collision_self.link_idx_2,
                dtype=torch.long,
                device=environment_by_sphere.device,
            )
            pair_parents_1 = sphere_parent[pair_1]
            pair_parents_2 = sphere_parent[pair_2]
            parent_link_counts.scatter_add_(
                1,
                pair_parents_1[None, :].expand(batch, -1),
                active_self_pairs.to(torch.long),
            )
            parent_link_counts.scatter_add_(
                1,
                pair_parents_2[None, :].expand(batch, -1),
                active_self_pairs.to(torch.long),
            )
        parent_link_mask = parent_link_counts > 0
        sphere_mask = parent_link_mask[:, sphere_parent]
        if self_by_pair.shape[-1]:
            self_pair_mask = sphere_mask[:, pair_1] & sphere_mask[:, pair_2]
        else:
            self_pair_mask = torch.zeros(
                (batch, 0), dtype=torch.bool, device=environment_by_sphere.device
            )
        return parent_link_mask, sphere_mask, self_pair_mask

    def _risk_scan_indices(self, horizon, device):
        """Return coarse points plus interval midpoints for the risk scan."""
        coarse = fixed_sample_indices(horizon, self.config["coarse_points"], device)
        if coarse.numel() < 2 or not self.config.get("probe_midpoints", True):
            return coarse
        midpoint = ((coarse[:-1] + coarse[1:]) // 2).long()
        return torch.unique(torch.cat((coarse, midpoint)), sorted=True)

    def select_from_clearances(
        self,
        q_dense,
        environment_clearance,
        self_clearance,
        parent_link_mask=None,
        environment_sphere_mask=None,
        self_pair_mask=None,
        fine_sphere_scan_cache=None,
        risk_mask_override=None,
        temporal_enabled_override=None,
        safe_bucket_override=None,
        span_certificate_statistics=None,
    ):
        batch, horizon, _ = q_dense.shape
        cfg = self.config
        if risk_mask_override is None:
            risk_mask = environment_clearance < float(cfg["environment_refine_margin"])
            risk_mask |= self_clearance < float(cfg["self_refine_margin"])
        else:
            risk_mask = risk_mask_override.clone()

        q_delta_threshold = cfg.get("q_delta_threshold") if risk_mask_override is None else None
        if q_delta_threshold is not None:
            movement = torch.linalg.norm(torch.diff(q_dense, dim=1), dim=-1)
            moving = movement > float(q_delta_threshold)
            risk_mask[:, :-1] |= moving
            risk_mask[:, 1:] |= moving

        if risk_mask_override is None:
            risk_mask = self._dilate(risk_mask, int(cfg["neighbor_dilation"]))
        risk_trigger_mask = risk_mask.clone()
        if risk_mask_override is None and cfg.get("always_keep_endpoints", True):
            risk_mask[:, 0] = True
            risk_mask[:, -1] = True

        # Bucket assignment and phase-index selection stay on the GPU. The old
        # implementation performed bool/nonzero/isin/unique inside a Python
        # loop, forcing one or more device synchronizations per candidate.
        temporal_enabled = (
            bool(temporal_enabled_override)
            if temporal_enabled_override is not None
            else bool(cfg.get("enabled", True)) or bool(cfg.get("conditional_enabled", False))
        )
        allowed_buckets = tuple(
            sorted({min(int(bucket), horizon) for bucket in cfg["buckets"]} | {horizon})
        )
        allowed = torch.tensor(allowed_buckets, dtype=torch.long, device=q_dense.device)
        risk_counts = risk_mask.sum(dim=1)
        risky_candidates = risk_trigger_mask.any(dim=1)
        if temporal_enabled:
            first_fitting_bucket = (risk_counts[:, None] <= allowed[None, :]).to(torch.int64).argmax(dim=1)
            bucket_sizes = allowed[first_fitting_bucket]
        else:
            # Candidate-only pruning: risky candidates retain the regular dense
            # ParentLinkFast path, so no temporal approximation is introduced.
            bucket_sizes = torch.full_like(risk_counts, horizon)
        safe_bucket = (
            int(safe_bucket_override)
            if safe_bucket_override is not None
            else (0 if self.candidate_pruning_enabled else horizon)
        )
        bucket_sizes = torch.where(
            risky_candidates,
            bucket_sizes,
            torch.full_like(bucket_sizes, safe_bucket),
        )
        proposed_active_ratio = float(
            (bucket_sizes.to(q_dense.dtype).sum() / float(batch * horizon))
            .detach()
            .cpu()
            .item()
        )
        temporal_sparse_applied = bool(temporal_enabled)
        if bool(cfg.get("conditional_enabled", False)) and not bool(cfg.get("enabled", True)):
            temporal_sparse_applied = proposed_active_ratio <= float(
                cfg.get("conditional_active_ratio_threshold", 0.35)
            )
            if not temporal_sparse_applied:
                fallback_bucket = 0 if self.candidate_pruning_enabled else horizon
                bucket_sizes = torch.where(
                    risky_candidates,
                    torch.full_like(bucket_sizes, horizon),
                    torch.full_like(bucket_sizes, fallback_bucket),
                )

        # Packed [B, H] representation avoids ragged per-candidate GPU work.
        # Valid indices occupy the first K columns and the remainder stays -1.
        active_index_matrix = torch.full(
            (batch, horizon),
            -1,
            dtype=torch.long,
            device=q_dense.device,
        )
        phase = torch.arange(horizon, dtype=torch.long, device=q_dense.device)
        for bucket in allowed_buckets:
            candidate_indices = torch.nonzero(bucket_sizes == bucket, as_tuple=False).flatten()
            if candidate_indices.numel() == 0:
                continue
            if bucket == horizon:
                selected = phase.expand(candidate_indices.numel(), -1)
            else:
                required = risk_mask.index_select(0, candidate_indices)
                uniform = torch.zeros(horizon, dtype=torch.bool, device=q_dense.device)
                uniform[fixed_sample_indices(horizon, bucket, q_dense.device)] = True

                # Required risk points rank first, then uniform coverage points,
                # then remaining phases. Within a rank, earlier phase wins. As
                # risk_count <= bucket, every risk point is guaranteed retained.
                priority = torch.where(
                    required,
                    phase[None, :],
                    torch.where(
                        uniform[None, :],
                        horizon + phase[None, :],
                        2 * horizon + phase[None, :],
                    ),
                )
                selected = torch.topk(priority, bucket, dim=1, largest=False, sorted=False).indices
                selected = torch.sort(selected, dim=1).values
            active_index_matrix[candidate_indices, :bucket] = selected

        return TemporalSelection(
            active_indices=None,
            bucket_sizes=bucket_sizes,
            risk_mask=risk_mask,
            environment_clearance=environment_clearance,
            self_clearance=self_clearance,
            active_index_matrix=active_index_matrix,
            bucket_options=allowed_buckets,
            predicted_active_ratio=proposed_active_ratio,
            temporal_sparse_applied=temporal_sparse_applied,
            parent_link_mask=parent_link_mask,
            environment_sphere_mask=environment_sphere_mask,
            self_pair_mask=self_pair_mask,
            fine_sphere_scan_cache=fine_sphere_scan_cache,
            span_certificate_statistics=span_certificate_statistics,
        )

    def select_span_certificate(
        self,
        q_dense,
        control_points,
        parametric_trajectory,
        environment_field=None,
        self_field=None,
    ):
        """Certify collision-free B-spline spans and refine only uncertainty.

        The screening path deliberately evaluates the original fine spheres
        with FK-only plus distance queries. It does not create or reuse the C1
        pose/SDF cache; uncertain leaves are handed to the regular B3 J-FK and
        sparse J^Tg path.
        """

        cfg = self.span_certificate_config
        batch, horizon, dof = q_dense.shape
        if control_points is None or parametric_trajectory is None:
            raise ValueError("span_certificate requires B-spline control points and trajectory.")

        def derivative_stage():
            derivative_cps = parametric_trajectory.get_control_points_derivatives(
                control_points.detach(), get_type="vel", get_time_repr=False
            )
            span_left, span_right, derivative_active = self._span_metadata(
                parametric_trajectory, q_dense.dtype, q_dense.device
            )
            abs_derivative = derivative_cps.abs()
            # [B, span, derivative-cp, dof], inactive entries cannot win max.
            per_joint_bound = torch.where(
                derivative_active[None, :, :, None],
                abs_derivative[:, None, :, :],
                torch.zeros((), dtype=q_dense.dtype, device=q_dense.device),
            ).amax(dim=2)
            return span_left, span_right, per_joint_bound

        (span_left, span_right, per_joint_bound), derivative_time = self._profiled(
            "span_derivative_bound", derivative_stage
        )
        n_spans = int(span_left.numel())
        initial_intervals = batch * n_spans
        sphere_column_bounds = self._sphere_jacobian_column_bounds(
            q_dense.dtype, q_dense.device
        )
        if sphere_column_bounds.shape[-1] != dof:
            raise ValueError(
                "Span-certificate Jacobian bound DOF does not match trajectory DOF."
            )
        mode = cfg.get("jacobian_bound_mode", "componentwise")
        if mode == "componentwise":
            sphere_speed = torch.einsum(
                "bsd,nd->bsn", per_joint_bound, sphere_column_bounds
            )
        else:
            velocity_norm = torch.linalg.norm(per_joint_bound, dim=-1)
            jacobian_norm = torch.linalg.norm(sphere_column_bounds, dim=-1)
            sphere_speed = velocity_norm[..., None] * jacobian_norm[None, None, :]

        if self_field is not None and len(self.robot.link_self_collision_tuples):
            pair_1 = torch.as_tensor(
                [value[0] for value in self.robot.link_self_collision_tuples],
                dtype=torch.long,
                device=q_dense.device,
            )
            pair_2 = torch.as_tensor(
                [value[1] for value in self.robot.link_self_collision_tuples],
                dtype=torch.long,
                device=q_dense.device,
            )
            pair_speed = sphere_speed.index_select(-1, pair_1) + sphere_speed.index_select(
                -1, pair_2
            )
        else:
            pair_speed = torch.empty(
                (batch, n_spans, 0), dtype=q_dense.dtype, device=q_dense.device
            )

        candidate_indices = torch.arange(batch, device=q_dense.device).repeat_interleave(
            n_spans
        )
        base_span_indices = torch.arange(n_spans, device=q_dense.device).repeat(batch)
        interval_left = span_left.repeat(batch)
        interval_right = span_right.repeat(batch)
        phase = parametric_trajectory.phase_time.s.detach().to(
            dtype=q_dense.dtype, device=q_dense.device
        )
        unresolved_leaves = None
        depth_statistics = []
        total_fk_time = total_sdf_time = total_bound_time = 0.0
        midpoint_queries = 0
        first_level_certified = 0
        initial_measure = float(batch) * float((span_right - span_left).sum().item())
        max_depth = int(cfg.get("max_subdivision_depth", 3))
        last_environment = torch.full(
            (batch, horizon), torch.inf, dtype=q_dense.dtype, device=q_dense.device
        )
        last_self = torch.full_like(last_environment, torch.inf)

        for depth in range(max_depth + 1):
            if candidate_indices.numel() == 0:
                depth_statistics.append(
                    {
                        "depth": depth,
                        "evaluated": 0,
                        "certified": 0,
                        "remaining": 0,
                        "remaining_count_ratio": 0.0,
                        "remaining_measure_ratio": 0.0,
                    }
                )
                break
            midpoint = (interval_left + interval_right) / 2
            phase_indices = torch.searchsorted(phase, midpoint).clamp(max=horizon - 1)
            previous_indices = (phase_indices - 1).clamp(min=0)
            choose_previous = (
                (phase[previous_indices] - midpoint).abs()
                <= (phase[phase_indices] - midpoint).abs()
            )
            phase_indices = torch.where(choose_previous, previous_indices, phase_indices)
            q_query = q_dense[candidate_indices, phase_indices]
            midpoint_queries += int(q_query.shape[0])

            def fk_stage():
                poses = self.robot.fk_collision_spheres_parent_links(q_query)
                poses = torch.stack(poses).transpose(0, 1)
                return poses, link_pos_from_link_tensor(poses)[..., : self.robot.task_space_dim]

            (poses, positions), fk_time = self._profiled("span_midpoint_fk", fk_stage)
            total_fk_time += fk_time

            def sdf_stage():
                positions_with_horizon = positions[:, None]
                environment_by_sphere = None
                if bool(cfg.get("exact_sdf_for_certificate", True)):
                    environment_by_sphere = self._exact_environment_clearance_by_sphere(
                        environment_field, positions
                    )
                if environment_by_sphere is None:
                    environment_by_sphere = self._environment_clearance_by_sphere(
                        environment_field, positions_with_horizon
                    )[:, 0]
                self_by_pair = self._self_clearance_by_pair(
                    self_field, positions_with_horizon
                )[:, 0]
                return environment_by_sphere, self_by_pair

            (environment_by_sphere, self_by_pair), sdf_time = self._profiled(
                "span_midpoint_sdf", sdf_stage
            )
            total_sdf_time += sdf_time
            last_environment[candidate_indices, phase_indices] = environment_by_sphere.amin(dim=-1)
            if self_by_pair.shape[-1]:
                last_self[candidate_indices, phase_indices] = self_by_pair.amin(dim=-1)

            def bound_stage():
                radius = torch.maximum(
                    phase[phase_indices] - interval_left,
                    interval_right - phase[phase_indices],
                )
                if bool(cfg.get("exact_sdf_for_certificate", True)):
                    grid_error = 0.0
                    in_bounds = torch.ones(
                        positions.shape[0], dtype=torch.bool, device=positions.device
                    )
                else:
                    grid_error, in_bounds = self._grid_error_and_in_bounds(
                        environment_field,
                        positions,
                        cfg.get("grid_error_scale", 1.0),
                    )
                selected_speed = sphere_speed[candidate_indices, base_span_indices]
                environment_lb = environment_by_sphere - selected_speed * radius[:, None]
                environment_lb = environment_lb - grid_error
                environment_safe = (
                    environment_lb > float(cfg.get("environment_safe_margin", 0.08))
                ).all(dim=-1)
                environment_safe &= in_bounds
                if self_by_pair.shape[-1]:
                    selected_pair_speed = pair_speed[candidate_indices, base_span_indices]
                    self_lb = self_by_pair - selected_pair_speed * radius[:, None]
                    self_safe = (
                        self_lb > float(cfg.get("self_safe_margin", 0.06))
                    ).all(dim=-1)
                else:
                    self_safe = torch.ones_like(environment_safe)
                return environment_safe & self_safe, environment_safe, self_safe

            (certified, environment_certified, self_certified), bound_time = self._profiled(
                "span_bound_arithmetic", bound_stage
            )
            total_bound_time += bound_time
            certified_count = int(certified.sum().detach().cpu().item())
            if depth == 0:
                first_level_certified = certified_count
            uncertain = ~certified
            remaining_measure = float(
                (interval_right[uncertain] - interval_left[uncertain]).sum().detach().cpu().item()
            )
            depth_statistics.append(
                {
                    "depth": depth,
                    "evaluated": int(certified.numel()),
                    "certified": certified_count,
                    "environment_certified": int(
                        environment_certified.sum().detach().cpu().item()
                    ),
                    "self_certified": int(self_certified.sum().detach().cpu().item()),
                    "remaining": int(uncertain.sum().detach().cpu().item()),
                    "remaining_count_ratio": float(uncertain.float().mean().detach().cpu().item()),
                    "remaining_measure_ratio": (
                        remaining_measure / initial_measure if initial_measure else 0.0
                    ),
                }
            )
            uncertain_candidates = candidate_indices[uncertain]
            uncertain_spans = base_span_indices[uncertain]
            uncertain_left = interval_left[uncertain]
            uncertain_right = interval_right[uncertain]
            if depth == max_depth or uncertain_candidates.numel() == 0:
                unresolved_leaves = (
                    uncertain_candidates,
                    uncertain_left,
                    uncertain_right,
                )
                break
            split = (uncertain_left + uncertain_right) / 2
            candidate_indices = uncertain_candidates.repeat_interleave(2)
            base_span_indices = uncertain_spans.repeat_interleave(2)
            interval_left = torch.stack((uncertain_left, split), dim=-1).reshape(-1)
            interval_right = torch.stack((split, uncertain_right), dim=-1).reshape(-1)

        active_mask = torch.zeros(
            (batch, horizon), dtype=torch.bool, device=q_dense.device
        )
        leaf_candidates, leaf_left, leaf_right = unresolved_leaves
        if leaf_candidates.numel():
            start_indices = torch.searchsorted(phase, leaf_left, right=False).clamp(max=horizon)
            end_indices = torch.searchsorted(phase, leaf_right, right=True).clamp(max=horizon)
            difference = torch.zeros(
                (batch, horizon + 1), dtype=torch.int32, device=q_dense.device
            )
            ones = torch.ones_like(start_indices, dtype=difference.dtype)
            difference.index_put_((leaf_candidates, start_indices), ones, accumulate=True)
            difference.index_put_((leaf_candidates, end_indices), -ones, accumulate=True)
            active_mask = difference[:, :-1].cumsum(dim=1) > 0

        stats = {
            "span_initial_intervals": initial_intervals,
            "span_midpoint_queries": midpoint_queries,
            "span_first_level_certified": first_level_certified,
            "span_first_level_certification_ratio": (
                first_level_certified / initial_intervals if initial_intervals else 1.0
            ),
            "span_unresolved_leaf_count": int(leaf_candidates.numel()),
            "span_time_derivative_bound_s": derivative_time,
            "span_time_midpoint_fk_s": total_fk_time,
            "span_time_midpoint_sdf_s": total_sdf_time,
            "span_time_bound_arithmetic_s": total_bound_time,
            "span_depth_statistics": depth_statistics,
        }
        for item in depth_statistics:
            depth = item["depth"]
            stats[f"span_depth_{depth}_evaluated"] = item["evaluated"]
            stats[f"span_depth_{depth}_certified"] = item["certified"]
            stats[f"span_depth_{depth}_environment_certified"] = item.get(
                "environment_certified", 0
            )
            stats[f"span_depth_{depth}_self_certified"] = item.get(
                "self_certified", 0
            )
            stats[f"span_depth_{depth}_remaining_count_ratio"] = item[
                "remaining_count_ratio"
            ]
            stats[f"span_depth_{depth}_remaining_measure_ratio"] = item[
                "remaining_measure_ratio"
            ]
        return self.select_from_clearances(
            q_dense,
            last_environment,
            last_self,
            risk_mask_override=active_mask,
            temporal_enabled_override=True,
            safe_bucket_override=0,
            span_certificate_statistics=stats,
        )

    def select(self, q_dense, environment_field=None, self_field=None):
        batch, horizon, _ = q_dense.shape
        broad_phase_enabled = bool(self.link_broad_phase_config.get("enabled", False))
        broad_phase_full_scan = broad_phase_enabled and bool(
            self.link_broad_phase_config.get("full_scan", True)
        )
        scan_geometry = self.link_broad_phase_config.get(
            "scan_geometry", "parent_bounds"
        )
        use_parent_bounds = self.use_parent_bounds_scan or (
            broad_phase_enabled and scan_geometry == "parent_bounds"
        )
        cache_fine_scan = (
            broad_phase_enabled
            and broad_phase_full_scan
            and scan_geometry == "fine_spheres"
            and bool(self.link_broad_phase_config.get("reuse_scan_cache", False))
        )
        environment_details = self_details = None
        fine_sphere_scan_cache = None
        if self.config.get("coarse_scan", True) and not broad_phase_full_scan:
            scan_indices = self._risk_scan_indices(horizon, q_dense.device)
            if use_parent_bounds:
                scan_result = self.compute_parent_bound_clearances(
                    q_dense[:, scan_indices], environment_field, self_field
                )
            else:
                scan_result = self.compute_clearances(
                    q_dense[:, scan_indices],
                    environment_field,
                    self_field,
                    return_details=broad_phase_enabled,
                    return_scan_cache=False,
                )
            environment_scan, self_scan = scan_result[:2]
            if broad_phase_enabled:
                environment_details, self_details = scan_result[2:4]

            # Preserve the dense index contract for the selector while evaluating
            # FK/SDF only at the coarse points and interval probes above.
            environment = torch.full(
                (batch, horizon),
                torch.inf,
                dtype=q_dense.dtype,
                device=q_dense.device,
            )
            self_clearance = torch.full_like(environment, torch.inf)
            environment[:, scan_indices] = environment_scan
            self_clearance[:, scan_indices] = self_scan
        else:
            # Reference path used by the ablation benchmark: scan every dense
            # point before selecting an active Jacobian bucket.
            if use_parent_bounds:
                dense_result = self.compute_parent_bound_clearances(
                    q_dense, environment_field, self_field
                )
            else:
                dense_result = self.compute_clearances(
                    q_dense,
                    environment_field,
                    self_field,
                    return_details=broad_phase_enabled,
                    return_scan_cache=cache_fine_scan,
                )
            environment, self_clearance = dense_result[:2]
            if broad_phase_enabled:
                environment_details, self_details = dense_result[2:4]
            if len(dense_result) > 4:
                fine_sphere_scan_cache = dense_result[4]
        masks = (None, None, None)
        if broad_phase_enabled:
            if use_parent_bounds:
                masks = self._link_broad_phase_masks_from_parent_bounds(
                    environment_details, self_details
                )
            else:
                masks = self._link_broad_phase_masks(
                    environment_details, self_details
                )
        return self.select_from_clearances(
            q_dense,
            environment,
            self_clearance,
            parent_link_mask=masks[0],
            environment_sphere_mask=masks[1],
            self_pair_mask=masks[2],
            fine_sphere_scan_cache=fine_sphere_scan_cache,
        )
