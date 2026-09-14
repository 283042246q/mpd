"""OMPL-free multi-query GPU planning and Torch collision auditing.

This module is intentionally safe to import in a CUDA-only spawned process:
it does not import PyBullet, pb_ompl, or the legacy CPU generator.  A separate
native worker remains responsible for endpoint proposal and the independent
PyBullet mesh gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy import interpolate
from scipy.interpolate import BSpline
import torch

from scripts.generate_data.gpu_batch_rrt import (
    GpuBatchRRTResult,
    GpuMultiQueryRRTConnect,
)
from torch_robotics.environments.env_warehouse_marvin_bimanual import (
    EnvWarehouseMarvinBimanual,
)
from torch_robotics.robots.robot_marvin_bimanual import RobotMarvinBimanual
from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    CollisionObjectDistanceField,
)


MODE_ACTIVE_INDICES = {
    "dual_independent": np.arange(14, dtype=np.int64),
    "left_only": np.arange(7, dtype=np.int64),
    "right_only": np.arange(7, 14, dtype=np.int64),
}


@dataclass
class GpuTrajectoryPlan:
    path: np.ndarray | None
    spline: tuple[np.ndarray, np.ndarray, int] | None
    rrt: GpuBatchRRTResult
    rejection_reason: str | None
    batch_seconds: float


def densify_path(path, max_joint_step):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or len(path) < 1:
        raise ValueError("path must have shape [waypoints, dof]")
    if max_joint_step <= 0:
        raise ValueError("max_joint_step must be positive")
    samples = [path[:1]]
    for start, goal in zip(path[:-1], path[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(goal - start)) / max_joint_step)))
        samples.append(
            start + np.linspace(0.0, 1.0, count + 1)[1:, None] * (goal - start)
        )
    return np.concatenate(samples)


def resample_path(path, waypoint_count):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or len(path) < 2 or waypoint_count < 2:
        raise ValueError("path and waypoint_count must describe at least two points")
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    distance, unique = np.unique(distance, return_index=True)
    path = path[unique]
    if len(distance) < 2:
        raise ValueError("path has zero length")
    sample = np.linspace(0.0, distance[-1], int(waypoint_count))
    return np.stack(
        [np.interp(sample, distance, path[:, joint]) for joint in range(path.shape[1])],
        axis=1,
    )


def fit_marvin_spline(path, degree, control_points, validation_points):
    """Fit the same 14-D clamped spline contract without importing pb_ompl."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 14:
        raise ValueError(f"Marvin path must have shape [N, 14], got {path.shape}")
    if control_points <= degree or validation_points < 2:
        raise ValueError("invalid spline dimensions")
    knot_count = degree + control_points
    knots = np.pad(
        np.linspace(0.0, 1.0, knot_count + 1 - 2 * degree), degree, "edge"
    )
    parameter = np.linspace(0.0, 1.0, len(path))
    interior = knots[degree + 1 : -(degree + 1)]
    coefficients = []
    for dimension in range(path.shape[1]):
        _, coefficient, _ = interpolate.splrep(
            parameter,
            path[:, dimension],
            k=degree,
            t=interior,
            task=-1,
            quiet=True,
        )
        coefficients.append(np.asarray(coefficient[:control_points]))
    coefficients = np.stack(coefficients, axis=0)
    coefficients[:, 0] = path[0]
    coefficients[:, -1] = path[-1]
    coefficients[:, 1] = coefficients[:, 0]
    coefficients[:, -2] = coefficients[:, -1]
    coefficients[:, 2] = coefficients[:, 0]
    coefficients[:, -3] = coefficients[:, -1]
    evaluated = BSpline(knots, coefficients.T, degree)(
        np.linspace(0.0, 1.0, int(validation_points))
    )
    return (knots, coefficients.copy(), int(degree)), evaluated


def batched_paths_collision_free(paths, collision_fn, max_joint_step):
    """Audit variable-length paths with one compact collision invocation."""
    if not paths:
        return []
    dense = [densify_path(path, max_joint_step) for path in paths]
    sizes = [len(states) for states in dense]
    collision = collision_fn(np.concatenate(dense)).detach().cpu().numpy().astype(bool)
    result = []
    offset = 0
    for size in sizes:
        result.append(not bool(collision[offset : offset + size].any()))
        offset += size
    return result


def shortcut_paths(paths, collision_fn, max_joint_step, attempts, seeds):
    """Random shortcut rounds, batching one candidate edge from every path."""
    if len(paths) != len(seeds):
        raise ValueError("shortcut seeds must match path count")
    result = [np.asarray(path, dtype=float).copy() for path in paths]
    generators = [np.random.default_rng(int(seed)) for seed in seeds]
    for _ in range(int(attempts)):
        proposals = []
        edges = []
        for query, (path, generator) in enumerate(zip(result, generators)):
            if len(path) < 3:
                continue
            start = int(generator.integers(0, len(path) - 2))
            goal = int(generator.integers(start + 2, len(path)))
            proposals.append((query, start, goal))
            edges.append(np.stack((path[start], path[goal])))
        if not edges:
            break
        valid = batched_paths_collision_free(edges, collision_fn, max_joint_step)
        for (query, start, goal), accepted in zip(proposals, valid):
            if accepted:
                result[query] = np.concatenate(
                    (result[query][: start + 1], result[query][goal:]), axis=0
                )
    return result


class MarvinGpuPlanningBackend:
    """Persistent CUDA owner for homogeneous Marvin planning batches."""

    def __init__(self, config):
        self.config = dict(config)
        device = torch.device(self.config.get("gpu_device", "cuda:0"))
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("MarvinGpuPlanningBackend requires an available CUDA device")
        tensor_args = {"device": device, "dtype": torch.float32}
        self.robot = RobotMarvinBimanual(tensor_args=tensor_args)
        environment = EnvWarehouseMarvinBimanual(
            precompute_sdf_obj_fixed=False,
            precompute_sdf_obj_extra=False,
            tensor_args=tensor_args,
        )
        self.object_field = CollisionObjectDistanceField(
            self.robot,
            df_obj_list_fn=environment.get_df_obj_list,
            link_margins_for_object_collision_checking_tensor=(
                self.robot.link_collision_spheres_radii
            ),
            cutoff_margin=float(self.config.get("min_distance_robot_env", 0.02)),
            tensor_args=tensor_args,
        )

    def collision_positions(self, q):
        parent_poses = torch.stack(
            self.robot.fk_collision_sphere_parent_links(q), dim=1
        )
        selected = parent_poses[:, self.robot.collision_sphere_parent_indices]
        return (
            torch.einsum(
                "bsij,sj->bsi",
                selected[..., :3, :3],
                self.robot.collision_sphere_local_positions,
            )
            + selected[..., :3, 3]
        )

    @torch.no_grad()
    def collision_mask(self, states):
        states = torch.as_tensor(states, **self.robot.tensor_args)
        if states.ndim == 1:
            states = states[None]
        result = torch.empty(len(states), dtype=torch.bool, device=states.device)
        batch_size = int(self.config.get("gpu_collision_batch_size", 256))
        for begin in range(0, len(states), batch_size):
            chunk = states[begin : begin + batch_size]
            out_of_bounds = (
                (chunk < self.robot.q_pos_min) | (chunk > self.robot.q_pos_max)
            ).any(dim=1)
            positions = self.collision_positions(chunk)
            self_collision = self.robot.df_collision_self.compute_cost(
                chunk, positions, field_type="occupancy"
            ).reshape(-1)
            object_collision = self.object_field.compute_cost(
                chunk, positions, field_type="occupancy"
            ).reshape(-1)
            result[begin : begin + len(chunk)] = (
                out_of_bounds | self_collision | object_collision
            )
        return result

    def _restore_contract(self, path, q_start, q_goal, active_indices):
        path = np.asarray(path, dtype=float).copy()
        inactive = np.setdiff1d(
            np.arange(14), active_indices, assume_unique=True
        )
        path[:, inactive] = np.asarray(q_start)[inactive]
        path[0] = q_start
        path[-1, active_indices] = np.asarray(q_goal)[active_indices]
        return path

    def plan_mode_batch(self, mode, q_starts, q_goals, query_seeds):
        if mode not in MODE_ACTIVE_INDICES:
            raise ValueError(f"unsupported task mode: {mode}")
        q_starts = np.asarray(q_starts, dtype=float)
        q_goals = np.asarray(q_goals, dtype=float)
        if q_starts.ndim != 2 or q_starts.shape != q_goals.shape or q_starts.shape[1] != 14:
            raise ValueError("Marvin endpoints must have shape [queries, 14]")
        if len(query_seeds) != len(q_starts):
            raise ValueError("query seeds must match endpoint count")
        active_indices = MODE_ACTIVE_INDICES[mode]
        planner = GpuMultiQueryRRTConnect(
            self.collision_mask,
            self.robot.q_pos_min[active_indices],
            self.robot.q_pos_max[active_indices],
            batch_size=int(self.config.get("gpu_rrt_edges_per_query", 16)),
            max_iterations=int(self.config.get("gpu_rrt_max_iterations", 4096)),
            extension_range=float(self.config.get("planner_range", 0.35)),
            collision_step=float(
                self.config.get(
                    "gpu_rrt_collision_max_joint_step",
                    self.config.get("collision_max_joint_step", 0.025),
                )
            ),
            goal_bias=float(self.config.get("gpu_rrt_goal_bias", 0.125)),
            seed=int(query_seeds[0]) if query_seeds else 0,
        )
        torch.cuda.synchronize(self.robot.q_pos_min.device)
        started = time.perf_counter()
        rrt_results = planner.plan_batch(
            q_starts,
            q_goals,
            active_indices,
            float(self.config.get("planner_allowed_time", 10.0)),
            query_seeds=query_seeds,
        )
        raw = []
        raw_queries = []
        for query, result in enumerate(rrt_results):
            if result.path is not None:
                raw_queries.append(query)
                raw.append(
                    self._restore_contract(
                        result.path, q_starts[query], q_goals[query], active_indices
                    )
                )
        shortcut = shortcut_paths(
            raw,
            self.collision_mask,
            float(self.config.get("collision_max_joint_step", 0.025)),
            int(self.config.get("gpu_shortcut_attempts", 64)),
            [int(query_seeds[query]) + 1_000_003 for query in raw_queries],
        )
        resampled = []
        resampled_queries = []
        for query, path in zip(raw_queries, shortcut):
            try:
                resampled.append(
                    self._restore_contract(
                        resample_path(
                            path, int(self.config.get("interpolate_num", 128))
                        ),
                        q_starts[query],
                        q_goals[query],
                        active_indices,
                    )
                )
                resampled_queries.append(query)
            except ValueError:
                pass
        raw_valid = batched_paths_collision_free(
            resampled,
            self.collision_mask,
            float(self.config.get("collision_max_joint_step", 0.025)),
        )

        paths_by_query = {
            query: path
            for query, path, valid in zip(resampled_queries, resampled, raw_valid)
            if valid
        }
        spline_by_query = {}
        spline_paths = []
        spline_queries = []
        for query, path in paths_by_query.items():
            try:
                spline, evaluated = fit_marvin_spline(
                    path,
                    int(self.config.get("bspline_degree", 5)),
                    int(self.config.get("bspline_num_control_points", 22)),
                    int(self.config.get("spline_validation_points", 512)),
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if not np.allclose(evaluated[[0, -1]], path[[0, -1]], atol=1e-5):
                continue
            spline_by_query[query] = spline
            spline_queries.append(query)
            spline_paths.append(evaluated)
        spline_valid = batched_paths_collision_free(
            spline_paths,
            self.collision_mask,
            float(self.config.get("collision_max_joint_step", 0.025)),
        )
        valid_spline_queries = {
            query for query, valid in zip(spline_queries, spline_valid) if valid
        }
        torch.cuda.synchronize(self.robot.q_pos_min.device)
        elapsed = time.perf_counter() - started

        plans = []
        for query, rrt in enumerate(rrt_results):
            if rrt.path is None:
                reason = "rrt"
            elif query not in paths_by_query:
                reason = "raw_torch_audit"
            elif query not in spline_by_query:
                reason = "spline_fit"
            elif query not in valid_spline_queries:
                reason = "spline_torch_audit"
            else:
                reason = None
            plans.append(
                GpuTrajectoryPlan(
                    paths_by_query.get(query) if reason is None else None,
                    spline_by_query.get(query) if reason is None else None,
                    rrt,
                    reason,
                    elapsed,
                )
            )
        return plans
