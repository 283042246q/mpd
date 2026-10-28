"""Known piecewise-linear obstacle motion, without prediction or inflation.

The serialized knots define the world. Both fixed-time and variable-time
planners query these same knots, including differentiable physical-time queries.
The snapshot implementation remains the default in the online runtimes.
"""
from __future__ import annotations

import numpy as np
import torch

from mpd.datasets.warehouse_joint_dataset import world_features
from mpd.inference.dynamic_collision import DynamicWorldError, FixedCapacityDynamicWorld


class PrescribedDynamicWorld(FixedCapacityDynamicWorld):
    motion_model = "prescribed_piecewise_linear"

    def load_forecast(self, times, positions, shapes, sizes, *, world_version=None):
        times = np.asarray(times, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)
        shapes = np.asarray(shapes, dtype=np.int64)
        sizes = np.asarray(sizes, dtype=np.float64)
        if (times.ndim != 1 or len(times) < 2 or times[0] != 0
                or not np.isfinite(times).all() or not (np.diff(times) > 0).all()
                or positions.shape != (len(times), len(shapes), 3)
                or sizes.shape != (len(shapes), 3) or len(shapes) > self.max_objects
                or not np.isfinite(positions).all() or not np.isfinite(sizes).all()
                or not np.isin(shapes, [0, 1, 2]).all()):
            raise ValueError("invalid prescribed obstacle trajectories")
        objects = []
        for i, (shape, size) in enumerate(zip(shapes, sizes)):
            geometry = ({"type": "sphere", "radius": float(size[0])} if shape == 0 else
                        {"type": "box", "size_xyz": (2 * size).tolist()} if shape == 1 else
                        {"type": "capsule", "radius": float(size[0]), "length": float(2 * size[1])})
            objects.append(dict(id=f"obstacle-{i}", local_sdf=geometry,
                pose=dict(position=positions[0, i].tolist(), orientation_xyzw=[0., 0., 0., 1.]),
                linear_velocity=[0., 0., 0.],
                inflation=dict(mode="linear", base_m=0., horizon_rate_m_s=0.)))
        version = self.world_version + 1 if world_version is None else world_version
        # Validate the complete snapshot before publishing the knot tensors.
        self.update(dict(world_version=version, frame_id="fr3_link0", stamp_unix_ns=1,
                         valid_until_unix_ns=1 + int(round(times[-1] * 1e9)), objects=objects))
        self.times = torch.as_tensor(times, dtype=torch.float64, device=self.position.device)
        self.positions = torch.as_tensor(positions, dtype=torch.float64, device=self.position.device)
        self.shapes = torch.as_tensor(shapes, dtype=torch.long, device=self.position.device)
        self.sizes = torch.as_tensor(sizes, dtype=torch.float64, device=self.position.device)
        self.set_plan_start(1, world_version=version)
        return version

    def features(self):
        return world_features(self.times.cpu().numpy(), self.positions.cpu().numpy(),
                              self.shapes.cpu().numpy(), self.sizes.cpu().numpy())

    def centers(self, times):
        grid = self.times.to(times)
        upper = torch.searchsorted(grid, times.contiguous(), right=True).clamp(1, len(grid) - 1)
        lower = upper - 1
        weight = (times - grid[lower]) / (grid[upper] - grid[lower])
        positions = self.positions.to(times)
        return positions[lower] + weight[..., None, None] * (positions[upper] - positions[lower])

    def _query_table(self, points, trajectory_times, *, validate_times=True):
        # Runtime construction warms an empty world before a forecast is loaded.
        if not hasattr(self, "times"):
            return super()._query_table(points, trajectory_times, validate_times=validate_times)
        if trajectory_times is None:
            trajectory_times = torch.linspace(0., self.trajectory_duration_s, points.shape[1],
                dtype=points.dtype, device=points.device)[None].expand(points.shape[0], -1)
        if trajectory_times.shape != points.shape[:2]:
            raise ValueError("trajectory_times must match [batch,time] of query points")
        offset = (self.plan_start_unix_ns - self.stamp_unix_ns) * 1e-9
        relative = trajectory_times.to(points) + offset
        # Never silently extrapolate, even when a caller skips monotonicity checks.
        if (not torch.isfinite(relative).all() or (relative < -1e-6).any()
                or (relative > self.times[-1] + 1e-6).any()):
            raise DynamicWorldError("query outside prescribed trajectory coverage")
        centers = self.centers(relative)
        capacity = self._active_capacity()
        if capacity > self.active_count:
            centers = torch.cat((centers, centers.new_zeros(*centers.shape[:2],
                                 capacity - self.active_count, 3)), dim=2)
        return dict(relative_times=relative, centers=centers,
                    inflation=points.new_zeros(*relative.shape, capacity),
                    rotation=self.rotation[:capacity].to(points),
                    parameters=self.parameters[:capacity].to(points), capacity=capacity)
