"""Frozen, query-indexed Warehouse predictions for offline spacetime planning.

No ROS or drawer dependencies. Obstacles are prescribed kinematic objects:
obstacle/static and obstacle/obstacle intersections never modify their motion.
The saved piecewise-linear forecast is the authoritative collision world.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


MOTION_MODES = (
    "linear", "accelerating", "speed_variation", "curved", "zigzag",
    "shuttle", "orbit", "vertical", "stop_go", "helix",
)
WORKSPACE_LOW = np.array([0.12, -0.78, 0.08])
WORKSPACE_HIGH = np.array([0.88, 0.68, 1.00])
EVENT_FRACTION_RANGE = (0.25, 0.65)
WORLD_SAMPLING_POLICY = {
    "shape": "iid_uniform_sphere_box_capsule",
    "event_fraction": "iid_uniform",
    "event_fraction_range": EVENT_FRACTION_RANGE,
    "motion_mode": "iid_uniform_with_replacement_from_motion_modes",
}


@dataclass
class Forecast:
    times: np.ndarray                 # [K], seconds from execution start
    positions: np.ndarray             # [K,M,3]
    shape: np.ndarray                 # sphere=0, axis-aligned box=1, z-capsule=2
    size: np.ndarray                  # radius; box half extents; capsule r,half-length
    event_times: np.ndarray
    modes: list
    anchors: np.ndarray

    @property
    def horizon(self):
        return float(self.times[-1])

    def metadata(self):
        return {
            "frame": "panda_link0", "time_origin": "execution_start",
            "evaluator": "piecewise_linear_v1", "modes": self.modes,
            "obstacle_static_collision": "ignored", "obstacle_obstacle_collision": "ignored",
            "workspace_low": WORKSPACE_LOW.tolist(), "workspace_high": WORKSPACE_HIGH.tolist(),
            "dynamic_known": True, "object_count": len(self.shape),
        }


def make_forecast(reference_centers, nominal_duration, horizon, rng, *,
                  object_count=3, modes=MOTION_MODES, forecast_dt=0.02):
    """Put each prescribed event on a distal reference-robot sphere in workspace.

    Reference centers are sampled at uniform phase. Event times are inside the
    nominal motion, not relative to planner compute time. The final generator
    checks again against the *accepted* trajectory duration.
    """
    if not 0 < nominal_duration <= horizon or object_count < 0 or forecast_dt <= 0:
        raise ValueError("invalid duration, object_count or forecast_dt")
    if object_count == 0:
        return Forecast(np.array([0., horizon]), np.empty((2, 0, 3)),
                        np.empty(0, dtype=np.int64), np.empty((0, 3)),
                        np.empty(0), [], np.empty((0, 3)))
    centers = np.asarray(reference_centers, dtype=float)
    if centers.ndim != 3 or centers.shape[-1] != 3 or len(centers) < 3:
        raise ValueError("reference_centers must be [phase,sphere,3]")
    if not modes or any(mode not in MOTION_MODES for mode in modes):
        raise ValueError("unknown motion mode")
    # Independent draws avoid coupling shape/event time to count or object
    # index. Events can cluster or occur in any object order; do not sort them.
    fractions = rng.uniform(*EVENT_FRACTION_RANGE, size=object_count)
    sampled_shapes = rng.integers(0, 3, size=object_count)
    sampled_modes = rng.integers(0, len(modes), size=object_count)
    events = fractions * nominal_duration
    grid = np.linspace(0, horizon, int(np.ceil(horizon / forecast_dt)) + 1)
    times = np.unique(np.r_[grid, events])  # exact anchor knots survive serialization
    positions, shapes, sizes, anchors, chosen = [], [], [], [], []
    for index, (fraction, event) in enumerate(zip(fractions, events)):
        phase = min(len(centers) - 1, int(round(fraction * (len(centers) - 1))))
        candidates = centers[phase, max(0, centers.shape[1] // 3):]
        inside = ((candidates >= WORKSPACE_LOW) & (candidates <= WORKSPACE_HIGH)).all(axis=1)
        if not inside.any():
            raise ValueError("reference has no moving-link anchor in warehouse workspace")
        anchor = candidates[inside][int(rng.integers(inside.sum()))].copy()
        direction = rng.normal(size=3)
        direction[2] *= 0.35
        direction /= np.linalg.norm(direction)
        lateral = np.cross(direction, [0., 0., 1.])
        lateral /= np.linalg.norm(lateral)
        vertical = np.cross(direction, lateral)
        mode = modes[int(sampled_modes[index])]
        speed = float(rng.uniform(0.22, 0.42))
        dt = times - event
        along, side, up = speed * dt, np.zeros_like(dt), np.zeros_like(dt)
        if mode == "accelerating":
            # Positive derivative throughout; no accidental reversal far from event.
            along += 0.16 * speed * (np.sqrt(dt * dt + 1) - 1)
        elif mode == "speed_variation":
            along += 0.35 * speed * np.sin(1.6 * dt) / 1.6
        elif mode == "curved":
            side = 0.16 * np.sin(1.4 * dt)
        elif mode == "zigzag":
            side = 0.16 * (2 / np.pi) * np.arcsin(np.sin(2.1 * dt))
        elif mode == "shuttle":
            along = 0.38 * np.sin(speed / 0.38 * dt)
        elif mode == "orbit":
            along = 0.32 * np.sin(speed / 0.32 * dt)
            side = 0.32 * (np.cos(speed / 0.32 * dt) - 1)
        elif mode == "vertical":
            direction = np.array([0., 0., 1.])
            lateral = np.array([1., 0., 0.])
            side = 0.08 * np.sin(1.8 * dt)
        elif mode == "stop_go":
            # Exact prescribed obstacle dwell around its interaction anchor.
            along = speed * np.sign(dt) * np.maximum(np.abs(dt) - 0.28, 0)
        elif mode == "helix":
            side = 0.12 * np.sin(2 * dt)
            up = 0.12 * (np.cos(2 * dt) - 1)
        positions.append(anchor + along[:, None] * direction + side[:, None] * lateral + up[:, None] * vertical)
        shape = int(sampled_shapes[index])
        radius = float(rng.uniform(0.04, 0.075))
        shapes.append(shape)
        sizes.append([radius, 0., 0.] if shape == 0 else
                     ([radius, radius * 1.3, radius * 1.1] if shape == 1 else [radius, 0.09, 0.]))
        anchors.append(anchor)
        chosen.append(mode)
    return Forecast(times, np.stack(positions, axis=1), np.asarray(shapes, dtype=np.int64),
                    np.asarray(sizes), events, chosen, np.asarray(anchors))


class ForecastBatch:
    """Ragged-world forecasts padded once; interpolation stays on the device."""

    def __init__(self, forecasts, device="cpu", dtype=torch.float32):
        if not forecasts:
            raise ValueError("at least one forecast required")
        self.forecasts = forecasts
        # Shared union grid makes heterogeneous event knots query-indexable.
        grid = np.unique(np.concatenate([f.times for f in forecasts]))
        if any(abs(f.horizon - grid[-1]) > 1e-7 for f in forecasts):
            raise ValueError("a batch must have a common forecast horizon")
        max_objects = max(len(f.shape) for f in forecasts)
        positions = np.zeros((len(forecasts), len(grid), max_objects, 3))
        shapes = np.zeros((len(forecasts), max_objects), dtype=np.int64)
        sizes = np.zeros((len(forecasts), max_objects, 3))
        masks = np.zeros((len(forecasts), max_objects), dtype=bool)
        for query, forecast in enumerate(forecasts):
            count = len(forecast.shape)
            for obj in range(count):
                for axis in range(3):
                    positions[query, :, obj, axis] = np.interp(grid, forecast.times, forecast.positions[:, obj, axis])
            shapes[query, :count] = forecast.shape
            sizes[query, :count] = forecast.size
            masks[query, :count] = True
        self.times = torch.as_tensor(grid, dtype=dtype, device=device)
        self.positions = torch.as_tensor(positions, dtype=dtype, device=device)
        self.shapes = torch.as_tensor(shapes, device=device)
        self.sizes = torch.as_tensor(sizes, dtype=dtype, device=device)
        self.masks = torch.as_tensor(masks, device=device)

    def centers(self, times, query_ids):
        times = torch.as_tensor(times, dtype=self.times.dtype, device=self.times.device)
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=self.times.device)
        upper = torch.searchsorted(self.times, times.contiguous()).clamp(1, len(self.times) - 1)
        lower = upper - 1
        weight = ((times - self.times[lower]) / (self.times[upper] - self.times[lower])).clamp(0, 1)
        return torch.lerp(self.positions[query_ids, lower], self.positions[query_ids, upper], weight[:, None, None])

    def clearance(self, robot_centers, robot_radii, times, query_ids):
        obstacle_centers = self.centers(times, query_ids)
        invalid_time = ~torch.isfinite(times) | (times < self.times[0] - 1e-6) | (times > self.times[-1] + 1e-6)
        if self.shapes.shape[1] == 0:
            result = robot_centers.new_full((len(times),), torch.inf)
            return result.masked_fill(invalid_time, -torch.inf)
        delta = robot_centers[:, :, None] - obstacle_centers[:, None]
        sizes = self.sizes[query_ids, None]
        sphere = torch.linalg.vector_norm(delta, dim=-1) - sizes[..., 0]
        box_delta = delta.abs() - sizes
        box = torch.linalg.vector_norm(box_delta.clamp_min(0), dim=-1) + box_delta.amax(dim=-1).clamp_max(0)
        capsule_delta = delta.clone()
        capsule_delta[..., 2] -= delta[..., 2].clamp(-sizes[..., 1], sizes[..., 1])
        capsule = torch.linalg.vector_norm(capsule_delta, dim=-1) - sizes[..., 0]
        sdf = torch.where(self.shapes[query_ids, None] == 0, sphere,
                          torch.where(self.shapes[query_ids, None] == 1, box, capsule))
        sdf = (sdf - robot_radii[None, :, None]).masked_fill(~self.masks[query_ids, None], torch.inf)
        result = sdf.amin(dim=(1, 2))
        return result.masked_fill(invalid_time, -torch.inf)


class WarehouseCollision:
    """One persistent Panda/static Warehouse; dynamic worlds indexed per sample."""

    def __init__(self, device="cpu", *, margin=0.005, chunk_size=512):
        from torch_robotics.robots import RobotPanda
        from torch_robotics.environments.env_warehouse import EnvWarehouse
        from torch_robotics.torch_planning_objectives.fields.distance_fields import CollisionObjectDistanceField

        if margin < 0 or chunk_size < 1:
            raise ValueError("invalid collision margin or chunk size")
        self.tensor_args = {"device": torch.device(device), "dtype": torch.float32}
        self.robot = RobotPanda(tensor_args=self.tensor_args)
        self.environment = EnvWarehouse(precompute_sdf_obj_fixed=False, precompute_sdf_obj_extra=False,
                                        tensor_args=self.tensor_args)
        self.field = CollisionObjectDistanceField(
            self.robot, df_obj_list_fn=self.environment.get_df_obj_list,
            link_margins_for_object_collision_checking_tensor=self.robot.link_collision_spheres_radii,
            cutoff_margin=margin, tensor_args=self.tensor_args)
        self.margin, self.chunk_size = float(margin), int(chunk_size)
        self.worlds = None

    def set_worlds(self, forecasts):
        self.worlds = ForecastBatch(forecasts, **self.tensor_args)

    @torch.no_grad()
    def centers(self, q):
        q = torch.as_tensor(q, **self.tensor_args)
        parent = torch.stack(self.robot.fk_collision_sphere_parent_links(q), dim=1)
        selected = parent[:, self.robot.collision_sphere_parent_indices]
        return (torch.einsum("bsij,sj->bsi", selected[..., :3, :3], self.robot.collision_sphere_local_positions)
                + selected[..., :3, 3])

    @torch.no_grad()
    def clearances(self, q, times=None, query_ids=None):
        q = torch.as_tensor(q, **self.tensor_args).reshape(-1, 7)
        if times is not None:
            times = torch.as_tensor(times, **self.tensor_args).reshape(-1)
            query_ids = torch.as_tensor(query_ids, device=q.device, dtype=torch.long).reshape(-1)
            if len(times) != len(q) or len(query_ids) != len(q) or self.worlds is None:
                raise ValueError("dynamic samples need matching time/query arrays and worlds")
        outputs = []
        for begin in range(0, len(q), self.chunk_size):
            part = q[begin:begin + self.chunk_size]
            centers = self.centers(part)
            static = (self.field.object_signed_distances(centers) - self.robot.link_collision_spheres_radii).amin(dim=(1, 2))
            self_distance = self.robot.df_collision_self.compute_embodiment_signed_distances(part, centers).amin(dim=1)
            valid = (torch.isfinite(part).all(dim=1) & (part >= self.robot.q_pos_min).all(dim=1)
                     & (part <= self.robot.q_pos_max).all(dim=1))
            dynamic = torch.full_like(static, torch.inf)
            if times is not None:
                dynamic = self.worlds.clearance(centers, self.robot.link_collision_spheres_radii,
                                               times[begin:begin + len(part)], query_ids[begin:begin + len(part)])
            values = torch.stack((static, self_distance, dynamic), dim=1)
            outputs.append(values.masked_fill(~valid[:, None], -torch.inf))
        return torch.cat(outputs) if outputs else torch.empty((0, 3), **self.tensor_args)

    def __call__(self, q, times, query_ids):
        return (self.clearances(q, times, query_ids) <= self.margin).any(dim=1)

    def static_collision(self, q):
        return (self.clearances(q)[:, :2] <= self.margin).any(dim=1)


def interaction_audit(forecast, duration):
    """All objects must visit the warehouse arm workspace during actual execution."""
    visits = []
    for obj in range(len(forecast.shape)):
        positions = forecast.positions[:, obj]
        inside = ((positions >= WORKSPACE_LOW) & (positions <= WORKSPACE_HIGH)).all(axis=1)
        during = (forecast.times > 0.05 * duration) & (forecast.times < 0.95 * duration)
        event = float(forecast.event_times[obj])
        visits.append(bool((inside & during).any() and 0.05 * duration < event < 0.95 * duration))
    return {"all_objects_interact_during_execution": all(visits), "workspace_visits": visits}
