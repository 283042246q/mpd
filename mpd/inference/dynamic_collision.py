"""Fixed-capacity dynamic collision fields for Phase 4 and Phase 5.

The static environment field is kept intact.  A small number of known dynamic
objects are represented by analytic local SDFs whose poses are predicted with
constant linear velocity and fixed orientation.  All persistent tensors have a
fixed shape so a world update never rebuilds the static SDF or planning task.
"""

from __future__ import annotations

import math
from typing import Any

import torch


class DynamicWorldError(ValueError):
    """The supplied dynamic-world snapshot violates the Phase-4 contract."""


_SHAPE_CODES = {"sphere": 0, "box": 1, "capsule": 2}
_INFLATION_CODES = {"linear": 0, "covariance": 1}


def _finite_numbers(value: Any, size: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise DynamicWorldError(f"{name} must contain {size} numbers")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise DynamicWorldError(f"{name} contains NaN or Inf")
    return result


def _rotation_xyzw(quaternion: list[float]) -> list[list[float]]:
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        raise DynamicWorldError("orientation_xyzw has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


class FixedCapacityDynamicWorld:
    """Device-resident dynamic-object snapshot with in-place updates."""

    def __init__(
        self,
        max_objects: int,
        *,
        trajectory_duration_s: float,
        tensor_args: dict[str, Any],
        covariance_sigma: float = 3.0,
        process_acceleration_std_m_s2: float = 0.01,
        capacity_buckets_enabled: bool = False,
        shape_grouping_enabled: bool = False,
        time_table_cache_enabled: bool = False,
        fused_reduction_enabled: bool = False,
    ) -> None:
        if max_objects < 1:
            raise ValueError("max_objects must be positive")
        if trajectory_duration_s <= 0.0:
            raise ValueError("trajectory_duration_s must be positive")
        self.max_objects = int(max_objects)
        self.trajectory_duration_s = float(trajectory_duration_s)
        self.tensor_args = dict(tensor_args)
        self.covariance_sigma = float(covariance_sigma)
        self.process_variance = float(process_acceleration_std_m_s2) ** 2
        self.capacity_buckets_enabled = bool(capacity_buckets_enabled)
        self.shape_grouping_enabled = bool(shape_grouping_enabled)
        self.time_table_cache_enabled = bool(time_table_cache_enabled)
        self.fused_reduction_enabled = bool(fused_reduction_enabled)
        self.capacity_buckets = tuple(
            size for size in (1, 2, 4, 8, 16, 32, 64) if size < self.max_objects
        ) + (self.max_objects,)

        def zeros(*shape, dtype=None):
            args = dict(self.tensor_args)
            if dtype is not None:
                args["dtype"] = dtype
            return torch.zeros(*shape, **args)

        self.active = zeros(self.max_objects, dtype=torch.bool)
        self.shape_code = zeros(self.max_objects, dtype=torch.long)
        self.position = zeros(self.max_objects, 3)
        self.velocity = zeros(self.max_objects, 3)
        self.rotation = zeros(self.max_objects, 3, 3)
        self.parameters = zeros(self.max_objects, 3)
        self.covariance = zeros(self.max_objects, 6, 6)
        self.inflation_code = zeros(self.max_objects, dtype=torch.long)
        self.base_inflation = zeros(self.max_objects)
        self.horizon_inflation_rate = zeros(self.max_objects)
        self._staging = {
            "active": zeros(self.max_objects, dtype=torch.bool),
            "shape_code": zeros(self.max_objects, dtype=torch.long),
            "position": zeros(self.max_objects, 3),
            "velocity": zeros(self.max_objects, 3),
            "rotation": zeros(self.max_objects, 3, 3),
            "parameters": zeros(self.max_objects, 3),
            "covariance": zeros(self.max_objects, 6, 6),
            "inflation_code": zeros(self.max_objects, dtype=torch.long),
            "base_inflation": zeros(self.max_objects),
            "horizon_inflation_rate": zeros(self.max_objects),
        }
        self.object_ids = tuple("" for _ in range(self.max_objects))
        self.world_version = 0
        self.frame_id = ""
        self.stamp_unix_ns = 0
        self.valid_until_unix_ns = 0
        self.plan_start_unix_ns = 0
        self.active_count = 0
        self.sphere_indices = zeros(0, dtype=torch.long)
        self.box_indices = zeros(0, dtype=torch.long)
        self.capsule_indices = zeros(0, dtype=torch.long)
        self.linear_inflation_indices = zeros(0, dtype=torch.long)
        self.covariance_inflation_indices = zeros(0, dtype=torch.long)
        self._time_table_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def update(self, snapshot: dict[str, Any]) -> int:
        if not isinstance(snapshot, dict):
            raise DynamicWorldError("world snapshot must be an object")
        version = snapshot.get("world_version")
        if isinstance(version, bool) or not isinstance(version, int) or version <= self.world_version:
            raise DynamicWorldError("world_version must increase monotonically")
        frame_id = snapshot.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise DynamicWorldError("frame_id must be a non-empty string")
        stamp = snapshot.get("stamp_unix_ns")
        valid_until = snapshot.get("valid_until_unix_ns")
        if (
            isinstance(stamp, bool)
            or not isinstance(stamp, int)
            or stamp < 0
            or isinstance(valid_until, bool)
            or not isinstance(valid_until, int)
            or valid_until <= stamp
        ):
            raise DynamicWorldError("world timestamps are invalid")
        objects = snapshot.get("objects")
        if not isinstance(objects, list) or len(objects) > self.max_objects:
            raise DynamicWorldError(f"objects must be a list with at most {self.max_objects} entries")

        for tensor in self._staging.values():
            tensor.zero_()
        active = self._staging["active"]
        shape_code = self._staging["shape_code"]
        position = self._staging["position"]
        velocity = self._staging["velocity"]
        rotation = self._staging["rotation"]
        parameters = self._staging["parameters"]
        covariance = self._staging["covariance"]
        inflation_code = self._staging["inflation_code"]
        base_inflation = self._staging["base_inflation"]
        horizon_rate = self._staging["horizon_inflation_rate"]
        object_ids: list[str] = []

        for index, item in enumerate(objects):
            if not isinstance(item, dict):
                raise DynamicWorldError(f"objects[{index}] must be an object")
            object_id = item.get("id")
            if not isinstance(object_id, str) or not object_id or object_id in object_ids:
                raise DynamicWorldError("dynamic object ids must be unique non-empty strings")
            shape = item.get("local_sdf")
            if not isinstance(shape, dict) or shape.get("type") not in _SHAPE_CODES:
                raise DynamicWorldError(f"objects[{index}].local_sdf.type is unsupported")
            shape_name = str(shape["type"])
            if shape_name == "sphere":
                radius = float(shape.get("radius", 0.0))
                if not math.isfinite(radius) or radius <= 0.0:
                    raise DynamicWorldError("sphere radius must be positive")
                shape_parameters = [radius, 0.0, 0.0]
            elif shape_name == "box":
                size = _finite_numbers(shape.get("size_xyz"), 3, "box size_xyz")
                if any(value <= 0.0 for value in size):
                    raise DynamicWorldError("box size_xyz must be positive")
                shape_parameters = [value * 0.5 for value in size]
            else:
                radius = float(shape.get("radius", 0.0))
                length = float(shape.get("length", 0.0))
                if not all(math.isfinite(value) and value > 0.0 for value in (radius, length)):
                    raise DynamicWorldError("capsule radius and length must be positive")
                shape_parameters = [radius, length * 0.5, 0.0]

            pose = item.get("pose")
            if not isinstance(pose, dict):
                raise DynamicWorldError("dynamic object pose is required")
            p = _finite_numbers(pose.get("position"), 3, "pose.position")
            quaternion = _finite_numbers(
                pose.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0]),
                4,
                "pose.orientation_xyzw",
            )
            v = _finite_numbers(item.get("linear_velocity", [0.0, 0.0, 0.0]), 3, "linear_velocity")
            flat_covariance = item.get("covariance_6x6", [0.0] * 36)
            flat_covariance = _finite_numbers(flat_covariance, 36, "covariance_6x6")
            inflation = item.get("inflation", {})
            if not isinstance(inflation, dict):
                raise DynamicWorldError("inflation must be an object")
            mode = inflation.get("mode", "linear")
            if mode not in _INFLATION_CODES:
                raise DynamicWorldError("inflation.mode must be linear or covariance")
            base = float(inflation.get("base_m", 0.0))
            rate = float(inflation.get("horizon_rate_m_s", 0.0))
            if not all(math.isfinite(value) and value >= 0.0 for value in (base, rate)):
                raise DynamicWorldError("inflation values must be finite and non-negative")

            active[index] = True
            shape_code[index] = _SHAPE_CODES[shape_name]
            position[index].copy_(torch.as_tensor(p, **self.tensor_args))
            velocity[index].copy_(torch.as_tensor(v, **self.tensor_args))
            rotation[index].copy_(torch.as_tensor(_rotation_xyzw(quaternion), **self.tensor_args))
            parameters[index].copy_(torch.as_tensor(shape_parameters, **self.tensor_args))
            covariance[index].copy_(torch.as_tensor(flat_covariance, **self.tensor_args).reshape(6, 6))
            inflation_code[index] = _INFLATION_CODES[str(mode)]
            base_inflation[index] = base
            horizon_rate[index] = rate
            object_ids.append(object_id)

        # Publish only after the complete snapshot has been validated.
        self.active.copy_(active)
        self.shape_code.copy_(shape_code)
        self.position.copy_(position)
        self.velocity.copy_(velocity)
        self.rotation.copy_(rotation)
        self.parameters.copy_(parameters)
        self.covariance.copy_(covariance)
        self.inflation_code.copy_(inflation_code)
        self.base_inflation.copy_(base_inflation)
        self.horizon_inflation_rate.copy_(horizon_rate)
        self.object_ids = tuple(object_ids + [""] * (self.max_objects - len(object_ids)))
        self.frame_id = frame_id
        self.stamp_unix_ns = stamp
        self.valid_until_unix_ns = valid_until
        self.world_version = version
        self.active_count = len(objects)
        active_slice = slice(0, self.active_count)
        active_shape_codes = self.shape_code[active_slice]
        active_inflation_codes = self.inflation_code[active_slice]
        self.sphere_indices = torch.nonzero(active_shape_codes == 0, as_tuple=False).flatten()
        self.box_indices = torch.nonzero(active_shape_codes == 1, as_tuple=False).flatten()
        self.capsule_indices = torch.nonzero(active_shape_codes == 2, as_tuple=False).flatten()
        self.linear_inflation_indices = torch.nonzero(
            active_inflation_codes == 0, as_tuple=False
        ).flatten()
        self.covariance_inflation_indices = torch.nonzero(
            active_inflation_codes == 1, as_tuple=False
        ).flatten()
        self._time_table_cache.clear()
        return version

    def set_plan_start(self, plan_start_unix_ns: int, *, world_version: int) -> None:
        if world_version != self.world_version:
            raise DynamicWorldError(f"requested world_version {world_version} != loaded {self.world_version}")
        if not isinstance(plan_start_unix_ns, int) or plan_start_unix_ns < self.stamp_unix_ns:
            raise DynamicWorldError("plan_start_unix_ns predates the world snapshot")
        horizon_end = plan_start_unix_ns + int(self.trajectory_duration_s * 1e9)
        if horizon_end > self.valid_until_unix_ns:
            raise DynamicWorldError("trajectory exceeds dynamic-world prediction validity")
        self.plan_start_unix_ns = plan_start_unix_ns
        self._time_table_cache.clear()

    def _active_capacity(self) -> int:
        if not self.capacity_buckets_enabled:
            return self.max_objects
        if self.active_count == 0:
            return 0
        return next(size for size in self.capacity_buckets if size >= self.active_count)

    def _relative_times(self, horizon: int, dtype, device) -> torch.Tensor:
        if self.plan_start_unix_ns <= 0:
            raise DynamicWorldError("dynamic plan context has not been set")
        plan_offset = (self.plan_start_unix_ns - self.stamp_unix_ns) * 1e-9
        trajectory_times = torch.linspace(0.0, self.trajectory_duration_s, horizon, dtype=dtype, device=device)
        return trajectory_times + plan_offset

    def _inflation(self, relative_times: torch.Tensor, capacity: int) -> torch.Tensor:
        # [...,H,M], where covariance is propagated by the constant-velocity model.
        dt = relative_times[..., None]
        base = self.base_inflation[:capacity]
        rate = self.horizon_inflation_rate[:capacity]
        inflation = torch.zeros(
            (*relative_times.shape, capacity),
            dtype=relative_times.dtype,
            device=relative_times.device,
        )
        linear_indices = self.linear_inflation_indices
        if linear_indices.numel():
            linear = base.index_select(0, linear_indices) + rate.index_select(
                0, linear_indices
            ) * dt
            inflation = inflation.index_copy(-1, linear_indices, linear)
        covariance_indices = self.covariance_inflation_indices
        if not covariance_indices.numel():
            return inflation
        selected_covariance = self.covariance.index_select(0, covariance_indices)
        selected_base = base.index_select(0, covariance_indices)
        p_pp = selected_covariance[:, :3, :3]
        p_pv = selected_covariance[:, :3, 3:]
        p_vp = selected_covariance[:, 3:, :3]
        p_vv = selected_covariance[:, 3:, 3:]
        propagated = (
            p_pp
            + dt[..., None, None] * (p_pv + p_vp)
            + dt[..., None, None].square() * p_vv
        )
        process = self.process_variance * dt.pow(3) / 3.0
        propagated = propagated + process[..., None, None] * torch.eye(
            3, dtype=relative_times.dtype, device=relative_times.device
        )
        # eigvalsh is deterministic and the object count is deliberately small.
        sigma = torch.linalg.eigvalsh(propagated).amax(dim=-1).clamp_min(0.0).sqrt()
        covariance = selected_base + self.covariance_sigma * sigma
        return inflation.index_copy(-1, covariance_indices, covariance)

    def _time_table(self, horizon: int, dtype, device) -> dict[str, Any]:
        capacity = self._active_capacity()
        key = (
            self.world_version,
            self.plan_start_unix_ns,
            horizon,
            dtype,
            device.type,
            device.index,
            capacity,
            self.shape_grouping_enabled,
        )
        if self.time_table_cache_enabled and key in self._time_table_cache:
            return self._time_table_cache[key]
        relative_times = self._relative_times(horizon, dtype, device)
        centers = self.position[None, :capacity, :] + relative_times[:, None, None] * self.velocity[
            None, :capacity, :
        ]
        table = {
            "relative_times": relative_times,
            "centers": centers,
            "inflation": self._inflation(relative_times, capacity),
            "rotation": self.rotation[:capacity],
            "parameters": self.parameters[:capacity],
            "capacity": capacity,
        }
        if self.time_table_cache_enabled:
            self._time_table_cache.clear()
            self._time_table_cache[key] = table
        return table

    def _candidate_time_table(self, trajectory_times: torch.Tensor) -> dict[str, Any]:
        """Build an uncached differentiable table for candidate-specific times."""

        if self.plan_start_unix_ns <= 0:
            raise DynamicWorldError("dynamic plan context has not been set")
        if trajectory_times.ndim != 2:
            raise ValueError("trajectory_times must have shape [batch,time]")
        if not torch.isfinite(trajectory_times).all().item():
            raise ValueError("trajectory_times contains NaN or Inf")
        if not torch.allclose(
            trajectory_times[:, 0],
            torch.zeros_like(trajectory_times[:, 0]),
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError("trajectory_times must begin at zero")
        if not (torch.diff(trajectory_times, dim=-1) > 0.0).all().item():
            raise ValueError("trajectory_times must be strictly increasing")

        plan_offset = (self.plan_start_unix_ns - self.stamp_unix_ns) * 1e-9
        relative_times = trajectory_times + plan_offset
        valid_horizon = (self.valid_until_unix_ns - self.stamp_unix_ns) * 1e-9
        if (relative_times[..., -1] > valid_horizon + 1e-9).any().item():
            raise DynamicWorldError("candidate trajectory exceeds dynamic-world prediction validity")
        capacity = self._active_capacity()
        centers = self.position[:capacity] + relative_times[..., None, None] * self.velocity[
            :capacity
        ]
        return {
            "relative_times": relative_times,
            "centers": centers,
            "inflation": self._inflation(relative_times, capacity),
            "rotation": self.rotation[:capacity],
            "parameters": self.parameters[:capacity],
            "capacity": capacity,
        }

    def _query_table(
        self,
        points: torch.Tensor,
        trajectory_times: torch.Tensor | None,
    ) -> dict[str, Any]:
        if trajectory_times is None:
            table = self._time_table(points.shape[1], points.dtype, points.device)
            # Give both paths the same [B,H,M,...] table contract.
            return {
                **table,
                "centers": table["centers"].unsqueeze(0),
                "inflation": table["inflation"].unsqueeze(0),
            }
        if trajectory_times.shape != points.shape[:2]:
            raise ValueError(
                f"trajectory_times must have shape {tuple(points.shape[:2])}, "
                f"got {tuple(trajectory_times.shape)}"
            )
        if trajectory_times.dtype != points.dtype or trajectory_times.device != points.device:
            raise ValueError("trajectory_times must match points dtype and device")
        return self._candidate_time_table(trajectory_times)

    @staticmethod
    def _shape_distance_and_gradient(
        local: torch.Tensor,
        parameters: torch.Tensor,
        shape_code: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = torch.finfo(local.dtype).eps
        if shape_code == 0:
            norm = torch.linalg.norm(local, dim=-1)
            return norm - parameters[None, None, :, None, 0], local / norm.clamp_min(eps)[..., None]
        if shape_code == 1:
            half_extents = parameters[None, None, :, None, :]
            box_q = torch.abs(local) - half_extents
            box_outside = torch.relu(box_q)
            outside_norm = torch.linalg.norm(box_outside, dim=-1)
            distance = outside_norm + torch.clamp(box_q.amax(dim=-1), max=0.0)
            outside_gradient = torch.sign(local) * box_outside / outside_norm.clamp_min(eps)[..., None]
            axis = box_q.argmax(dim=-1)
            inside_gradient = torch.zeros_like(local).scatter_(
                -1, axis[..., None], torch.gather(torch.sign(local), -1, axis[..., None])
            )
            gradient = torch.where(
                (outside_norm > eps)[..., None], outside_gradient, inside_gradient
            )
            return distance, gradient
        capsule_half = parameters[None, None, :, None, 1]
        closest_z = local[..., 2].clamp(-capsule_half, capsule_half)
        capsule_delta = local.clone()
        capsule_delta[..., 2] = capsule_delta[..., 2] - closest_z
        norm = torch.linalg.norm(capsule_delta, dim=-1)
        return (
            norm - parameters[None, None, :, None, 0],
            capsule_delta / norm.clamp_min(eps)[..., None],
        )

    def _evaluate_indices(
        self,
        points: torch.Tensor,
        table: dict[str, Any],
        indices: torch.Tensor,
        shape_code: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centers = table["centers"].index_select(2, indices)
        rotation = table["rotation"].index_select(0, indices)
        parameters = table["parameters"].index_select(0, indices)
        world_delta = points[:, :, None, :, :] - centers[:, :, :, None, :]
        local = torch.einsum("bhmld,mdk->bhmlk", world_delta, rotation)
        distance, gradient_local = self._shape_distance_and_gradient(local, parameters, shape_code)
        gradient_world = torch.einsum(
            "bhmld,mdk->bhmlk", gradient_local, rotation.transpose(-1, -2)
        )
        distance = distance - table["inflation"].index_select(2, indices)[..., None]
        return distance, gradient_world

    def _evaluate_all_shapes(
        self, points: torch.Tensor, table: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        capacity = table["capacity"]
        world_delta = points[:, :, None, :, :] - table["centers"][:, :, :, None, :]
        local = torch.einsum("bhmld,mdk->bhmlk", world_delta, table["rotation"])
        distances_by_shape = []
        gradients_by_shape = []
        for shape_code in range(3):
            distance, gradient = self._shape_distance_and_gradient(
                local, table["parameters"], shape_code
            )
            distances_by_shape.append(distance)
            gradients_by_shape.append(gradient)
        shape = self.shape_code[None, None, :capacity, None]
        distances = torch.where(
            shape == 0,
            distances_by_shape[0],
            torch.where(shape == 1, distances_by_shape[1], distances_by_shape[2]),
        )
        gradients_local = torch.where(
            (shape == 0)[..., None],
            gradients_by_shape[0],
            torch.where(
                (shape == 1)[..., None], gradients_by_shape[1], gradients_by_shape[2]
            ),
        )
        gradients_world = torch.einsum(
            "bhmld,mdk->bhmlk",
            gradients_local,
            table["rotation"].transpose(-1, -2),
        )
        return (
            distances - table["inflation"][..., None],
            gradients_world,
        )

    def _shape_groups(self, capacity: int) -> tuple[tuple[int, torch.Tensor], ...]:
        if self.shape_grouping_enabled:
            return (
                (0, self.sphere_indices),
                (1, self.box_indices),
                (2, self.capsule_indices),
            )
        indices = torch.arange(capacity, dtype=torch.long, device=self.active.device)
        return tuple((code, indices[self.shape_code[:capacity] == code]) for code in range(3))

    def signed_distances_and_gradients(
        self,
        points: torch.Tensor,
        trajectory_times: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return effective distances/gradients as ``[B,H,M,L(,3)]``."""

        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("dynamic SDF expects [batch,time,links,3] points")
        batch, horizon, links, _ = points.shape
        table = self._query_table(points, trajectory_times)
        capacity = table["capacity"]
        distances = torch.full(
            (batch, horizon, capacity, links), torch.inf, dtype=points.dtype, device=points.device
        )
        gradients_world = torch.zeros(
            (batch, horizon, capacity, links, 3), dtype=points.dtype, device=points.device
        )
        if capacity and not self.shape_grouping_enabled:
            distances, gradients_world = self._evaluate_all_shapes(points, table)
        else:
            for shape_code, indices in self._shape_groups(capacity):
                if not indices.numel():
                    continue
                group_distance, group_gradient = self._evaluate_indices(
                    points, table, indices, shape_code
                )
                distances = distances.index_copy(2, indices, group_distance)
                gradients_world = gradients_world.index_copy(2, indices, group_gradient)
        inactive = ~self.active[None, None, :capacity, None]
        distances = torch.where(inactive, torch.full_like(distances, torch.inf), distances)
        gradients_world = torch.where(
            inactive[..., None], torch.zeros_like(gradients_world), gradients_world
        )
        return distances, gradients_world

    def minimum_signed_distance_and_gradient(
        self,
        points: torch.Tensor,
        trajectory_times: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce local SDFs without materializing the full object dimension."""

        if not self.fused_reduction_enabled or not self.shape_grouping_enabled:
            distances, gradients = self.signed_distances_and_gradients(
                points, trajectory_times=trajectory_times
            )
            minimum, active_object = distances.min(dim=-2)
            gather_index = active_object.unsqueeze(-2).unsqueeze(-1).expand(
                *active_object.shape[:-1], 1, active_object.shape[-1], 3
            )
            return minimum, gradients.gather(-3, gather_index).squeeze(-3)
        _, horizon, links, _ = points.shape
        table = self._query_table(points, trajectory_times)
        best_distance = torch.full(
            points.shape[:-2] + (links,), torch.inf, dtype=points.dtype, device=points.device
        )
        best_gradient = torch.zeros_like(points)
        for shape_code, indices in self._shape_groups(table["capacity"]):
            if not indices.numel():
                continue
            distances, gradients = self._evaluate_indices(points, table, indices, shape_code)
            group_distance, group_object = distances.min(dim=-2)
            gather_index = group_object.unsqueeze(-2).unsqueeze(-1).expand(
                *group_object.shape[:-1], 1, group_object.shape[-1], 3
            )
            group_gradient = gradients.gather(-3, gather_index).squeeze(-3)
            replace = group_distance < best_distance
            best_distance = torch.where(replace, group_distance, best_distance)
            best_gradient = torch.where(replace[..., None], group_gradient, best_gradient)
        return best_distance, best_gradient


class StaticDynamicCollisionField:
    """Collision-field adapter combining the resident static SDF and dynamic SDFs."""

    def __init__(self, static_field, dynamic_world: FixedCapacityDynamicWorld) -> None:
        self.static_field = static_field
        self.dynamic_world = dynamic_world
        self.robot = static_field.robot
        self.collision_margins = static_field.collision_margins
        self.cutoff_margin = static_field.cutoff_margin
        self.clamp_sdf = static_field.clamp_sdf

    def object_signed_distances(self, link_pos, get_gradient=False, **kwargs):
        trajectory_times = kwargs.pop("trajectory_times", None)
        dynamic_distance, dynamic_gradient = self.dynamic_world.signed_distances_and_gradients(
            link_pos, trajectory_times=trajectory_times
        )
        if get_gradient:
            static_distance, static_gradient = self.static_field.object_signed_distances(
                link_pos, get_gradient=True, **kwargs
            )
            return (
                torch.cat((static_distance, dynamic_distance), dim=-2),
                torch.cat((static_gradient, dynamic_gradient), dim=-3),
            )
        static_distance = self.static_field.object_signed_distances(link_pos, **kwargs)
        return torch.cat((static_distance, dynamic_distance), dim=-2)

    def object_signed_distance_gradients(self, link_pos, **kwargs):
        trajectory_times = kwargs.pop("trajectory_times", None)
        static_gradient = self.static_field.object_signed_distance_gradients(link_pos, **kwargs)
        _, dynamic_gradient = self.dynamic_world.signed_distances_and_gradients(
            link_pos, trajectory_times=trajectory_times
        )
        return torch.cat((static_gradient, dynamic_gradient), dim=-3)

    def compute_embodiment_taskspace_sdf_and_gradient(self, link_pos, **kwargs):
        trajectory_times = kwargs.pop("trajectory_times", None)
        if self.dynamic_world.fused_reduction_enabled:
            static_cost, static_gradient = self.static_field.compute_distance_field_cost_and_gradient(
                link_pos, **kwargs
            )
            dynamic_distance, dynamic_distance_gradient = (
                self.dynamic_world.minimum_signed_distance_and_gradient(
                    link_pos, trajectory_times=trajectory_times
                )
            )
            margins = self.collision_margins
            link_indices = kwargs.get("link_indices")
            if link_indices is not None:
                margins = margins.index_select(
                    0,
                    torch.as_tensor(link_indices, dtype=torch.long, device=link_pos.device),
                )
            dynamic_cost = torch.relu(margins + self.cutoff_margin - dynamic_distance)
            dynamic_gradient = torch.where(
                (dynamic_cost > 0.0)[..., None],
                -dynamic_distance_gradient,
                torch.zeros_like(dynamic_distance_gradient),
            )
            use_dynamic = dynamic_cost > static_cost
            return (
                torch.where(use_dynamic, dynamic_cost, static_cost),
                torch.where(use_dynamic[..., None], dynamic_gradient, static_gradient),
            )
        distances, gradients = self.object_signed_distances(
            link_pos, get_gradient=True, trajectory_times=trajectory_times
        )
        margins = self.collision_margins
        link_indices = kwargs.get("link_indices")
        if link_indices is not None:
            margins = margins.index_select(0, torch.as_tensor(link_indices, dtype=torch.long, device=link_pos.device))
        penetration = torch.relu(margins + self.cutoff_margin - distances)
        cost, active_object = penetration.max(dim=-2)
        gather_index = (
            active_object.unsqueeze(-2)
            .unsqueeze(-1)
            .expand(*active_object.shape[:-1], 1, active_object.shape[-1], gradients.shape[-1])
        )
        active_gradient = gradients.gather(-3, gather_index).squeeze(-3)
        active_gradient = torch.where((cost > 0.0)[..., None], -active_gradient, torch.zeros_like(active_gradient))
        return cost, active_gradient

    def compute_distance_field_cost_and_gradient(self, link_pos, **kwargs):
        return self.compute_embodiment_taskspace_sdf_and_gradient(link_pos, **kwargs)

    def compute_cost(self, q_pos, link_pos, *, field_type="sdf", **kwargs):
        """Compatibility path used by the final best-trajectory recheck."""

        del q_pos
        if link_pos.shape[-2:] == (3, 4):
            link_pos = link_pos[..., :3, 3]
        squeeze_batch = link_pos.ndim == 3
        if squeeze_batch:
            link_pos = link_pos.unsqueeze(0)
        if link_pos.ndim != 4:
            raise ValueError("collision field expects [batch,time,links,3] positions")
        cutoff = float(kwargs.get("margin", self.cutoff_margin))
        margins = self.collision_margins + cutoff
        trajectory_times = kwargs.pop("trajectory_times", None)
        if self.dynamic_world.fused_reduction_enabled:
            static_distances = self.static_field.object_signed_distances(link_pos, **kwargs)
            dynamic_distance, _ = self.dynamic_world.minimum_signed_distance_and_gradient(
                link_pos, trajectory_times=trajectory_times
            )
            if field_type == "occupancy":
                result = (
                    (static_distances <= margins).any(dim=-2) | (dynamic_distance <= margins)
                ).any(dim=-1)
            elif field_type == "sdf":
                static_penetration = torch.relu(margins - static_distances).max(dim=-2).values
                dynamic_penetration = torch.relu(margins - dynamic_distance)
                result = torch.maximum(static_penetration, dynamic_penetration).sum(dim=-1)
            else:
                raise ValueError(f"unsupported field_type {field_type!r}")
            return result.squeeze(0) if squeeze_batch else result
        distances = self.object_signed_distances(
            link_pos, trajectory_times=trajectory_times, **kwargs
        )
        collisions = distances <= margins
        if field_type == "occupancy":
            result = collisions.any(dim=-1).any(dim=-1)
        elif field_type == "sdf":
            result = torch.relu(margins - distances).max(dim=-2).values.sum(dim=-1)
        else:
            raise ValueError(f"unsupported field_type {field_type!r}")
        return result.squeeze(0) if squeeze_batch else result
