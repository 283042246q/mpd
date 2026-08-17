"""Fixed-capacity, fixed-timing dynamic collision fields for Phase 4.

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
        process_acceleration_std_m_s2: float = 0.25,
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
            covariance[index].copy_(
                torch.as_tensor(flat_covariance, **self.tensor_args).reshape(6, 6)
            )
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
        return version

    def set_plan_start(self, plan_start_unix_ns: int, *, world_version: int) -> None:
        if world_version != self.world_version:
            raise DynamicWorldError(
                f"requested world_version {world_version} != loaded {self.world_version}"
            )
        if not isinstance(plan_start_unix_ns, int) or plan_start_unix_ns < self.stamp_unix_ns:
            raise DynamicWorldError("plan_start_unix_ns predates the world snapshot")
        horizon_end = plan_start_unix_ns + int(self.trajectory_duration_s * 1e9)
        if horizon_end > self.valid_until_unix_ns:
            raise DynamicWorldError("trajectory exceeds dynamic-world prediction validity")
        self.plan_start_unix_ns = plan_start_unix_ns

    def _relative_times(self, horizon: int, dtype, device) -> torch.Tensor:
        if self.plan_start_unix_ns <= 0:
            raise DynamicWorldError("dynamic plan context has not been set")
        plan_offset = (self.plan_start_unix_ns - self.stamp_unix_ns) * 1e-9
        trajectory_times = torch.linspace(
            0.0, self.trajectory_duration_s, horizon, dtype=dtype, device=device
        )
        return trajectory_times + plan_offset

    def _inflation(self, relative_times: torch.Tensor) -> torch.Tensor:
        # [H,M], where covariance is propagated by the constant-velocity model.
        dt = relative_times[:, None]
        linear = self.base_inflation[None, :] + self.horizon_inflation_rate[None, :] * dt
        p_pp = self.covariance[:, :3, :3]
        p_pv = self.covariance[:, :3, 3:]
        p_vp = self.covariance[:, 3:, :3]
        p_vv = self.covariance[:, 3:, 3:]
        propagated = (
            p_pp[None]
            + dt[..., None, None] * (p_pv + p_vp)[None]
            + dt[..., None, None].square() * p_vv[None]
        )
        process = self.process_variance * dt.pow(3) / 3.0
        propagated = propagated + process[..., None, None] * torch.eye(
            3, dtype=relative_times.dtype, device=relative_times.device
        )
        # eigvalsh is deterministic and the object count is deliberately small.
        sigma = torch.linalg.eigvalsh(propagated).amax(dim=-1).clamp_min(0.0).sqrt()
        covariance = self.base_inflation[None, :] + self.covariance_sigma * sigma
        return torch.where(self.inflation_code[None, :] == 1, covariance, linear)

    def signed_distances_and_gradients(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return effective distances/gradients as ``[B,H,M,L(,3)]``."""

        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("dynamic SDF expects [batch,time,links,3] points")
        _, horizon, _, _ = points.shape
        relative_times = self._relative_times(horizon, points.dtype, points.device)
        centers = self.position[None, :, :] + relative_times[:, None, None] * self.velocity[None, :, :]
        world_delta = points[:, :, None, :, :] - centers[None, :, :, None, :]
        # R maps local to world, so row-vector world points transform with R.
        local = torch.einsum("bhmld,mdk->bhmlk", world_delta, self.rotation)
        eps = torch.finfo(points.dtype).eps

        norm = torch.linalg.norm(local, dim=-1)
        sphere_distance = norm - self.parameters[None, None, :, None, 0]
        sphere_gradient_local = local / norm.clamp_min(eps)[..., None]

        half_extents = self.parameters[None, None, :, None, :]
        box_q = torch.abs(local) - half_extents
        box_outside = torch.relu(box_q)
        box_outside_norm = torch.linalg.norm(box_outside, dim=-1)
        box_distance = box_outside_norm + torch.clamp(box_q.amax(dim=-1), max=0.0)
        box_outside_gradient = (
            torch.sign(local) * box_outside / box_outside_norm.clamp_min(eps)[..., None]
        )
        box_axis = box_q.argmax(dim=-1)
        box_inside_gradient = torch.zeros_like(local).scatter_(
            -1,
            box_axis[..., None],
            torch.gather(torch.sign(local), -1, box_axis[..., None]),
        )
        box_gradient_local = torch.where(
            (box_outside_norm > eps)[..., None], box_outside_gradient, box_inside_gradient
        )

        capsule_half = self.parameters[None, None, :, None, 1]
        closest_z = local[..., 2].clamp(-capsule_half, capsule_half)
        capsule_delta = local.clone()
        capsule_delta[..., 2] = capsule_delta[..., 2] - closest_z
        capsule_norm = torch.linalg.norm(capsule_delta, dim=-1)
        capsule_distance = capsule_norm - self.parameters[None, None, :, None, 0]
        capsule_gradient_local = capsule_delta / capsule_norm.clamp_min(eps)[..., None]

        shape = self.shape_code[None, None, :, None]
        distances = torch.where(
            shape == 0,
            sphere_distance,
            torch.where(shape == 1, box_distance, capsule_distance),
        )
        gradients_local = torch.where(
            (shape == 0)[..., None],
            sphere_gradient_local,
            torch.where((shape == 1)[..., None], box_gradient_local, capsule_gradient_local),
        )
        gradients_world = torch.einsum("bhmld,mdk->bhmlk", gradients_local, self.rotation.transpose(-1, -2))
        distances = distances - self._inflation(relative_times)[None, :, :, None]
        inactive = ~self.active[None, None, :, None]
        distances = torch.where(inactive, torch.full_like(distances, torch.inf), distances)
        gradients_world = torch.where(inactive[..., None], torch.zeros_like(gradients_world), gradients_world)
        return distances, gradients_world


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
        dynamic_distance, dynamic_gradient = self.dynamic_world.signed_distances_and_gradients(link_pos)
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
        static_gradient = self.static_field.object_signed_distance_gradients(link_pos, **kwargs)
        _, dynamic_gradient = self.dynamic_world.signed_distances_and_gradients(link_pos)
        return torch.cat((static_gradient, dynamic_gradient), dim=-3)

    def compute_embodiment_taskspace_sdf_and_gradient(self, link_pos, **kwargs):
        distances, gradients = self.object_signed_distances(link_pos, get_gradient=True)
        margins = self.collision_margins
        link_indices = kwargs.get("link_indices")
        if link_indices is not None:
            margins = margins.index_select(
                0, torch.as_tensor(link_indices, dtype=torch.long, device=link_pos.device)
            )
        penetration = torch.relu(margins + self.cutoff_margin - distances)
        cost, active_object = penetration.max(dim=-2)
        gather_index = active_object.unsqueeze(-2).unsqueeze(-1).expand(
            *active_object.shape[:-1], 1, active_object.shape[-1], gradients.shape[-1]
        )
        active_gradient = gradients.gather(-3, gather_index).squeeze(-3)
        active_gradient = torch.where((cost > 0.0)[..., None], -active_gradient, torch.zeros_like(active_gradient))
        return cost, active_gradient

    def compute_distance_field_cost_and_gradient(self, link_pos, **kwargs):
        return self.compute_embodiment_taskspace_sdf_and_gradient(link_pos, **kwargs)
