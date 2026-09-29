"""FR3 sphere-model collision checks for a space-time sampling baseline."""

from __future__ import annotations

import math
from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation


def _vector(value, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain {size} finite numbers")
    return array


def _rotation_xyzw(value) -> np.ndarray:
    quaternion = _vector(value, 4, "orientation_xyzw")
    if np.linalg.norm(quaternion) < 1e-9:
        raise ValueError("orientation quaternion is zero")
    return Rotation.from_quat(quaternion).as_matrix()


def _sdf(points: np.ndarray, shape: dict) -> np.ndarray:
    kind = shape["type"]
    if kind == "sphere":
        return np.linalg.norm(points, axis=-1) - float(shape["radius"])
    if kind == "box":
        half = 0.5 * _vector(shape["size_xyz"], 3, "box size")
        delta = np.abs(points) - half
        return np.linalg.norm(np.maximum(delta, 0.0), axis=-1) + np.minimum(np.max(delta, axis=-1), 0.0)
    if kind == "capsule":
        offset = points.copy()
        offset[..., 2] -= np.clip(offset[..., 2], -0.5 * float(shape["length"]), 0.5 * float(shape["length"]))
        return np.linalg.norm(offset, axis=-1) - float(shape["radius"])
    raise ValueError(f"unsupported obstacle type {kind!r}")


class StrrtCollisionWorld:
    """Use the same 56-sphere robot model and obstacle inflation as the ROS guard."""

    def __init__(self, robot, static_scene: str | Path, *, covariance_sigma: float = 3.0,
                 process_acceleration_std_m_s2: float = 0.01) -> None:
        from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor

        self.robot = robot
        self._link_pos_from_link_tensor = link_pos_from_link_tensor
        self.radii = robot.link_collision_spheres_radii.detach().cpu().numpy().astype(np.float64)
        self.q_min = np.asarray(robot.q_pos_min_np, dtype=np.float64)
        self.q_max = np.asarray(robot.q_pos_max_np, dtype=np.float64)
        self.dq_max = np.asarray(robot.dq_max_np, dtype=np.float64)
        self.ddq_max = np.asarray(robot.ddq_max_np, dtype=np.float64)
        pairs = np.asarray([(a, b) for a, b, *_ in robot.link_self_collision_tuples], dtype=np.intp)
        self.self_a, self.self_b = pairs[:, 0], pairs[:, 1]
        self.covariance_sigma = float(covariance_sigma)
        self.process_variance = float(process_acceleration_std_m_s2) ** 2
        scene = json.loads(Path(static_scene).read_text(encoding="utf-8"))
        if scene.get("env_name") != "EnvOpenDrawerShelf" or scene.get("unsupported_obstacles"):
            raise ValueError("static scene does not fully describe EnvOpenDrawerShelf")
        self.static = []
        for item in scene["obstacles"]:
            kind = item["type"]
            shape = {"type": kind}
            if kind == "box":
                shape["size_xyz"] = item["size"]
            elif kind == "sphere":
                shape["radius"] = item["radius"]
            else:
                raise ValueError(f"unsupported static obstacle {kind!r}")
            w, x, y, z = item["orientation"]
            self.static.append((shape, _vector(item["position"], 3, "static position"),
                                _rotation_xyzw([x, y, z, w])))
        self.world_version = 0
        self.snapshot: dict | None = None

    def update(self, snapshot: dict) -> int:
        version = snapshot.get("world_version")
        if not isinstance(version, int) or isinstance(version, bool) or version <= self.world_version:
            raise ValueError("dynamic world version must increase")
        if snapshot.get("frame_id") != "fr3_link0":
            raise ValueError("dynamic world must be in fr3_link0")
        stamp, valid_until = int(snapshot["stamp_unix_ns"]), int(snapshot["valid_until_unix_ns"])
        if valid_until <= stamp or not isinstance(snapshot.get("objects"), list):
            raise ValueError("dynamic world validity interval or objects are invalid")
        prepared = []
        for item in snapshot["objects"]:
            local_sdf = dict(item["local_sdf"])
            if local_sdf.get("type") not in ("sphere", "box", "capsule"):
                raise ValueError("unsupported dynamic object")
            inflation = dict(item["inflation"])
            if inflation.get("mode") not in ("linear", "covariance"):
                raise ValueError("unsupported inflation mode")
            prepared.append((
                local_sdf,
                _vector(item["pose"]["position"], 3, "dynamic position"),
                _rotation_xyzw(item["pose"]["orientation_xyzw"]),
                _vector(item["linear_velocity"], 3, "dynamic velocity"),
                np.asarray(item["covariance_6x6"], dtype=np.float64).reshape(6, 6),
                inflation,
            ))
        self.snapshot = {"world_version": version, "stamp_s": stamp * 1e-9,
                         "valid_until_s": valid_until * 1e-9, "objects": prepared}
        self.world_version = version
        return version

    def spheres(self, q: np.ndarray) -> np.ndarray:
        return self.spheres_many(np.asarray(q, dtype=np.float64).reshape(1, 7))[0]

    def spheres_many(self, q: np.ndarray) -> np.ndarray:
        import torch

        state = torch.as_tensor(np.asarray(q, dtype=np.float32).reshape(-1, 7), dtype=torch.float32)
        with torch.no_grad():
            poses = torch.stack(self.robot.fk_collision_spheres(state), dim=-3)
            points = self._link_pos_from_link_tensor(poses)[..., :3]
        return points.detach().cpu().numpy().astype(np.float64)

    def static_valid(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all():
            return False
        if np.any(q < self.q_min) or np.any(q > self.q_max):
            return False
        return self._static_spheres_valid(self.spheres(q))

    def _static_spheres_valid(self, centers: np.ndarray) -> bool:
        if np.any(np.linalg.norm(centers[self.self_a] - centers[self.self_b], axis=-1)
                  <= self.radii[self.self_a] + self.radii[self.self_b]):
            return False
        for shape, position, rotation in self.static:
            local = (centers - position) @ rotation
            if np.any(_sdf(local, shape) <= self.radii):
                return False
        return True

    def is_valid(self, q: np.ndarray, absolute_time_s: float) -> bool:
        q = np.asarray(q, dtype=np.float64)
        world = self.snapshot
        if q.shape != (7,) or not np.isfinite(q).all() or not math.isfinite(absolute_time_s):
            return False
        if np.any(q < self.q_min) or np.any(q > self.q_max) or world is None:
            return False
        if absolute_time_s < world["stamp_s"] - 1e-6 or absolute_time_s > world["valid_until_s"] + 1e-6:
            return False
        centers = self.spheres(q)
        return self.centers_valid(centers, absolute_time_s)

    def centers_valid(self, centers: np.ndarray, absolute_time_s: float) -> bool:
        world = self.snapshot
        if world is None or absolute_time_s < world["stamp_s"] - 1e-6 or absolute_time_s > world["valid_until_s"] + 1e-6:
            return False
        if not self._static_spheres_valid(centers):
            return False
        dt = absolute_time_s - world["stamp_s"]
        for shape, position, rotation, velocity, covariance, inflation in world["objects"]:
            local = (centers - (position + dt * velocity)) @ rotation
            if inflation["mode"] == "linear":
                extra = float(inflation["base_m"]) + float(inflation["horizon_rate_m_s"]) * dt
            else:
                pp, pv, vp, vv = covariance[:3, :3], covariance[:3, 3:], covariance[3:, :3], covariance[3:, 3:]
                predicted = pp + dt * (pv + vp) + dt * dt * vv + np.eye(3) * self.process_variance * dt**3 / 3.0
                extra = float(inflation["base_m"]) + self.covariance_sigma * math.sqrt(
                    max(0.0, float(np.linalg.eigvalsh(predicted).max())))
            if np.any(_sdf(local, shape) - extra <= self.radii):
                return False
        return True
