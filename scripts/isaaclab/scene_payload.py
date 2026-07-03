"""MPD scene payload helpers shared by the MPD and IsaacLab entry points."""

from __future__ import annotations

import numpy as np

from torch_robotics.environments.primitives import MultiBoxField, MultiSphereField
from torch_robotics.torch_kinematics_tree.geometrics.quaternion import rotation_matrix_to_q
from torch_robotics.torch_utils.torch_utils import to_numpy


def _as_3d_vector(value, fill_value=0.0):
    value_np = np.asarray(to_numpy(value), dtype=float).reshape(-1)
    if value_np.shape[0] == 2:
        value_np = np.concatenate([value_np, np.asarray([fill_value], dtype=float)])
    if value_np.shape[0] != 3:
        raise ValueError(f"Expected a 2D or 3D vector, got shape {value_np.shape}.")
    return value_np


def _object_pose(obj):
    pos = _as_3d_vector(obj.pos)
    ori = np.asarray(to_numpy(obj.ori), dtype=float)
    if ori.shape != (3, 3):
        raise ValueError(f"Expected object orientation to have shape (3, 3), got {ori.shape}.")
    quat_wxyz = np.asarray(to_numpy(rotation_matrix_to_q(obj.ori)), dtype=float).reshape(4)
    return pos, ori, quat_wxyz


def export_isaaclab_scene_payload(env, include_boxes=False):
    """Serialize MPD primitive obstacles into a small IsaacLab-friendly payload."""

    obstacles = []
    unsupported_obstacles = []
    object_groups = [
        ("fixed", getattr(env, "obj_fixed_list", None) or []),
        ("extra", getattr(env, "obj_extra_list", None) or []),
    ]

    for group_name, objects in object_groups:
        for obj_idx, obj in enumerate(objects):
            obj_pos, obj_ori, obj_quat_wxyz = _object_pose(obj)
            object_name = getattr(obj, "name", f"{group_name}_{obj_idx}")
            for field_idx, field in enumerate(getattr(obj, "fields", [])):
                if isinstance(field, MultiSphereField):
                    for prim_idx, (center, radius) in enumerate(zip(field.centers, field.radii)):
                        center_local = _as_3d_vector(center)
                        center_world = obj_ori @ center_local + obj_pos
                        obstacles.append(
                            {
                                "type": "sphere",
                                "name": f"{object_name}_{field_idx}_{prim_idx}",
                                "group": group_name,
                                "position": center_world.tolist(),
                                "orientation": [1.0, 0.0, 0.0, 0.0],
                                "radius": float(np.asarray(to_numpy(radius)).reshape(())),
                            }
                        )
                elif isinstance(field, MultiBoxField) and include_boxes:
                    for prim_idx, (center, size) in enumerate(zip(field.centers, field.sizes)):
                        center_local = _as_3d_vector(center)
                        size_3d = _as_3d_vector(size, fill_value=0.1)
                        center_world = obj_ori @ center_local + obj_pos
                        obstacles.append(
                            {
                                "type": "box",
                                "name": f"{object_name}_{field_idx}_{prim_idx}",
                                "group": group_name,
                                "position": center_world.tolist(),
                                "orientation": obj_quat_wxyz.tolist(),
                                "size": size_3d.tolist(),
                            }
                        )
                else:
                    unsupported_obstacles.append(
                        {
                            "object": str(object_name),
                            "group": group_name,
                            "field_type": type(field).__name__,
                            "reason": "box export disabled" if isinstance(field, MultiBoxField) else "unsupported",
                        }
                    )

    return {
        "schema": "mpd_isaaclab_scene",
        "schema_version": 1,
        "env_name": getattr(env, "name", type(env).__name__),
        "obstacles": obstacles,
        "unsupported_obstacles": unsupported_obstacles,
    }
