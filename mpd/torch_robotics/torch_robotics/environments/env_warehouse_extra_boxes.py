"""YAML-driven extra-box environments for Warehouse/Panda planning."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

from torch_robotics.environments.env_warehouse import EnvWarehouse
from torch_robotics.environments.primitives import MultiBoxField, ObjectField
from torch_robotics.torch_kinematics_tree.utils.files import get_configs_path
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


WAREHOUSE_EXTRA_BOXES_SCHEMA = "mpd_warehouse_extra_boxes"
WAREHOUSE_EXTRA_BOXES_SCHEMA_VERSION = 1
DEFAULT_WAREHOUSE_EXTRA_BOXES_YAML = Path("warehouse/extra_boxes/warehouse_extra_boxes_v00.yaml")


def resolve_warehouse_extra_boxes_yaml(extra_boxes_yaml) -> Path:
    """Resolve an absolute, cwd-relative, or torch_robotics-config-relative YAML path."""

    if extra_boxes_yaml is None:
        raise ValueError("extra_boxes_yaml must be provided for a parameterized Warehouse extra-box environment.")

    path = Path(os.path.expandvars(os.path.expanduser(str(extra_boxes_yaml))))
    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Warehouse extra-box YAML does not exist: {resolved}")
        return resolved

    candidates = (
        (Path.cwd() / path).resolve(),
        (Path(get_configs_path()) / path).resolve(),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Warehouse extra-box YAML {path!s} was not found. Searched: {searched}")


def _validated_box_vector(value, field_name: str, box_index: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"boxes[{box_index}].{field_name} must contain three numeric values.") from error
    if vector.shape != (3,):
        raise ValueError(f"boxes[{box_index}].{field_name} must have shape [3], got {list(vector.shape)}.")
    if not np.isfinite(vector).all():
        raise ValueError(f"boxes[{box_index}].{field_name} contains NaN or Inf.")
    return vector


def load_warehouse_extra_boxes_yaml(extra_boxes_yaml) -> dict:
    """Load and validate a Warehouse extra-box scenario."""

    path = resolve_warehouse_extra_boxes_yaml(extra_boxes_yaml)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"Warehouse extra-box YAML must contain a mapping at the root: {path}")
    if config.get("schema") != WAREHOUSE_EXTRA_BOXES_SCHEMA:
        raise ValueError(
            f"Warehouse extra-box YAML schema must be {WAREHOUSE_EXTRA_BOXES_SCHEMA!r}, "
            f"got {config.get('schema')!r}: {path}"
        )
    if config.get("schema_version") != WAREHOUSE_EXTRA_BOXES_SCHEMA_VERSION:
        raise ValueError(
            f"Warehouse extra-box YAML schema_version must be {WAREHOUSE_EXTRA_BOXES_SCHEMA_VERSION}, "
            f"got {config.get('schema_version')!r}: {path}"
        )

    reference_frame = config.get("reference_frame", "panda_link0_unrotated")
    if reference_frame != "panda_link0_unrotated":
        raise ValueError(
            "Warehouse extra-box reference_frame must be 'panda_link0_unrotated'; " f"got {reference_frame!r}."
        )

    boxes = config.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError(f"Warehouse extra-box YAML boxes must be a list: {path}")

    normalized_boxes = []
    names = set()
    for box_index, box in enumerate(boxes):
        if not isinstance(box, dict):
            raise ValueError(f"boxes[{box_index}] must be a mapping.")

        name = box.get("name", f"box_{box_index:03d}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"boxes[{box_index}].name must be a non-empty string.")
        if name in names:
            raise ValueError(f"Duplicate Warehouse extra-box name {name!r}.")
        names.add(name)

        center = _validated_box_vector(box.get("center"), "center", box_index)
        size = _validated_box_vector(box.get("size"), "size", box_index)
        if np.any(size <= 0):
            raise ValueError(f"boxes[{box_index}].size values must all be positive.")

        normalized_boxes.append({"name": name, "center": center, "size": size})

    object_name = config.get("object_name", "warehouse-extra-boxes")
    if not isinstance(object_name, str) or not object_name.strip():
        raise ValueError("Warehouse extra-box object_name must be a non-empty string.")

    normalized_config = dict(config)
    normalized_config["reference_frame"] = reference_frame
    normalized_config["object_name"] = object_name
    normalized_config["boxes"] = normalized_boxes
    normalized_config["source_path"] = path.as_posix()
    return normalized_config


def create_warehouse_extra_box_fields(extra_boxes_yaml, tensor_args=DEFAULT_TENSOR_ARGS):
    """Create the ObjectField list consumed by EnvWarehouse."""

    config = load_warehouse_extra_boxes_yaml(extra_boxes_yaml)
    boxes = config["boxes"]
    if not boxes:
        return [], config

    centers = np.stack([box["center"] for box in boxes])
    sizes = np.stack([box["size"] for box in boxes])
    boxes_field = MultiBoxField(centers, sizes, tensor_args=tensor_args)
    return [ObjectField([boxes_field], config["object_name"])], config


class EnvWarehouseExtraBoxes(EnvWarehouse):
    """Warehouse whose extra axis-aligned boxes are defined by a YAML file."""

    def __init__(self, extra_boxes_yaml, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        if "obj_extra_list" in kwargs:
            raise ValueError("Use extra_boxes_yaml instead of obj_extra_list with EnvWarehouseExtraBoxes.")

        obj_extra_list, config = create_warehouse_extra_box_fields(
            extra_boxes_yaml,
            tensor_args=tensor_args,
        )
        self.extra_boxes_yaml = config["source_path"]
        self.extra_boxes_config = config
        super().__init__(obj_extra_list=obj_extra_list, tensor_args=tensor_args, **kwargs)


class EnvWarehouseExtraObjectsV00(EnvWarehouseExtraBoxes):
    """Backward-compatible scene id backed by the external V00 YAML."""

    def __init__(
        self,
        extra_boxes_yaml=DEFAULT_WAREHOUSE_EXTRA_BOXES_YAML,
        tensor_args=DEFAULT_TENSOR_ARGS,
        **kwargs,
    ):
        super().__init__(
            extra_boxes_yaml=extra_boxes_yaml,
            tensor_args=tensor_args,
            **kwargs,
        )
