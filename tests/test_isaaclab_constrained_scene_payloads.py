from types import SimpleNamespace

import numpy as np
import torch

# Compatibility for the older NetworkX version pinned by this repository.
np.int = int
np.float = float
np.bool = bool

from scripts.inference.inference import (
    ISAACLAB_BATCH_SUPPORTED_ENVS,
    ISAACLAB_BOX_OBSTACLE_ENVS,
    _require_isaaclab_batch_support,
)
from scripts.isaaclab.scene_payload import export_isaaclab_scene_payload
from torch_robotics.environments import EnvOpenDrawerShelf, EnvThreePillarsPassage
from torch_robotics.environments.env_open_drawer_shelf import (
    ADJACENT_SHELF_BOXES,
    DRAWER_CABINET_BOXES,
    OPEN_BOTTOM_DRAWER_BOXES,
)
from torch_robotics.environments.env_three_pillars_passage import THREE_PILLAR_BOXES


TENSOR_ARGS = {"device": "cpu", "dtype": torch.float32}


def _make_env(env_cls):
    return env_cls(
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        tensor_args=TENSOR_ARGS,
    )


def _assert_box_payload_matches_specs(payload, box_specs):
    expected_centers = np.concatenate([np.asarray(spec["centers"]) for spec in box_specs], axis=0)
    expected_sizes = np.concatenate([np.asarray(spec["sizes"]) for spec in box_specs], axis=0)
    obstacles = payload["obstacles"]

    assert len(obstacles) == len(expected_centers)
    assert {obstacle["type"] for obstacle in obstacles} == {"box"}
    np.testing.assert_allclose([obstacle["position"] for obstacle in obstacles], expected_centers)
    np.testing.assert_allclose([obstacle["size"] for obstacle in obstacles], expected_sizes)
    np.testing.assert_allclose(
        [obstacle["orientation"] for obstacle in obstacles],
        np.tile([1.0, 0.0, 0.0, 0.0], (len(obstacles), 1)),
    )


def test_open_drawer_shelf_exports_all_boxes_for_isaaclab():
    env = _make_env(EnvOpenDrawerShelf)
    payload = export_isaaclab_scene_payload(env, include_boxes=True)

    assert payload["schema"] == "mpd_isaaclab_scene"
    assert payload["schema_version"] == 1
    assert payload["env_name"] == "EnvOpenDrawerShelf"
    assert payload["unsupported_obstacles"] == []
    assert {obstacle["group"] for obstacle in payload["obstacles"]} == {"fixed"}
    _assert_box_payload_matches_specs(
        payload,
        [DRAWER_CABINET_BOXES, OPEN_BOTTOM_DRAWER_BOXES, ADJACENT_SHELF_BOXES],
    )


def test_three_pillars_exports_all_boxes_for_isaaclab():
    env = _make_env(EnvThreePillarsPassage)
    payload = export_isaaclab_scene_payload(env, include_boxes=True)

    assert payload["env_name"] == "EnvThreePillarsPassage"
    assert payload["unsupported_obstacles"] == []
    _assert_box_payload_matches_specs(payload, [THREE_PILLAR_BOXES])


def test_constrained_environments_are_enabled_for_isaaclab_batch_evaluation():
    for env_cls in (EnvOpenDrawerShelf, EnvThreePillarsPassage):
        env = _make_env(env_cls)
        planning_task = SimpleNamespace(env=env)
        assert env.name in ISAACLAB_BATCH_SUPPORTED_ENVS
        assert env.name in ISAACLAB_BOX_OBSTACLE_ENVS
        assert _require_isaaclab_batch_support(planning_task) == env.name
