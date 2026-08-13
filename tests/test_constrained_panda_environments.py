from pathlib import Path

import numpy as np
import torch
import yaml

# Compatibility for the older NetworkX version pinned by this repository.
np.int = int
np.float = float
np.bool = bool

from torch_robotics.environments import EnvOpenDrawerShelf, EnvThreePillarsPassage
from torch_robotics.environments.env_open_drawer_shelf import (
    ADJACENT_SHELF_BOXES,
    DRAWER_CABINET_BOXES,
    OPEN_BOTTOM_DRAWER_BOXES,
)
from torch_robotics.environments.env_three_pillars_passage import THREE_PILLAR_BOXES


TENSOR_ARGS = {"device": "cpu", "dtype": torch.float32}
REPO_ROOT = Path(__file__).resolve().parents[1]
REGION_DIR = REPO_ROOT / "scripts/inference/cfgs/start_goal_regions"


def _load_yaml(name):
    with (REGION_DIR / name).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _points_from_region(region):
    axes = []
    for axis in ("x", "y", "z"):
        intervals = region["translation"][axis]
        axes.append([value for bounds in intervals for value in bounds])
    return np.asarray(
        [
            [x, y, z]
            for x in (min(axes[0]), max(axes[0]))
            for y in (min(axes[1]), max(axes[1]))
            for z in (min(axes[2]), max(axes[2]))
        ]
    )


def _assert_region_centers_outside_boxes(region, box_specs, clearance=0.0):
    points = _points_from_region(region)
    centers = np.concatenate([np.asarray(spec["centers"]) for spec in box_specs], axis=0)
    half_sizes = np.concatenate([np.asarray(spec["sizes"]) for spec in box_specs], axis=0) / 2
    for point in points:
        inside = np.all(np.abs(point[None, :] - centers) <= half_sizes + clearance, axis=-1)
        assert not inside.any(), f"region corner {point.tolist()} lies inside an obstacle"


def test_open_drawer_scene_is_independent_and_structured():
    env = EnvOpenDrawerShelf(
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        tensor_args=TENSOR_ARGS,
    )
    assert env.name == "EnvOpenDrawerShelf"
    assert [obj.name for obj in env.obj_fixed_list] == [
        "drawer_cabinet",
        "open_bottom_drawer",
        "adjacent_shelf",
    ]
    assert "table" not in " ".join(obj.name for obj in env.obj_fixed_list)
    assert len(DRAWER_CABINET_BOXES["centers"]) == 8
    assert len(OPEN_BOTTOM_DRAWER_BOXES["centers"]) == 5
    assert len(ADJACENT_SHELF_BOXES["centers"]) == 7

    drawer_front_y = OPEN_BOTTOM_DRAWER_BOXES["centers"][3][1]
    cabinet_front_y = DRAWER_CABINET_BOXES["centers"][6][1]
    assert drawer_front_y < cabinet_front_y
    bottom_bay_height = DRAWER_CABINET_BOXES["centers"][5][2]
    assert bottom_bay_height >= 0.45


def test_three_pillars_are_arm_width_and_floor_standing():
    env = EnvThreePillarsPassage(
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        tensor_args=TENSOR_ARGS,
    )
    centers = np.asarray(THREE_PILLAR_BOXES["centers"])
    sizes = np.asarray(THREE_PILLAR_BOXES["sizes"])
    assert env.name == "EnvThreePillarsPassage"
    assert [obj.name for obj in env.obj_fixed_list] == ["three_pillars"]
    assert centers.shape == sizes.shape == (3, 3)
    np.testing.assert_allclose(sizes[:, :2], 0.16)
    np.testing.assert_allclose(centers[:, 2] - sizes[:, 2] / 2, 0.0)
    assert np.all(sizes[:, 2] >= 1.2)


def test_region_files_match_scenes_and_do_not_place_ee_centers_in_boxes():
    cases = [
        (
            "EnvOpenDrawerShelf-RobotPanda-regions-to-drawer.yaml",
            "EnvOpenDrawerShelf",
            [DRAWER_CABINET_BOXES, OPEN_BOTTOM_DRAWER_BOXES, ADJACENT_SHELF_BOXES],
        ),
        (
            "EnvOpenDrawerShelf-RobotPanda-regions-drawer-to-shelf.yaml",
            "EnvOpenDrawerShelf",
            [DRAWER_CABINET_BOXES, OPEN_BOTTOM_DRAWER_BOXES, ADJACENT_SHELF_BOXES],
        ),
        (
            "EnvThreePillarsPassage-RobotPanda-regions.yaml",
            "EnvThreePillarsPassage",
            [THREE_PILLAR_BOXES],
        ),
    ]
    for filename, env_id, boxes in cases:
        cfg = _load_yaml(filename)
        assert cfg["env_id"] == env_id
        assert cfg["robot_id"] == "RobotPanda"
        assert cfg["rotate_with_environment"] is False
        assert cfg["max_sampling_attempts"] >= 100
        for region_group in ("start_regions", "goal_regions"):
            assert cfg[region_group]
            for region in cfg[region_group].values():
                _assert_region_centers_outside_boxes(region, boxes)
