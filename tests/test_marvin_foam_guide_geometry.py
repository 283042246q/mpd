"""Offline gates for the guidance-only Foam Marvin/Pika collision profile."""

from pathlib import Path

import numpy as np
import torch
import yaml
from dotmap import DotMap


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT / "mpd/torch_robotics/torch_robotics/data/configs/marvin"
)
PRODUCTION = CONFIG_ROOT / "pika"
GUIDE = CONFIG_ROOT / "pika_foam_guide"
GUIDE_200_60 = CONFIG_ROOT / "pika_foam_marvin_200_pika_60_guide"
PIKA_LINKS = {
    f"{side}_{name}"
    for side in ("left", "right")
    for name in (
        "pika_adaptor_link",
        "gripper_base_link",
        "gripper_left_link",
        "gripper_right_link",
    )
}
ARM_LINKS = {
    f"{name}_{side}"
    for side in ("L", "R")
    for name in (
        "Base",
        "Link1",
        "Link2",
        "Link3",
        "Link4",
        "Link5",
        "Link6",
        "Link7",
    )
}


def _yaml(path):
    return yaml.safe_load(path.read_text())


def _sphere_count(config):
    return sum(len(value) for key, value in config.items() if key != "self_collision")


def test_foam_profile_is_guidance_only_and_does_not_modify_arm_geometry():
    production = _yaml(PRODUCTION / "collision_spheres.yaml")
    guide = _yaml(GUIDE / "collision_spheres.yaml")
    manifest = _yaml(GUIDE / "guide_geometry_manifest.yaml")

    assert manifest["schema"] == "marvin_foam_guide_geometry/v1"
    assert manifest["profile"] == "foam_pika_100"
    assert manifest["guidance_only"] is True
    assert manifest["production_validator_profile"] == "pika"
    assert manifest["method"] == "grid"
    assert manifest["pika_sphere_count"] == 96
    assert 90 <= manifest["pika_sphere_count"] <= 110
    assert _sphere_count(production) == 1035
    assert _sphere_count(guide) == manifest["total_sphere_count"] == 489
    assert sum(len(guide[name]) for name in PIKA_LINKS) == 96
    for name in set(production) - PIKA_LINKS - {"self_collision"}:
        assert guide[name] == production[name]
    assert guide["self_collision"] == production["self_collision"]


def test_foam_profile_reports_conservative_sampled_surface_coverage():
    manifest = _yaml(GUIDE / "guide_geometry_manifest.yaml")
    assert set(manifest["links"]) == PIKA_LINKS
    for report in manifest["links"].values():
        coverage = report["coverage"]
        assert coverage["surface_samples"] == 20000
        assert coverage["uncovered_ratio"] == 0.0
        assert coverage["maximum_uncovered_m"] == 0.0
        assert report["preprocessing"]["processed_faces"] > 0
        assert report["preprocessing"]["processed_faces"] < report["preprocessing"]["input_faces"]


def test_foam_marvin_200_pika_60_profile_has_requested_budgets():
    production = _yaml(PRODUCTION / "collision_spheres.yaml")
    guide = _yaml(GUIDE_200_60 / "collision_spheres.yaml")
    manifest = _yaml(GUIDE_200_60 / "guide_geometry_manifest.yaml")

    assert manifest["schema"] == "marvin_foam_guide_geometry/v2"
    assert manifest["profile"] == "foam_marvin_200_pika_60"
    assert manifest["guidance_only"] is True
    assert manifest["production_validator_profile"] == "pika"
    assert manifest["marvin_arm_sphere_count"] == 199
    assert manifest["pika_sphere_count"] == 62
    assert manifest["fixed_sphere_count"] == 40
    assert manifest["total_sphere_count"] == _sphere_count(guide) == 301
    assert sum(len(guide[name]) for name in ARM_LINKS) == 199
    assert sum(len(guide[name]) for name in PIKA_LINKS) == 62
    assert guide["base_link"] == production["base_link"]
    assert guide["column_link"] == production["column_link"]
    assert guide["self_collision"] == production["self_collision"]

    assert set(manifest["links"]) == ARM_LINKS | PIKA_LINKS
    for report in manifest["links"].values():
        coverage = report["coverage"]
        assert coverage["surface_samples"] == 20000
        assert coverage["uncovered_ratio"] == 0.0
        assert coverage["maximum_uncovered_m"] == 0.0


def test_reduced_robot_preserves_kinematics_but_reduces_pairs():
    from torch_robotics.robots.robot_marvin_bimanual import RobotMarvinBimanual

    tensor_args = {"device": "cpu", "dtype": torch.float64}
    production = RobotMarvinBimanual(tensor_args=tensor_args)
    guide = RobotMarvinBimanual(
        collision_geometry_profile="foam_pika_100", tensor_args=tensor_args
    )
    assert production.collision_geometry_profile == "production"
    assert guide.collision_geometry_profile == "foam_pika_100"
    assert production.model_hash == guide.model_hash
    assert production.joint_names == guide.joint_names
    assert production.collision_geometry_sphere_count == 1035
    assert guide.collision_geometry_sphere_count == 489
    assert len(guide.link_self_collision_tuples) < len(production.link_self_collision_tuples)
    assert guide.guide_geometry_manifest["guidance_only"] is True
    assert production.collision_geometry_hash != guide.collision_geometry_hash

    generator = torch.Generator().manual_seed(29)
    q = torch.rand(3, 14, generator=generator, dtype=torch.float64) - 0.5
    torch.testing.assert_close(production.fk_left(q), guide.fk_left(q))
    torch.testing.assert_close(production.fk_right(q), guide.fk_right(q))


def test_foam_marvin_200_pika_60_robot_preserves_kinematics():
    from torch_robotics.robots.robot_marvin_bimanual import RobotMarvinBimanual

    tensor_args = {"device": "cpu", "dtype": torch.float64}
    production = RobotMarvinBimanual(tensor_args=tensor_args)
    guide = RobotMarvinBimanual(
        collision_geometry_profile="foam_marvin_200_pika_60",
        tensor_args=tensor_args,
    )
    assert guide.collision_geometry_sphere_count == 301
    assert len(guide.link_self_collision_tuples) < len(production.link_self_collision_tuples)
    assert guide.guide_geometry_manifest["guidance_only"] is True

    generator = torch.Generator().manual_seed(31)
    q = torch.rand(3, 14, generator=generator, dtype=torch.float64) - 0.5
    torch.testing.assert_close(production.fk_left(q), guide.fk_left(q))
    torch.testing.assert_close(production.fk_right(q), guide.fk_right(q))


def test_reduced_guide_geometry_has_an_independent_switch_and_new_profile_default():
    from mpd.inference.guidance_config import resolve_collision_optimization_config

    default = resolve_collision_optimization_config({})["reduced_guide_geometry"]
    assert default == {
        "enabled": False,
        "profile": "foam_marvin_200_pika_60",
    }
    disabled = resolve_collision_optimization_config(
        {"collision_optimization": {"reduced_guide_geometry": {"enabled": False}}}
    )["reduced_guide_geometry"]
    assert disabled["enabled"] is False
    assert disabled["profile"] == "foam_marvin_200_pika_60"


def test_reduced_profile_parent_bounds_contain_every_guide_sphere():
    for guide_dir in (GUIDE, GUIDE_200_60):
        spheres = _yaml(guide_dir / "collision_spheres.yaml")
        bounds = _yaml(guide_dir / "collision_parent_bounds.yaml")["parent_bounds"]
        assert set(bounds) == set(spheres) - {"self_collision"}
        for name, entries in bounds.items():
            fine = np.asarray(spheres[name], dtype=float)
            assert sorted(
                index for entry in entries for index in entry["source_sphere_indices"]
            ) == list(range(len(fine)))
            covered = np.zeros(len(fine), dtype=bool)
            for entry in entries:
                indices = entry["source_sphere_indices"]
                values = fine[indices]
                covered[indices] |= (
                    np.linalg.norm(values[:, :3] - entry["center"], axis=1)
                    + values[:, 3]
                    <= entry["radius"] + 1e-12
                )
            assert covered.all(), f"{guide_dir.name}:{name}"


def test_reduced_geometry_is_wired_only_into_collision_guide_costs():
    from mpd.bimanual.cost_guide import BimanualCostGuideManagerParametricTrajectory
    from mpd.bimanual.planning_task import BimanualPlanningTask
    from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline
    from torch_robotics.environments.env_warehouse_marvin_bimanual import (
        EnvWarehouseMarvinBimanual,
    )
    from torch_robotics.robots.robot_marvin_bimanual import RobotMarvinBimanual

    class Dataset:
        context_ee_goal_pose = True
        context_ee_goal_pose_bimanual = True

        @staticmethod
        def unnormalize_control_points(value):
            return value

        @staticmethod
        def grad_unnormalized_wrt_control_points_normalized(value):
            return torch.ones_like(value)

    tensor_args = {"device": "cpu", "dtype": torch.float32}
    production_robot = RobotMarvinBimanual(tensor_args=tensor_args)
    environment = EnvWarehouseMarvinBimanual(tensor_args=tensor_args)
    trajectory = ParametricTrajectoryBspline(
        n_control_points=8,
        degree=3,
        remove_outer_control_points=False,
        keep_last_control_point=False,
        num_T_pts=8,
        tensor_args=tensor_args,
    )
    task = BimanualPlanningTask(
        parametric_trajectory=trajectory,
        env=environment,
        robot=production_robot,
        tensor_args=tensor_args,
    )
    args = DotMap(
        {
            "project_gradient_hierarchy": False,
            "gradient_pruning": {"enabled": False},
            "costs": {
                "CostTaskSpaceCollisionObjects": {
                    "weight": 1.0,
                    "use_only_on_extra_objects": False,
                },
                "CostTaskSpaceCollisionSelfLeftArm": {"weight": 1.0},
                "CostTaskSpaceCollisionSelfRightArm": {"weight": 1.0},
                "CostTaskSpaceCollisionInterArm": {"weight": 5.0},
                "CostJointSpaceJointLimits": {"weight": 1.0},
            },
        }
    )
    manager = BimanualCostGuideManagerParametricTrajectory(
        task, Dataset(), args, tensor_args=tensor_args
    )

    assert task.robot is production_robot
    assert manager.validation_robot is production_robot
    assert manager.robot is manager.guide_collision_robot
    assert manager.robot.collision_geometry_sphere_count == 301
    for key in (
        "CostTaskSpaceCollisionObjects",
        "CostTaskSpaceCollisionSelfLeftArm",
        "CostTaskSpaceCollisionSelfRightArm",
        "CostTaskSpaceCollisionInterArm",
    ):
        assert manager.costs[key].cost.robot is manager.guide_collision_robot
    assert manager.costs.CostJointSpaceJointLimits.cost.robot is production_robot

    args.collision_optimization = {
        "reduced_guide_geometry": {"enabled": False}
    }
    production_manager = BimanualCostGuideManagerParametricTrajectory(
        task, Dataset(), args, tensor_args=tensor_args
    )
    assert production_manager.guide_collision_robot is None
    assert production_manager.robot is production_robot
    for key in (
        "CostTaskSpaceCollisionObjects",
        "CostTaskSpaceCollisionSelfLeftArm",
        "CostTaskSpaceCollisionSelfRightArm",
        "CostTaskSpaceCollisionInterArm",
    ):
        assert production_manager.costs[key].cost.robot is production_robot
