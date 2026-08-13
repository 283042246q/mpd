#!/usr/bin/env python3
"""Validate constrained Panda region tasks with full-body IK and OMPL paths.

Run from the repository root:

    conda run --no-capture-output -n mpd-splines-public-cu128 \
      python scripts/inference/validate_constrained_panda_regions.py
"""

import argparse
from pathlib import Path

import numpy as np

# Compatibility for the older NetworkX version pinned by this repository.
np.int = int
np.float = float
np.bool = bool

import torch
import yaml

from mpd.inference.inference import EvaluationSamplesGenerator
from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline
from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGION_FILES = (
    REPO_ROOT / "scripts/inference/cfgs/start_goal_regions/EnvOpenDrawerShelf-RobotPanda-regions-to-drawer.yaml",
    REPO_ROOT / "scripts/inference/cfgs/start_goal_regions/EnvOpenDrawerShelf-RobotPanda-regions-drawer-to-shelf.yaml",
    REPO_ROOT / "scripts/inference/cfgs/start_goal_regions/EnvThreePillarsPassage-RobotPanda-regions.yaml",
)
TENSOR_ARGS = {"device": "cpu", "dtype": torch.float32}


def _minimum_clearances(task, q_path):
    """Return object and self clearances after subtracting collision-sphere radii."""
    q_path = torch.as_tensor(q_path, **TENSOR_ARGS)
    link_positions = task.robot.fk_map_collision(q_path)

    object_distances = task.df_collision_objects.object_signed_distances(link_positions)
    object_clearance = object_distances - task.robot.link_collision_spheres_radii

    self_clearance = torch.tensor(float("inf"), **TENSOR_ARGS)
    if task.df_collision_self is not None:
        self_distances = task.df_collision_self.compute_embodiment_signed_distances(q_path, link_positions)
        self_clearance = self_distances.min()

    return float(object_clearance.min().item()), float(self_clearance.item())


def validate_region_file(region_path, samples, plans, planner_time, interpolate_num):
    with region_path.open("r", encoding="utf-8") as stream:
        region_cfg = yaml.safe_load(stream)

    env_class = getattr(environments, region_cfg["env_id"])
    robot_class = getattr(robots, region_cfg["robot_id"])
    env = env_class(
        precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False,
        tensor_args=TENSOR_ARGS,
    )
    robot = robot_class(gripper=True, tensor_args=TENSOR_ARGS)
    trajectory = ParametricTrajectoryBspline(num_T_pts=128, tensor_args=TENSOR_ARGS)
    task = PlanningTask(
        env=env,
        robot=robot,
        parametric_trajectory=trajectory,
        min_distance_robot_env=0.02,
        obstacle_cutoff_margin=0.07,
        margin_for_dense_collision_checking=0.0,
        tensor_args=TENSOR_ARGS,
    )
    sampler = EvaluationSamplesGenerator(
        planning_task=task,
        train_subset=None,
        val_subset=None,
        start_goal_source="regions",
        start_goal_regions_path=str(region_path),
        tensor_args=TENSOR_ARGS,
    )

    sampled_pairs = []
    sampling_attempts = []
    try:
        for sample_idx in range(samples):
            q_start, q_goal, _ = sampler.get_data_sample(sample_idx)
            sampled_pairs.append((q_start, q_goal))
            sampling_attempts.append(int(sampler.last_sample_metadata["sampling_attempt"]))

        planned = 0
        minimum_object_clearance = float("inf")
        minimum_self_clearance = float("inf")
        interface = sampler.generate_data_ompl_worker.pbompl_interface
        interface.si.setStateValidityCheckingResolution(0.002)
        for q_start, q_goal in sampled_pairs:
            result = interface.plan_start_goal(
                q_start.detach().cpu().numpy().astype(np.float64),
                q_goal.detach().cpu().numpy().astype(np.float64),
                allowed_time=planner_time,
                simplify_path=True,
                interpolate_num=interpolate_num,
                fit_bspline=False,
            )
            if not result["success"]:
                continue

            q_path = result["sol_path"]
            pb_valid = all(interface.is_state_valid(q, check_bounds=True) for q in q_path)
            torch_collision = task.compute_collision(
                torch.as_tensor(q_path, **TENSOR_ARGS),
                margin=task.margin_for_dense_collision_checking,
            )
            if not pb_valid or bool(torch_collision.any().item()):
                print(
                    f"REJECT {region_path.name}: dense validation failed "
                    f"(pybullet_valid={pb_valid}, torch_collision={bool(torch_collision.any().item())})"
                )
                continue

            object_clearance, self_clearance = _minimum_clearances(task, q_path)
            minimum_object_clearance = min(minimum_object_clearance, object_clearance)
            minimum_self_clearance = min(minimum_self_clearance, self_clearance)
            planned += 1
            if planned >= plans:
                break

        if planned < plans:
            raise RuntimeError(f"{region_path.name}: planned {planned}/{plans} requested paths.")

        print(
            f"PASS {region_path.name}: samples={samples}, plans={planned}, "
            f"max_sampling_attempts_used={max(sampling_attempts)}, "
            f"min_object_clearance={minimum_object_clearance:.4f} m, "
            f"min_self_clearance={minimum_self_clearance:.4f} m"
        )
    finally:
        sampler.generate_data_ompl_worker.terminate()
        robot.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("region_files", nargs="*", type=Path, default=list(DEFAULT_REGION_FILES))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--plans", type=int, default=2)
    parser.add_argument("--planner-time", type=float, default=20.0)
    parser.add_argument("--interpolate-num", type=int, default=256)
    args = parser.parse_args()

    if args.samples < 1 or args.plans < 1 or args.plans > args.samples:
        parser.error("require samples >= plans >= 1")

    np.random.seed(7)
    torch.manual_seed(7)
    for region_file in args.region_files:
        validate_region_file(
            region_file.resolve(),
            samples=args.samples,
            plans=args.plans,
            planner_time=args.planner_time,
            interpolate_num=args.interpolate_num,
        )


if __name__ == "__main__":
    main()
