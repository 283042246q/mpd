from mpd.utils.patches import numpy_monkey_patch

numpy_monkey_patch()

import argparse
import os
import pickle
from pathlib import Path
from pprint import pprint

import torch

from mpd.parametric_trajectory.trajectory_waypoints import ParametricTrajectoryWaypoints
from scripts.isaaclab.subprocess_utils import run_isaaclab_evaluator_subprocess
from torch_robotics.environments import EnvSpheres3D, GraspedObjectBox
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path
from torch_robotics.torch_utils.seed import fix_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Replay Panda planning results in a simulator backend.")
    parser.add_argument("--base_file_name", default="panda_spheres_CHOMP")
    parser.add_argument("--sim_backend", choices=("none", "isaacgym", "isaaclab"), default="isaaclab")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--isaaclab_root", default="/home/eric/IsaacLab_ori")
    parser.add_argument("--isaaclab_conda_env", default="env_isaaclab_ori")
    parser.add_argument("--isaaclab_device", default="cuda:0")
    parser.add_argument("--isaaclab_timeout_s", default=900, type=int)
    parser.add_argument("--isaaclab_action_repeat", default=4, type=int)
    return parser.parse_args()


def load_planning_results(base_file_name):
    results_path = os.path.join(
        "../../../deps/experiment_launcher/examples/", f"{base_file_name}-results_data_dict.pickle"
    )
    with open(results_path, "rb") as handle:
        return pickle.load(handle)


def build_task(tensor_args):
    env = EnvSpheres3D(tensor_args=tensor_args)
    robot = RobotPanda(
        grasped_object=GraspedObjectBox(
            attached_to_frame=RobotPanda.link_name_ee, object_collision_margin=0.05, tensor_args=tensor_args
        ),
        gripper=True,
        tensor_args=tensor_args,
    )
    parametric_trajectory = ParametricTrajectoryWaypoints(tensor_args=tensor_args)
    task = PlanningTask(
        env=env,
        robot=robot,
        parametric_trajectory=parametric_trajectory,
        ws_limits=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], **tensor_args),
        margin_for_dense_collision_checking=0.01,
        tensor_args=tensor_args,
    )
    return env, robot, task


def run_isaacgym_replay(env, robot, trajs_pos, base_file_name):
    from torch_robotics.isaac_gym_envs.motion_planning_envs import (
        MotionPlanningControllerIsaacGym,
        MotionPlanningIsaacGymEnv,
    )

    motion_planning_isaac_env = MotionPlanningIsaacGymEnv(
        env,
        robot,
        asset_root=get_robot_path().as_posix(),
        robot_asset_file=robot.robot_urdf_file.replace(get_robot_path().as_posix() + "/", ""),
        num_envs=trajs_pos.shape[1],
        all_robots_in_one_env=True,
        show_viewer=True,
        sync_viewer_with_real_time=False,
        viewer_time_between_steps=0.1,
        render_camera_global=True,
        color_robots=False,
        draw_goal_configuration=True,
        draw_collision_spheres=False,
        draw_contact_forces=False,
        draw_end_effector_frame=True,
        draw_end_effector_path=False,
        camera_global_from_top=True if env.dim == 2 else False,
        add_ground_plane=False if env.dim == 2 else True,
    )
    motion_planning_controller = MotionPlanningControllerIsaacGym(motion_planning_isaac_env)
    return motion_planning_controller.execute_trajectories(
        trajs_pos,
        q_pos_starts=trajs_pos[0],
        q_pos_goal=trajs_pos[-1][0],
        n_first_steps=30,
        n_last_steps=30,
        stop_robot_if_in_contact=False,
        make_video=True,
        video_duration=5.0,
        video_path=base_file_name + "isaac-controller-position.mp4",
        make_gif=False,
    )


def run_isaaclab_replay(trajs_pos, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = output_dir / f"{args.base_file_name}-isaaclab-trajectories.pt"
    statistics_path = output_dir / f"{args.base_file_name}-isaaclab-statistics.json"
    log_path = output_dir / f"{args.base_file_name}-isaaclab.log"
    video_path = output_dir / f"{args.base_file_name}-isaaclab-controller-position.mp4"

    torch.save(
        {
            "q_trajs_pos": trajs_pos.detach().cpu(),
            "q_pos_starts": trajs_pos[0].detach().cpu(),
            "q_pos_goal": trajs_pos[-1][0].detach().cpu(),
            "robot_name": "panda",
            "env_name": "EnvSpheres3D",
            "dt": 0.1,
        },
        trajectories_path,
    )
    return run_isaaclab_evaluator_subprocess(
        trajectories_path=trajectories_path,
        statistics_path=statistics_path,
        log_path=log_path,
        isaaclab_root=args.isaaclab_root,
        isaaclab_conda_env=args.isaaclab_conda_env,
        isaaclab_device=args.isaaclab_device,
        isaaclab_action_repeat=args.isaaclab_action_repeat,
        isaaclab_timeout_s=args.isaaclab_timeout_s,
        make_video=True,
        video_path=video_path,
    )


def main():
    args = parse_args()
    fix_random_seed(0)
    tensor_args = {"device": "cpu", "dtype": torch.float32}

    results_planning = load_planning_results(args.base_file_name)
    env, robot, task = build_task(tensor_args)

    trajs_iters = results_planning["trajs_iters_free"]
    trajs_pos = task.get_position(trajs_iters[-1]).movedim(1, 0)
    results_planning["dt"] = 0.1

    if args.sim_backend == "none":
        print("sim_backend=none; skipping simulator replay.")
        return
    if args.sim_backend == "isaacgym":
        isaac_statistics = run_isaacgym_replay(env, robot, trajs_pos, args.base_file_name)
    else:
        isaac_statistics = run_isaaclab_replay(trajs_pos, args)

    print("-----------------")
    print("isaac_statistics:")
    pprint(isaac_statistics)
    print("-----------------")


if __name__ == "__main__":
    main()
