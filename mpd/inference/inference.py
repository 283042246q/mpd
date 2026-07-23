import os
from math import ceil
from pathlib import Path

import numpy as np
import torch
from dotmap import DotMap
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from mpd.models import GaussianDiffusionModel, guide_gradient_steps, CVAEModel
from mpd.utils.loaders import load_params_from_yaml
from pb_ompl.pb_ompl import add_box, fit_bspline_to_path
from scripts.generate_data.generate_trajectories import GenerateDataOMPL, get_random_pose_from_region
from mpd.inference.cost_guides import CostGuideManagerParametricTrajectory, NoCostException
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import (
    to_numpy,
    freeze_torch_model_params,
    to_torch,
    dict_to_device,
    DEFAULT_TENSOR_ARGS,
)
from torch_robotics.trajectory.metrics import compute_smoothness, compute_ee_pose_errors, compute_path_length


DEFAULT_BEST_TRAJECTORY_WEIGHTS = {
    "ee_position": 0.35,
    "ee_orientation": 0.20,
    "path_length": 0.15,
    "smoothness": 0.15,
    "velocity_limit_utilization": 0.075,
    "acceleration_limit_utilization": 0.075,
}

DEFAULT_BEST_TRAJECTORY_METRIC_SCALES = {
    "ee_position": 0.02,
    "ee_orientation": 2.0,
    "path_length": None,
    "smoothness": None,
    "velocity_limit_utilization": 1.0,
    "acceleration_limit_utilization": 1.0,
}


def _config_value(config, key, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_best_trajectory_metric(values, configured_scale=None):
    if configured_scale is None or float(configured_scale) <= 0.0:
        scale = values.detach().median()
    else:
        scale = torch.as_tensor(configured_scale, dtype=values.dtype, device=values.device)
    scale = torch.clamp(scale, min=torch.finfo(values.dtype).eps)
    return values / scale, scale


class EvaluationSamplesGenerator:
    """
    Get start and goal joint positions from a dataset, an explicit states file, or workspace pose regions.
    """

    def __init__(
        self,
        planning_task,
        train_subset,
        val_subset,
        selection_start_goal="training",  # training, validation
        start_goal_source="auto",  # auto, dataset, states_file, regions
        start_goal_regions_path=None,
        start_goal_max_sampling_attempts=100,
        rotation_z_axis_deg=0.0,
        grasped_object=None,
        tensor_args=DEFAULT_TENSOR_ARGS,
        debug=False,
        render_pybullet=False,
        min_distance_q_pos_start_goal=None,
        **kwargs,
    ):
        self.tensor_args = tensor_args
        self.debug = debug
        self.planning_task = planning_task

        self.selection_start_goal = selection_start_goal
        self.start_goal_source = str(start_goal_source).lower()
        if self.start_goal_source == "auto":
            self.start_goal_source = "dataset" if selection_start_goal in {"training", "validation"} else "states_file"
        valid_start_goal_sources = {"dataset", "states_file", "regions"}
        if self.start_goal_source not in valid_start_goal_sources:
            raise ValueError(
                f"Unknown start_goal_source={self.start_goal_source!r}. "
                f"Expected one of {sorted(valid_start_goal_sources)}."
            )

        self.select_start_goal_from_file = None
        self.start_goal_regions = None
        self.dataset_subset = None
        self.idxs_dataset_subset = None
        self.train_subset = train_subset
        self.val_subset = val_subset
        if self.start_goal_source == "dataset":
            if selection_start_goal == "training":
                self.dataset_subset = train_subset
            elif selection_start_goal == "validation":
                self.dataset_subset = val_subset
            else:
                raise ValueError(
                    "selection_start_goal must be 'training' or 'validation' when start_goal_source='dataset'."
                )
            self.idxs_dataset_subset = np.random.permutation(len(self.dataset_subset))
        elif self.start_goal_source == "states_file":
            self.select_start_goal_from_file = load_params_from_yaml(selection_start_goal)
        else:
            if not start_goal_regions_path:
                raise ValueError("start_goal_regions_path is required when start_goal_source='regions'.")
            start_goal_regions_path = os.path.expandvars(os.path.expanduser(str(start_goal_regions_path)))
            self.start_goal_regions = load_params_from_yaml(start_goal_regions_path)
            if not self.start_goal_regions.get("start_regions"):
                raise ValueError(f"No start_regions defined in {start_goal_regions_path}.")
            if not self.start_goal_regions.get("goal_regions"):
                raise ValueError(f"No goal_regions defined in {start_goal_regions_path}.")

            configured_env_id = self.start_goal_regions.get("env_id")
            current_env_id = getattr(planning_task.env, "name", type(planning_task.env).__name__)
            if configured_env_id and not current_env_id.startswith(configured_env_id):
                raise ValueError(
                    f"Region config env_id={configured_env_id!r} does not match current environment "
                    f"{current_env_id!r}."
                )
            configured_robot_id = self.start_goal_regions.get("robot_id")
            current_robot_id = type(planning_task.robot).__name__
            if configured_robot_id and configured_robot_id != current_robot_id:
                raise ValueError(
                    f"Region config robot_id={configured_robot_id!r} does not match current robot "
                    f"{current_robot_id!r}."
                )

        self.start_goal_max_sampling_attempts = int(start_goal_max_sampling_attempts)
        if self.start_goal_regions is not None:
            self.start_goal_max_sampling_attempts = int(
                self.start_goal_regions.get("max_sampling_attempts", self.start_goal_max_sampling_attempts)
            )
        if self.start_goal_max_sampling_attempts < 1:
            raise ValueError("start_goal_max_sampling_attempts must be >= 1.")

        self.rotation_z_axis_deg = float(rotation_z_axis_deg)
        rotation_z_axis_rad = np.deg2rad(self.rotation_z_axis_deg)
        self.environment_rotation = np.eye(4)
        self.environment_rotation[:2, :2] = np.array(
            [
                [np.cos(rotation_z_axis_rad), -np.sin(rotation_z_axis_rad)],
                [np.sin(rotation_z_axis_rad), np.cos(rotation_z_axis_rad)],
            ]
        )
        self.rotate_regions_with_environment = True
        if self.start_goal_regions is not None:
            self.rotate_regions_with_environment = bool(self.start_goal_regions.get("rotate_with_environment", True))

        self.min_distance_q_pos_start_goal = float(min_distance_q_pos_start_goal or 0.0)
        self.last_sample_metadata = {}

        # OMPL worker to generate random start and goal joint positions
        self.generate_data_ompl_worker = GenerateDataOMPL(
            None,
            None,
            env_tr=planning_task.env,
            robot_tr=planning_task.robot,
            gripper=True,
            grasped_object=grasped_object,
            min_distance_robot_env=planning_task.min_distance_robot_env,
            tensor_args=tensor_args,
            pybullet_mode="GUI" if debug or render_pybullet else "DIRECT",
            debug=debug or render_pybullet,
        )

        self.ee_markers_ids = []

    def _rotate_region_pose(self, ee_pose):
        if not self.rotate_regions_with_environment:
            return ee_pose
        return self.environment_rotation @ ee_pose

    def _solve_region_ik(self, ee_pose, state_name, region_id):
        try:
            return self.generate_data_ompl_worker.pbompl_interface.get_state_not_in_collision(
                ee_pose_target=ee_pose,
                debug=self.debug,
            )
        except RuntimeError as error:
            target_position = np.round(ee_pose[:3, 3], decimals=4).tolist()
            raise RuntimeError(f"{state_name} region {region_id} at position {target_position}: {error}") from error

    def _sample_region_candidate(self):
        start_region_ids = list(self.start_goal_regions["start_regions"].keys())
        goal_region_ids = list(self.start_goal_regions["goal_regions"].keys())
        start_region_id = str(np.random.choice(start_region_ids))
        goal_region_id = str(np.random.choice(goal_region_ids))

        ee_pose_start = get_random_pose_from_region(self.start_goal_regions["start_regions"][start_region_id])
        ee_pose_goal = get_random_pose_from_region(self.start_goal_regions["goal_regions"][goal_region_id])
        ee_pose_start = self._rotate_region_pose(ee_pose_start)
        ee_pose_goal = self._rotate_region_pose(ee_pose_goal)

        q_pos_start = self._solve_region_ik(ee_pose_start, "start", start_region_id)
        q_pos_goal = self._solve_region_ik(ee_pose_goal, "goal", goal_region_id)

        metadata = {
            "source": "regions",
            "start_region_id": start_region_id,
            "goal_region_id": goal_region_id,
            "rotation_z_axis_deg": self.rotation_z_axis_deg,
            "rotate_with_environment": self.rotate_regions_with_environment,
            "ee_pose_start": ee_pose_start.tolist(),
            "ee_pose_goal": ee_pose_goal.tolist(),
        }
        return q_pos_start, q_pos_goal, ee_pose_goal[:3, :4], metadata

    def _get_candidate(self, idx):
        if self.start_goal_source == "regions":
            return self._sample_region_candidate()

        if self.start_goal_source == "states_file":
            sample_idx = idx % len(self.select_start_goal_from_file)
            sample = self.select_start_goal_from_file[sample_idx]
            q_pos_start = sample["q_pos_start"]
            q_pos_goal = sample["q_pos_goal"]
            ee_pose_goal = np.asarray(sample["ee_pose_goal"]).reshape(3, 4)
            return (
                q_pos_start,
                q_pos_goal,
                ee_pose_goal,
                {
                    "source": "states_file",
                    "sample_index": int(sample_idx),
                },
            )

        sample_idx = self.idxs_dataset_subset[idx % len(self.idxs_dataset_subset)]
        input_data_one_sample = self.dataset_subset[sample_idx]
        q_pos_start = input_data_one_sample[self.dataset_subset.dataset.field_key_q_start]
        q_pos_goal = input_data_one_sample[self.dataset_subset.dataset.field_key_q_goal]
        ee_pose_goal = input_data_one_sample[self.dataset_subset.dataset.field_key_context_ee_goal_pose]
        return (
            q_pos_start,
            q_pos_goal,
            ee_pose_goal,
            {
                "source": "dataset",
                "selection": self.selection_start_goal,
                "sample_index": int(sample_idx),
            },
        )

    def _state_invalid_reason(self, q_pos, state_name):
        q_pos = to_torch(q_pos, **self.tensor_args)
        q_pos_np = to_numpy(q_pos)
        if not self.generate_data_ompl_worker.pbompl_interface.is_state_valid(q_pos_np, check_bounds=True):
            return f"{state_name} is invalid in PyBullet."

        if torch.any(q_pos < self.planning_task.robot.q_pos_min) or torch.any(
            q_pos > self.planning_task.robot.q_pos_max
        ):
            return f"{state_name} is outside the Torch robot joint limits."

        collision = self.planning_task.compute_collision(
            q_pos,
            margin=self.planning_task.margin_for_dense_collision_checking,
        )
        if bool(torch.as_tensor(collision).any().item()):
            return f"{state_name} is in collision in the Torch collision field."
        return None

    def get_data_sample(self, idx, **kwargs):
        last_failure = None
        for attempt in range(self.start_goal_max_sampling_attempts):
            try:
                q_pos_start, q_pos_goal, ee_pose_goal, metadata = self._get_candidate(idx + attempt)
            except Exception as error:
                last_failure = f"Sampling/IK failed: {error}"
                print(f"{last_failure} Resampling...")
                continue

            q_pos_start = to_torch(q_pos_start, **self.tensor_args)
            q_pos_goal = to_torch(q_pos_goal, **self.tensor_args)
            ee_pose_goal = to_torch(ee_pose_goal, **self.tensor_args).view(3, 4)

            last_failure = self._state_invalid_reason(q_pos_start, "Start state")
            if last_failure is None:
                last_failure = self._state_invalid_reason(q_pos_goal, "Goal state")
            if (
                last_failure is None
                and torch.linalg.norm(q_pos_goal - q_pos_start) < self.min_distance_q_pos_start_goal
            ):
                last_failure = (
                    f"Start and goal states are closer than "
                    f"{self.min_distance_q_pos_start_goal:.3f} in joint space."
                )

            if last_failure is not None:
                print(f"{last_failure} Resampling...")
                continue

            metadata["sampling_attempt"] = attempt + 1
            self.last_sample_metadata = metadata
            if self.start_goal_source == "regions":
                print(
                    f"Sampled start region {metadata['start_region_id']} and goal region "
                    f"{metadata['goal_region_id']} with rotation_z_axis_deg="
                    f"{self.rotation_z_axis_deg:.3f}."
                )
            return q_pos_start, q_pos_goal, ee_pose_goal

        raise RuntimeError(
            f"Could not obtain a valid start-goal sample after "
            f"{self.start_goal_max_sampling_attempts} attempts. Last failure: {last_failure}"
        )

    def add_start_goal_marker(self, q_pos_start, q_pos_goal=None, ee_pose_goal=None, **kwargs):
        # remove markers first
        if self.ee_markers_ids:
            for marker_id in self.ee_markers_ids:
                self.generate_data_ompl_worker.pybullet_client.removeBody(marker_id)
            self.ee_markers_ids = []

        # adds a box to the pybullet environment to visualize the start and goal states
        ee_pose_start_np = self.generate_data_ompl_worker.pbompl_interface.get_ee_pose(to_numpy(q_pos_start))
        box_id = add_box(
            self.generate_data_ompl_worker.pybullet_client,
            ee_pose_start_np[0],
            [0.02] * 3,
            orientation=ee_pose_start_np[1],
            color=(0.0, 0.0, 1.0, 1.0),
        )
        self.ee_markers_ids.append(box_id)

        if ee_pose_goal is not None:
            ee_pose_goal_np = to_numpy(ee_pose_goal)
            ee_pose_goal_np = (ee_pose_goal_np[:3, -1], Rotation.from_matrix(ee_pose_goal_np[:3, :3]).as_quat())
        elif q_pos_goal is not None:
            ee_pose_goal_np = self.generate_data_ompl_worker.pbompl_interface.get_ee_pose(to_numpy(q_pos_goal))
        else:
            return
        box_id = add_box(
            self.generate_data_ompl_worker.pybullet_client,
            ee_pose_goal_np[0],
            [0.02] * 3,
            orientation=ee_pose_goal_np[1],
            color=(1.0, 0.0, 0.0, 1.0),
        )
        self.ee_markers_ids.append(box_id)


def render_results(
    args_inference,
    planning_task,
    q_pos_start,
    q_pos_goal,
    results_single_plan,
    idx,
    results_dir,
    render_joint_space_time_iters=False,
    render_joint_space_env_iters=False,
    render_planning_env_robot_opt_iters=False,
    render_planning_env_robot_trajectories=False,
    debug=False,
    **kwargs,
):
    base_file_name = Path(os.path.basename(__file__)).stem

    if results_single_plan.q_trajs_pos_best is not None:
        q_pos_traj_best = results_single_plan.q_trajs_pos_best
        q_vel_traj_best = results_single_plan.q_trajs_vel_best
        q_acc_traj_best = results_single_plan.q_trajs_acc_best
    else:
        q_pos_traj_best = None
        q_vel_traj_best = None
        q_acc_traj_best = None

    q_vel_start = results_single_plan.get("q_vel_start", torch.zeros_like(q_pos_start))
    q_vel_goal = results_single_plan.get("q_vel_goal", torch.zeros_like(q_pos_goal))
    q_acc_start = results_single_plan.get("q_acc_start", torch.zeros_like(q_pos_start))
    q_acc_goal = results_single_plan.get("q_acc_goal", torch.zeros_like(q_pos_goal))

    if render_joint_space_time_iters:
        planning_task.animate_opt_iters_joint_space_state(
            q_pos_trajs=results_single_plan.q_trajs_pos_iters,
            q_vel_trajs=results_single_plan.q_trajs_vel_iters,
            q_acc_trajs=results_single_plan.q_trajs_acc_iters,
            pos_start_state=q_pos_start,
            pos_goal_state=q_pos_goal,
            vel_start_state=q_vel_start,
            vel_goal_state=q_vel_goal,
            acc_start_state=q_acc_start,
            acc_goal_state=q_acc_goal,
            q_pos_traj_best=q_pos_traj_best,
            q_vel_traj_best=q_vel_traj_best,
            q_acc_traj_best=q_acc_traj_best,
            video_filepath=os.path.join(results_dir, f"{base_file_name}-joint_space-time-opt-iters-{idx:03d}.mp4"),
            n_frames=max((2, len(results_single_plan.q_trajs_pos_iters))),
            anim_time=args_inference.trajectory_duration,
            set_joint_limits=True,
            set_joint_vel_limits=True,
            set_joint_acc_limits=True,
            filter_joint_limits_vel_acc=True,
        )

        # reconstructed control points and trajectories at each diffusion iteration
        if results_single_plan.q_trajs_pos_recon_iters is not None:
            planning_task.animate_opt_iters_joint_space_state(
                q_pos_trajs=results_single_plan.q_trajs_pos_recon_iters,
                q_vel_trajs=results_single_plan.q_trajs_vel_recon_iters,
                q_acc_trajs=results_single_plan.q_trajs_acc_recon_iters,
                pos_start_state=q_pos_start,
                pos_goal_state=q_pos_goal,
                vel_start_state=q_vel_start,
                vel_goal_state=q_vel_goal,
                acc_start_state=q_acc_start,
                acc_goal_state=q_acc_goal,
                q_pos_traj_best=None,
                video_filepath=os.path.join(
                    results_dir, f"{base_file_name}-joint_space-time-opt-iters-recon-{idx:03d}.mp4"
                ),
                n_frames=max((2, len(results_single_plan.q_trajs_pos_iters))),
                anim_time=args_inference.trajectory_duration,
                set_joint_limits=True,
                set_joint_vel_limits=True,
                set_joint_acc_limits=True,
                filter_joint_limits_vel_acc=True,
            )

    if render_joint_space_env_iters:
        # visualize trajectories in the joint space
        planning_task.animate_opt_iters_joint_space_env(
            trajs_pos=results_single_plan.q_trajs_pos_iters,
            start_state=q_pos_start,
            goal_state=q_pos_goal,
            traj_pos_best=results_single_plan.q_trajs_pos_best,
            control_points=results_single_plan.control_points_iters,
            video_filepath=os.path.join(results_dir, f"{base_file_name}-joint_space-env-opt-iters-{idx:03d}.mp4"),
            n_frames=max((2, len(results_single_plan.q_trajs_pos_iters))),
            anim_time=args_inference.trajectory_duration,
            filter_joint_limits_vel_acc=True,
        )

    if render_planning_env_robot_opt_iters:
        # visualize in the planning environment
        planning_task.animate_opt_iters_robots(
            trajs_pos=results_single_plan.q_trajs_pos_iters,
            start_state=q_pos_start,
            goal_state=q_pos_goal,
            traj_pos_best=results_single_plan.q_trajs_pos_best,
            control_points=results_single_plan.control_points_iters,
            video_filepath=os.path.join(results_dir, f"{base_file_name}-robot-env-opt-iters-{idx:03d}.mp4"),
            n_frames=max((2, len(results_single_plan.q_trajs_pos_iters))),
            anim_time=args_inference.trajectory_duration,
            filter_joint_limits_vel_acc=True,
        )

        # reconstructed control points and trajectories at each diffusion iteration
        if results_single_plan.q_trajs_pos_recon_iters is not None:
            planning_task.animate_opt_iters_robots(
                trajs_pos=results_single_plan.q_trajs_pos_recon_iters,
                start_state=q_pos_start,
                goal_state=q_pos_goal,
                traj_pos_best=None,
                control_points=results_single_plan.control_points_recon_iters,
                video_filepath=os.path.join(results_dir, f"{base_file_name}-robot-env-opt-iters-recon-{idx:03d}.mp4"),
                n_frames=max((2, len(results_single_plan.q_trajs_pos_iters))),
                anim_time=args_inference.trajectory_duration,
                filter_joint_limits_vel_acc=True,
            )

    if render_planning_env_robot_trajectories:
        # visualize in the planning environment
        planning_task.animate_robot_trajectories(
            q_pos_trajs=results_single_plan.q_trajs_pos_iters[-1],
            q_pos_start=q_pos_start,
            q_pos_goal=q_pos_goal,
            plot_x_trajs=True,
            video_filepath=os.path.join(results_dir, f"{base_file_name}-robot-env-{idx:03d}.mp4"),
            n_frames=max((2, results_single_plan.q_trajs_pos_iters[-1].shape[1] // 10)),
            anim_time=args_inference.trajectory_duration,
            filter_joint_limits_vel_acc=True,
        )

    if debug:
        plt.show()


class GenerativeOptimizationPlanner:

    def __init__(
        self,
        planning_task,
        dataset,
        args_train,
        args_inference,
        tensor_args=DEFAULT_TENSOR_ARGS,
        sampling_based_planner_fn=None,
        debug=False,
        **kwargs,
    ):
        self.planning_task = planning_task
        self.dataset = dataset

        self.args_inference = args_inference

        self.tensor_args = tensor_args

        self.sampling_based_planner_fn = sampling_based_planner_fn

        self.debug = debug

        ################################################################################################################
        # Load the generative model
        # model_path = os.path.join(
        #     args_inference.model_dir, 'checkpoints',
        #     f'{"ema_" if args_train["use_ema"] else ""}model_current.pth'
        # )
        model_path = os.path.join(
            args_inference.model_dir, "checkpoints", f'{"ema_" if args_train["use_ema"] else ""}model_current.pth'
        )
        self.model = torch.load(model_path, map_location=tensor_args["device"], weights_only=False)
        if tensor_args["device"].type == "cpu" and isinstance(
            getattr(self.model, "model", None), torch.nn.DataParallel
        ):
            self.model.model = self.model.model.module
        self.model = self.model.to(tensor_args["device"])
        self.model.eval()
        freeze_torch_model_params(self.model)

        ################################################################################################################
        # Setup the generative model
        self.sample_fn_kwargs = {}
        if isinstance(self.model, GaussianDiffusionModel):
            diffusion_sampling_args = args_inference[args_inference.diffusion_sampling_method]
            if args_inference.diffusion_sampling_method == "ddpm":
                t_start_guide = ceil(
                    diffusion_sampling_args.t_start_guide_steps_fraction * self.model.n_diffusion_steps
                )
            elif args_inference.diffusion_sampling_method == "ddim":
                t_start_guide = ceil(
                    diffusion_sampling_args.t_start_guide_steps_fraction
                    * diffusion_sampling_args.ddim_sampling_timesteps
                )
            else:
                raise ValueError

            diffusion_sampling_args.update(
                method=args_inference.diffusion_sampling_method,
                t_start_guide=t_start_guide,
                n_diffusion_steps_without_noise=args_inference.n_diffusion_steps_without_noise,
                compute_costs_with_xrecon=args_inference.compute_costs_with_xrecon,
            )

            self.sample_fn_kwargs = diffusion_sampling_args
        elif isinstance(self.model, CVAEModel):
            pass
        else:
            raise NotImplementedError

        ################################################################################################################
        # Setup the costs and guided sampling
        self.cost_guide = None
        if args_inference.costs is not None:
            try:
                self.cost_guide = CostGuideManagerParametricTrajectory(
                    planning_task, dataset, args_inference, tensor_args, debug, **kwargs
                )
            except NoCostException:
                self.cost_guide = None

        ################################################################################################################
        # Warmup the model and guide costs
        self.warmup()

    def warmup(self, warmup_rounds=5, **kwargs):
        # cache the model for faster inference
        if self.debug:
            print(f"{'=' * 80}\nWarming up...\n{'=' * 80}")
        shape_x = (self.args_inference.n_trajectory_samples, *self.dataset.control_points_dim)
        for _ in range(warmup_rounds):
            self.model.warmup(shape_x, device=self.tensor_args["device"])
            if self.cost_guide is not None:
                self.cost_guide.warmup(shape_x)

    def plan_trajectory(
        self,
        q_pos_start,
        q_pos_goal,
        EE_pose_goal,
        n_trajectory_samples=None,
        results_ns: DotMap = None,
        debug=False,
        best_trajectory_selection="weighted_metrics",
        q_vel_start=None,
        q_vel_goal=None,
        q_acc_start=None,
        q_acc_goal=None,
        **kwargs,
    ):

        if results_ns is None:
            results_ns = DotMap()

        if n_trajectory_samples is None:
            n_trajectory_samples = self.args_inference.n_trajectory_samples

        # Prepare the input data and the context
        q_pos_start = to_torch(q_pos_start, **self.tensor_args)
        q_pos_goal = to_torch(q_pos_goal, **self.tensor_args)
        ee_pose_goal = to_torch(EE_pose_goal, **self.tensor_args)
        q_vel_start = (
            torch.zeros_like(q_pos_start) if q_vel_start is None else to_torch(q_vel_start, **self.tensor_args)
        )
        q_vel_goal = torch.zeros_like(q_pos_goal) if q_vel_goal is None else to_torch(q_vel_goal, **self.tensor_args)
        q_acc_start = (
            torch.zeros_like(q_pos_start) if q_acc_start is None else to_torch(q_acc_start, **self.tensor_args)
        )
        q_acc_goal = torch.zeros_like(q_pos_goal) if q_acc_goal is None else to_torch(q_acc_goal, **self.tensor_args)

        results_ns.update(
            q_pos_start=q_pos_start,
            q_pos_goal=q_pos_goal,
            ee_pose_goal=ee_pose_goal,
            q_vel_start=q_vel_start,
            q_vel_goal=q_vel_goal,
            q_acc_start=q_acc_start,
            q_acc_goal=q_acc_goal,
        )

        # Set the start and goal states
        self.planning_task.set_q_pos_start_goal(q_pos_start, q_pos_goal)
        self.planning_task.set_ee_pose_goal(ee_pose_goal)
        self.planning_task.parametric_trajectory.set_boundary_conditions(
            q_pos_start=q_pos_start,
            q_pos_goal=q_pos_goal,
            q_vel_start=q_vel_start,
            q_vel_goal=q_vel_goal,
            q_acc_start=q_acc_start,
            q_acc_goal=q_acc_goal,
        )

        # Plan trajectories with the generative optimization planner
        # Get also the reconstructed control points
        input_data_one_sample = self.dataset.create_data_sample_normalized(
            q_pos_start,
            q_pos_goal,
            ee_pose_goal=ee_pose_goal,
        )
        input_data_one_sample = dict_to_device(input_data_one_sample, self.tensor_args["device"])
        hard_conds = input_data_one_sample["hard_conds"]
        context_d = self.dataset.build_context(input_data_one_sample)

        with TimerCUDA() as t_inference_total:
            control_points_recon_normalized_iters = None
            if "rrtconnect" in self.args_inference.planner_alg:
                with TimerCUDA() as t_generator:
                    assert self.sampling_based_planner_fn is not None, "sampling_based_planner_fn must be provided"
                    assert (
                        self.args_inference.n_trajectory_samples == 1
                    ), "n_trajectory_samples must be 1 for RRTConnect"
                    # Use the RRTConnect planner to get an initial trajectory
                    q_pos_start_np = to_numpy(q_pos_start, dtype=np.float64)
                    q_pos_goal_np = to_numpy(q_pos_goal, dtype=np.float64)
                    results_plan_d = self.sampling_based_planner_fn(1, q_pos_start_np, q_pos_goal_np)
                    bspline_params = fit_bspline_to_path(  # tck
                        results_plan_d[0]["sol_path"],
                        self.planning_task.parametric_trajectory.bspline.d,
                        self.planning_task.parametric_trajectory.bspline.n_pts,
                        self.planning_task.parametric_trajectory.zero_vel_at_start_and_goal,
                        self.planning_task.parametric_trajectory.zero_acc_at_start_and_goal,
                    )
                results_ns.update(
                    t_generator=t_generator.elapsed,
                )
                control_points_unnnormalized_np = bspline_params[1].T
                control_points_unnormalized_all = to_torch(control_points_unnnormalized_np, **self.tensor_args)[
                    None, None, ...
                ]
                control_points_unnormalized_iters = self.planning_task.parametric_trajectory.remove_control_points_fn(
                    control_points_unnormalized_all,
                )
                control_points_normalized_iters = self.dataset.normalize_control_points(
                    control_points_unnormalized_iters
                )

            elif "gp_prior" in self.args_inference.planner_alg:
                # Construct a GP trajectory prior between the start and goal states
                # If the EE goal pose is given, we use the q_pos_goal from the dataset, which can be understood as
                # doing inverse kinematics before planning
                ts = np.array([[0.0], [1.0]])
                xs = np.array([to_numpy(q_pos_start), to_numpy(q_pos_goal)])

                length_scale_bound_lower = np.linalg.norm(xs[0] - xs[1]) / 2
                print(f"length_scale_bound_lower: {length_scale_bound_lower}")
                kernel = 1 * RBF(
                    length_scale=length_scale_bound_lower, length_scale_bounds=(length_scale_bound_lower, 1e4)
                )
                gaussian_process = GaussianProcessRegressor(
                    kernel=kernel,
                    # optimizer=None,
                    n_restarts_optimizer=10,
                )
                gaussian_process.fit(ts, xs)
                print(gaussian_process.kernel_)

                ts_ = np.linspace(0, 1, self.dataset.n_learnable_control_points + 2)[:, None]
                xs_samples = gaussian_process.sample_y(
                    ts_, n_samples=self.args_inference.n_trajectory_samples, random_state=None
                )
                xs_samples = np.moveaxis(xs_samples, -1, 0)

                control_points_unnormalized_iters = to_torch(xs_samples[:, 1:-1, :], **self.tensor_args)[None, ...]
                control_points_normalized_iters = self.dataset.normalize_control_points(
                    control_points_unnormalized_iters
                )

            elif "knn_prior" in self.args_inference.planner_alg:
                # Get the K nearest neighbors from the dataset
                control_points_normalized = self.dataset.get_knn_control_points(
                    q_pos_start, q_pos_goal, EE_pose_goal, k=self.args_inference.n_trajectory_samples, normalized=True
                )
                control_points_normalized_iters = control_points_normalized[None, ...]

            else:
                control_points_normalized_iters = self.model.run_inference(
                    guide=self.cost_guide if self.args_inference.planner_alg == "mpd" else None,
                    context_d=context_d,
                    hard_conds=hard_conds,
                    n_samples=n_trajectory_samples,
                    horizon=self.dataset.n_learnable_control_points,
                    return_chain=True,
                    return_chain_x_recon=False,
                    results_ns=results_ns,
                    **self.sample_fn_kwargs,
                    debug=debug,
                )

            # run additional guide steps for the prior + guide planner
            if self.cost_guide is not None and (
                self.args_inference.planner_alg
                in [
                    "diffusion_prior_then_guide",
                    "gp_prior_then_guide",
                    "knn_prior_then_guide",
                    "cvae_prior_then_guide",
                    "rrtconnect_then_guide",
                ]
            ):
                with TimerCUDA() as t_guide:
                    # the same number of guide steps as used in MPD
                    sample_fn_kwargs_copy = self.sample_fn_kwargs.copy()
                    sample_fn_kwargs_copy.update(**self.args_inference[self.args_inference.planner_alg])
                    sample_fn_kwargs_copy.update(n_guide_steps=1)
                    control_points_normalized_iters_post = [control_points_normalized_iters[-1].detach().clone()]
                    for _ in range(self.args_inference[self.args_inference.planner_alg].n_guide_steps):
                        control_points_normalized_tmp = guide_gradient_steps(
                            control_points_normalized_iters_post[-1].detach().clone(),
                            hard_conds=hard_conds,
                            context_d=context_d,
                            guide=self.cost_guide,
                            **sample_fn_kwargs_copy,
                        )
                        control_points_normalized_iters_post.append(control_points_normalized_tmp)
                    control_points_normalized_iters = torch.cat(
                        [control_points_normalized_iters, torch.stack(control_points_normalized_iters_post)], dim=0
                    )
                results_ns.t_guide = t_guide.elapsed

            # run additional motion planning gradient steps
            if self.cost_guide is not None and self.args_inference.extra_mp_steps > 0:
                with TimerCUDA() as t_guide_extra_mp_steps:
                    sample_fn_kwargs_copy = self.sample_fn_kwargs.copy()
                    # use the same arguments are the diffusion prior + guide planner
                    sample_fn_kwargs_copy.update(**self.args_inference["diffusion_prior_then_guide"])
                    sample_fn_kwargs_copy.update(n_guide_steps=1)
                    control_points_normalized_iters_post = [control_points_normalized_iters[-1].detach().clone()]
                    # update the CostTaskSpaceCollisionObjects cost to use all the collision objects
                    self.cost_guide.use_all_collision_objects()
                    for _ in range(self.args_inference.extra_mp_steps):
                        control_points_normalized_tmp = guide_gradient_steps(
                            control_points_normalized_iters_post[-1].detach().clone(),
                            hard_conds=hard_conds,
                            context_d=context_d,
                            guide=self.cost_guide,
                            **sample_fn_kwargs_copy,
                        )
                        control_points_normalized_iters_post.append(control_points_normalized_tmp)
                    control_points_normalized_iters = torch.cat(
                        [control_points_normalized_iters, torch.stack(control_points_normalized_iters_post)], dim=0
                    )

                results_ns.t_guide += t_guide_extra_mp_steps.elapsed

        results_ns.t_inference_total = t_inference_total.elapsed

        # unnormalize control point samples from the models and get the trajectory from the control points
        control_points_iters = self.dataset.unnormalize_control_points(control_points_normalized_iters)
        q_trajs_pos_iters, q_trajs_vel_iters, q_trajs_acc_iters = self.compute_trajectories_from_control_points(
            q_pos_start, q_pos_goal, control_points_iters
        )
        if control_points_recon_normalized_iters is not None:
            control_points_recon_iters = self.dataset.unnormalize_control_points(control_points_recon_normalized_iters)
            q_trajs_pos_recon_iters, q_trajs_vel_recon_iters, q_trajs_acc_recon_iters = (
                self.compute_trajectories_from_control_points(q_pos_start, q_pos_goal, control_points_recon_iters)
            )
        else:
            control_points_recon_iters = None
            q_trajs_pos_recon_iters = None
            q_trajs_vel_recon_iters = None
            q_trajs_acc_recon_iters = None

        # Filter the valid trajectories
        control_points_iter_0 = control_points_iters[-1]
        q_trajs_pos_iter_0 = q_trajs_pos_iters[-1]
        q_trajs_vel_iter_0 = q_trajs_vel_iters[-1]
        q_trajs_acc_iter_0 = q_trajs_acc_iters[-1]
        q_trajs_iter_0 = torch.cat([q_trajs_pos_iter_0, q_trajs_vel_iter_0, q_trajs_acc_iter_0], dim=-1)
        _, _, q_trajs_final_valid, valid_idxs, collision_waypoint_mask = self.planning_task.get_trajs_unvalid_and_valid(
            q_trajs_iter_0,
            return_indices=True,
            filter_joint_limits_vel_acc=True,
        )
        if valid_idxs.ndim == 2:
            valid_idxs = valid_idxs.squeeze(1)

        collision_trajectory_mask = collision_waypoint_mask.any(dim=-1)
        first_collision_steps = torch.full(
            (collision_waypoint_mask.shape[0],),
            -1,
            dtype=torch.long,
            device=collision_waypoint_mask.device,
        )
        first_collision_steps[collision_trajectory_mask] = torch.argmax(
            collision_waypoint_mask[collision_trajectory_mask].to(torch.long), dim=-1
        )

        joint_position_violation_mask = (
            torch.logical_or(
                q_trajs_pos_iter_0 < self.planning_task.robot.q_pos_min,
                q_trajs_pos_iter_0 > self.planning_task.robot.q_pos_max,
            )
            .any(dim=-1)
            .any(dim=-1)
        )
        joint_velocity_violation_mask = torch.zeros_like(joint_position_violation_mask)
        if self.planning_task.robot.dq_max is not None:
            joint_velocity_violation_mask = (
                torch.logical_or(
                    q_trajs_vel_iter_0 < -self.planning_task.robot.dq_max,
                    q_trajs_vel_iter_0 > self.planning_task.robot.dq_max,
                )
                .any(dim=-1)
                .any(dim=-1)
            )
        joint_acceleration_violation_mask = torch.zeros_like(joint_position_violation_mask)
        if self.planning_task.robot.ddq_max is not None:
            joint_acceleration_violation_mask = (
                torch.logical_or(
                    q_trajs_acc_iter_0 < -self.planning_task.robot.ddq_max,
                    q_trajs_acc_iter_0 > self.planning_task.robot.ddq_max,
                )
                .any(dim=-1)
                .any(dim=-1)
            )

        valid_trajectory_mask = torch.zeros_like(collision_trajectory_mask)
        valid_idxs_flat = valid_idxs.reshape(-1).long()
        if valid_idxs_flat.numel() > 0:
            valid_trajectory_mask[valid_idxs_flat] = True

        control_points_valid = control_points_iter_0[valid_idxs]
        q_trajs_pos_valid = q_trajs_pos_iter_0[valid_idxs]
        q_trajs_vel_valid = q_trajs_vel_iter_0[valid_idxs]
        q_trajs_acc_valid = q_trajs_acc_iter_0[valid_idxs]

        # Get the "best" trajectory from all the valid ones
        best_trajectory_selection_details = None
        if valid_idxs.numel() == 0:
            control_points_best = None
            q_trajs_pos_best = None
            q_trajs_vel_best = None
            q_trajs_acc_best = None
        else:
            best_trajectory_selection = self.args_inference.get("best_trajectory_selection", best_trajectory_selection)
            best_trajectory_selection_details = {
                "method": best_trajectory_selection,
            }
            ee_pose_goal_error_position_norm = None
            ee_pose_goal_error_orientation_norm = None
            if self.dataset.context_ee_goal_pose:
                ee_pose_goal_achieved = self.planning_task.robot.get_EE_pose(q_trajs_pos_valid[..., -1, :])
                error_ee_pose_goal_position, error_ee_pose_goal_orientation = compute_ee_pose_errors(
                    ee_pose_goal, ee_pose_goal_achieved
                )
                ee_pose_goal_error_position_norm = torch.linalg.norm(error_ee_pose_goal_position, dim=-1)
                ee_pose_goal_error_orientation_norm = torch.rad2deg(
                    torch.linalg.norm(error_ee_pose_goal_orientation, dim=-1)
                )

            if best_trajectory_selection == "weighted_metrics":
                metric_values = {
                    "path_length": compute_path_length(q_trajs_pos_valid, self.planning_task.robot),
                    "smoothness": compute_smoothness(
                        q_trajs_pos_valid,
                        self.planning_task.robot,
                        trajs_acc=q_trajs_acc_valid,
                    ),
                }
                if self.dataset.context_ee_goal_pose:
                    metric_values.update(
                        ee_position=ee_pose_goal_error_position_norm,
                        ee_orientation=ee_pose_goal_error_orientation_norm,
                    )
                if self.planning_task.robot.dq_max is not None:
                    dq_max = torch.as_tensor(
                        self.planning_task.robot.dq_max,
                        dtype=q_trajs_vel_valid.dtype,
                        device=q_trajs_vel_valid.device,
                    )
                    metric_values["velocity_limit_utilization"] = torch.amax(
                        torch.abs(q_trajs_vel_valid) / dq_max,
                        dim=(-2, -1),
                    )
                if self.planning_task.robot.ddq_max is not None:
                    ddq_max = torch.as_tensor(
                        self.planning_task.robot.ddq_max,
                        dtype=q_trajs_acc_valid.dtype,
                        device=q_trajs_acc_valid.device,
                    )
                    metric_values["acceleration_limit_utilization"] = torch.amax(
                        torch.abs(q_trajs_acc_valid) / ddq_max,
                        dim=(-2, -1),
                    )

                configured_weights = self.args_inference.get("best_trajectory_weights", {})
                configured_scales = self.args_inference.get("best_trajectory_metric_scales", {})
                active_metrics = []
                active_weight_sum = 0.0
                for metric_name, values in metric_values.items():
                    weight = float(
                        _config_value(
                            configured_weights,
                            metric_name,
                            DEFAULT_BEST_TRAJECTORY_WEIGHTS[metric_name],
                        )
                    )
                    if weight <= 0.0:
                        continue
                    configured_scale = _config_value(
                        configured_scales,
                        metric_name,
                        DEFAULT_BEST_TRAJECTORY_METRIC_SCALES[metric_name],
                    )
                    normalized_values, scale = _normalize_best_trajectory_metric(values, configured_scale)
                    active_metrics.append((metric_name, values, normalized_values, scale, weight))
                    active_weight_sum += weight

                if not active_metrics:
                    raise ValueError("weighted_metrics requires at least one available metric with a positive weight.")

                weighted_score = torch.zeros_like(active_metrics[0][1])
                for _, _, normalized_values, _, weight in active_metrics:
                    normalized_weight = weight / active_weight_sum
                    weighted_score = weighted_score + normalized_weight * normalized_values.square()
                idx_min_cost = torch.argmin(weighted_score)

                component_details = {}
                for metric_name, values, normalized_values, scale, weight in active_metrics:
                    normalized_weight = weight / active_weight_sum
                    component_details[metric_name] = {
                        "value": float(values[idx_min_cost].detach().cpu().item()),
                        "scale": float(scale.detach().cpu().item()),
                        "weight": normalized_weight,
                        "normalized_value": float(normalized_values[idx_min_cost].detach().cpu().item()),
                        "weighted_term": float(
                            (normalized_weight * normalized_values[idx_min_cost].square()).detach().cpu().item()
                        ),
                    }
                best_trajectory_selection_details.update(
                    selected_valid_index=int(idx_min_cost.detach().cpu().item()),
                    score=float(weighted_score[idx_min_cost].detach().cpu().item()),
                    components=component_details,
                )
            elif best_trajectory_selection == "lowest_ee_position_error":
                if not self.dataset.context_ee_goal_pose:
                    raise ValueError("lowest_ee_position_error requires context_ee_goal_pose=True.")
                idx_min_cost = torch.argmin(ee_pose_goal_error_position_norm)
                best_trajectory_selection_details.update(
                    selected_valid_index=int(idx_min_cost.detach().cpu().item()),
                    score=float(ee_pose_goal_error_position_norm[idx_min_cost].detach().cpu().item()),
                )
            elif best_trajectory_selection == "lowest_ee_orientation_error":
                if not self.dataset.context_ee_goal_pose:
                    raise ValueError("lowest_ee_orientation_error requires context_ee_goal_pose=True.")
                idx_min_cost = torch.argmin(ee_pose_goal_error_orientation_norm)
                best_trajectory_selection_details.update(
                    selected_valid_index=int(idx_min_cost.detach().cpu().item()),
                    score=float(ee_pose_goal_error_orientation_norm[idx_min_cost].detach().cpu().item()),
                )
            elif best_trajectory_selection == "lowest_weighted_cost":
                if self.cost_guide is None:
                    raise ValueError("lowest_weighted_cost requires an active cost guide.")
                costs_valid, *_ = self.cost_guide(control_points_valid, return_cost=True)
                idx_min_cost = torch.argmin(costs_valid)
                best_trajectory_selection_details.update(
                    selected_valid_index=int(idx_min_cost.detach().cpu().item()),
                    score=float(costs_valid[idx_min_cost].detach().cpu().item()),
                )
            elif best_trajectory_selection == "lowest_smoothness_cost":
                batch_smoothness = compute_smoothness(
                    q_trajs_pos_valid, self.planning_task.robot, trajs_acc=q_trajs_acc_valid
                )
                idx_min_cost = torch.argmin(batch_smoothness)
                best_trajectory_selection_details.update(
                    selected_valid_index=int(idx_min_cost.detach().cpu().item()),
                    score=float(batch_smoothness[idx_min_cost].detach().cpu().item()),
                )
            elif best_trajectory_selection == "shortest_path_length":
                batch_path_length = compute_path_length(q_trajs_pos_valid, self.planning_task.robot)
                idx_min_cost = torch.argmin(batch_path_length)
                best_trajectory_selection_details.update(
                    selected_valid_index=int(idx_min_cost.detach().cpu().item()),
                    score=float(batch_path_length[idx_min_cost].detach().cpu().item()),
                )
            else:
                raise ValueError(f"Unknown best_trajectory_selection={best_trajectory_selection!r}.")

            control_points_best = control_points_valid[idx_min_cost]
            q_trajs_pos_best = q_trajs_pos_valid[idx_min_cost]
            q_trajs_vel_best = q_trajs_vel_valid[idx_min_cost]
            q_trajs_acc_best = q_trajs_acc_valid[idx_min_cost]

        results_ns.update(
            # control points and trajectories at each diffusion iteration
            control_points_iters=control_points_iters,
            q_trajs_pos_iters=q_trajs_pos_iters,
            q_trajs_vel_iters=q_trajs_vel_iters,
            q_trajs_acc_iters=q_trajs_acc_iters,
            # reconstructed control points and trajectories at each diffusion iteration
            control_points_recon_iters=control_points_recon_iters,
            q_trajs_pos_recon_iters=q_trajs_pos_recon_iters,
            q_trajs_vel_recon_iters=q_trajs_vel_recon_iters,
            q_trajs_acc_recon_iters=q_trajs_acc_recon_iters,
            # control points and trajectories at the last iteration
            control_points_iter_0=control_points_iter_0,
            q_trajs_pos_iter_0=q_trajs_pos_iter_0,
            q_trajs_vel_iter_0=q_trajs_vel_iter_0,
            q_trajs_acc_iter_0=q_trajs_acc_iter_0,
            # final trajectory validity diagnostics
            collision_waypoint_mask=collision_waypoint_mask,
            collision_trajectory_mask=collision_trajectory_mask,
            first_collision_steps=first_collision_steps,
            joint_position_violation_mask=joint_position_violation_mask,
            joint_velocity_violation_mask=joint_velocity_violation_mask,
            joint_acceleration_violation_mask=joint_acceleration_violation_mask,
            valid_trajectory_mask=valid_trajectory_mask,
            # valid control points and trajectories
            control_points_valid=control_points_valid,
            q_trajs_pos_valid=q_trajs_pos_valid,
            q_trajs_vel_valid=q_trajs_vel_valid,
            q_trajs_acc_valid=q_trajs_acc_valid,
            # best control points and trajectories
            control_points_best=control_points_best,
            q_trajs_pos_best=q_trajs_pos_best,
            q_trajs_vel_best=q_trajs_vel_best,
            q_trajs_acc_best=q_trajs_acc_best,
            best_trajectory_selection_details=best_trajectory_selection_details,
            # trajectory time steps
            timesteps=self.planning_task.parametric_trajectory.get_timesteps(num=q_trajs_pos_iter_0.shape[1]),
        )
        return results_ns

    def compute_trajectories_from_control_points(self, q_pos_start, q_pos_goal, control_points, **kwargs):
        # Get the position, velocity and acceleration trajectories
        q_traj_d = self.planning_task.parametric_trajectory.get_q_trajectory(
            control_points, q_pos_start, q_pos_goal, get_type=("pos", "vel", "acc"), get_time_representation=True
        )
        q_trajs_pos_iters = q_traj_d["pos"]
        q_trajs_vel_iters = q_traj_d["vel"]
        q_trajs_acc_iters = q_traj_d["acc"]
        return q_trajs_pos_iters, q_trajs_vel_iters, q_trajs_acc_iters
