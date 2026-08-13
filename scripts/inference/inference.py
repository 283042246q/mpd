from mpd.utils.patches import numpy_monkey_patch

numpy_monkey_patch()

import time
from functools import partial, wraps

from dotmap import DotMap

import gc
import os
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from experiment_launcher import single_experiment_yaml, run_experiment
from experiment_launcher.utils import create_results_dir
from mpd.inference.inference import EvaluationSamplesGenerator, GenerativeOptimizationPlanner, render_results
from mpd.metrics.metrics import PlanningMetricsCalculator
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml, save_to_yaml
from scripts.isaaclab.scene_payload import export_isaaclab_scene_payload
from scripts.isaaclab.subprocess_utils import run_isaaclab_evaluator_subprocess, run_isaaclab_replay_subprocess
from torch_robotics.robots import RobotPanda
from torch_robotics.trajectory.metrics import compute_path_length
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy

allow_ops_in_compiled_graph()


def single_experiment_yaml_optional_artifacts(experiment_function):
    launcher_experiment = single_experiment_yaml(experiment_function)

    @wraps(experiment_function)
    def wrapper(*args, **kwargs):
        if not kwargs.get("lightweight_output", False):
            return launcher_experiment(*args, **kwargs)
        create_results_dir(kwargs, make_dirs_with_seed=True)
        return experiment_function(*args, **kwargs)

    return wrapper


ISAACLAB_BATCH_SUPPORTED_ENVS = {
    "EnvOpenDrawerShelf",
    "EnvSpheres3D",
    "EnvSpheres3DExtraObjectsV00",
    "EnvThreePillarsPassage",
    "EnvWarehouse",
    "EnvWarehouseExtraObjectsV00",
}

ISAACLAB_BOX_OBSTACLE_ENVS = {
    "EnvOpenDrawerShelf",
    "EnvThreePillarsPassage",
    "EnvWarehouse",
    "EnvWarehouseExtraObjectsV00",
}


def _resolve_sim_backend(sim_backend, run_evaluation_issac_gym, run_evaluation_isaac_lab):
    if run_evaluation_issac_gym and run_evaluation_isaac_lab:
        raise ValueError("Only one simulation backend can be enabled at a time.")

    sim_backend = (sim_backend or "none").lower()
    if run_evaluation_issac_gym:
        sim_backend = "isaacgym"
    elif run_evaluation_isaac_lab:
        sim_backend = "isaaclab"

    valid_backends = {"none", "isaacgym", "isaaclab"}
    if sim_backend not in valid_backends:
        raise ValueError(f"Unknown sim_backend={sim_backend!r}. Expected one of {sorted(valid_backends)}.")
    return sim_backend


def _planning_env_name(planning_task):
    return getattr(planning_task.env, "name", type(planning_task.env).__name__)


def _require_isaaclab_batch_support(planning_task):
    env_name = _planning_env_name(planning_task)
    if env_name not in ISAACLAB_BATCH_SUPPORTED_ENVS:
        raise NotImplementedError(
            "IsaacLab batch evaluation currently supports only "
            f"{sorted(ISAACLAB_BATCH_SUPPORTED_ENVS)}. Got env_name={env_name!r}. "
            "Run with --sim_backend none for MPD-only inference, or use the dedicated IsaacLab replay/export path "
            "after obstacle export support is enabled for this environment."
        )
    return env_name


def _release_torch_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0].item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value, precision=6):
    value_float = _to_float(value)
    if value_float is None or not np.isfinite(value_float):
        return "N/A"
    return f"{value_float:.{precision}f}"


def _mapping_get(mapping, key, default=None):
    if mapping is None:
        return default
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _first_collision_summary(first_collision_steps, timesteps=None):
    if first_collision_steps is None:
        return None
    first_collision_steps = torch.as_tensor(first_collision_steps).detach().cpu().long().reshape(-1)
    collided_steps = first_collision_steps[first_collision_steps >= 0]
    if collided_steps.numel() == 0:
        return {
            "steps": first_collision_steps.tolist(),
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
            "time_min": None,
            "time_mean": None,
            "time_median": None,
            "time_max": None,
        }

    collided_steps_float = collided_steps.float()
    summary = {
        "steps": first_collision_steps.tolist(),
        "min": int(collided_steps.min().item()),
        "mean": float(collided_steps_float.mean().item()),
        "median": float(collided_steps_float.median().item()),
        "max": int(collided_steps.max().item()),
        "time_min": None,
        "time_mean": None,
        "time_median": None,
        "time_max": None,
    }
    if timesteps is not None:
        timesteps = torch.as_tensor(timesteps).detach().cpu().reshape(-1)
        collision_times = timesteps[collided_steps]
        summary.update(
            time_min=float(collision_times.min().item()),
            time_mean=float(collision_times.float().mean().item()),
            time_median=float(collision_times.float().median().item()),
            time_max=float(collision_times.max().item()),
        )
    return summary


def _write_inference_text_report(
    results_single_plan,
    planning_task,
    args_inference,
    sim_backend,
    idx_sg,
    results_dir,
):
    q_trajs_pos_all = results_single_plan.q_trajs_pos_iter_0
    n_generated = int(q_trajs_pos_all.shape[0])
    n_total = int(args_inference.n_trajectory_samples)
    horizon = int(q_trajs_pos_all.shape[1])

    dense_complete = _mapping_get(results_single_plan, "dense_validation_complete")
    dense_candidates_checked = _mapping_get(
        results_single_plan, "dense_validation_candidates_checked"
    )
    if dense_complete is False and dense_candidates_checked is not None:
        validity_denominator = int(dense_candidates_checked)
        validity_denominator_label = "dense_checked_candidates"
    else:
        validity_denominator = n_total
        validity_denominator_label = "n_trajectory_samples"
    denominator = max(validity_denominator, 1)

    q_trajs_pos_valid = results_single_plan.q_trajs_pos_valid
    n_valid = 0 if q_trajs_pos_valid is None else int(q_trajs_pos_valid.shape[0])
    collision_trajectory_mask = results_single_plan.collision_trajectory_mask.detach().cpu().bool()
    collision_waypoint_mask = results_single_plan.collision_waypoint_mask.detach().cpu().bool()
    collision_horizon = int(collision_waypoint_mask.shape[-1])
    n_collision = int(collision_trajectory_mask.sum().item())
    n_collision_waypoints = int(collision_waypoint_mask.sum().item())

    joint_position_violation_mask = results_single_plan.joint_position_violation_mask.detach().cpu().bool()
    joint_velocity_violation_mask = results_single_plan.joint_velocity_violation_mask.detach().cpu().bool()
    joint_acceleration_violation_mask = results_single_plan.joint_acceleration_violation_mask.detach().cpu().bool()

    path_lengths_all = compute_path_length(q_trajs_pos_all, planning_task.robot).detach().cpu()
    path_lengths_valid = None
    if n_valid > 0:
        path_lengths_valid = compute_path_length(q_trajs_pos_valid, planning_task.robot).detach().cpu()

    first_collision = _first_collision_summary(
        results_single_plan.first_collision_steps,
        results_single_plan.timesteps,
    )

    metrics = results_single_plan.metrics.toDict() if hasattr(results_single_plan.metrics, "toDict") else {}
    metrics_all = metrics.get("trajs_all", {})
    metrics_valid = metrics.get("trajs_valid", {})
    guidance_schedule = _mapping_get(results_single_plan, "ddim_guidance_schedule", [])
    ee_goal_position_weight_schedule = [
        _mapping_get(
            step,
            "ee_goal_position_weight",
            _mapping_get(step, "ee_pose_goal_weight"),
        )
        for step in guidance_schedule
    ]
    ee_goal_orientation_weight_schedule = [
        _mapping_get(
            step,
            "ee_goal_orientation_weight",
            _mapping_get(step, "ee_pose_goal_weight"),
        )
        for step in guidance_schedule
    ]
    prior_weight_schedule = [_mapping_get(step, "prior_weight") for step in guidance_schedule]
    best_selection_details = _mapping_get(results_single_plan, "best_trajectory_selection_details")
    best_selection_lines = ["[BEST TRAJECTORY SELECTION]"]
    if best_selection_details is None:
        best_selection_lines.extend(
            [
                "method: N/A",
                "selected_valid_index: N/A",
                "score: N/A",
            ]
        )
    else:
        best_selection_lines.extend(
            [
                f"method: {_mapping_get(best_selection_details, 'method', 'N/A')}",
                f"selected_valid_index: {_mapping_get(best_selection_details, 'selected_valid_index', 'N/A')}",
                f"selected_candidate_index: {_mapping_get(best_selection_details, 'selected_candidate_index', 'N/A')}",
                f"score: {_format_float(_mapping_get(best_selection_details, 'score'))}",
            ]
        )
        components = _mapping_get(best_selection_details, "components", {})
        for component_name, component in components.items():
            best_selection_lines.append(
                f"{component_name}: value={_format_float(_mapping_get(component, 'value'))}, "
                f"scale={_format_float(_mapping_get(component, 'scale'))}, "
                f"weight={_format_float(_mapping_get(component, 'weight'))}, "
                f"normalized={_format_float(_mapping_get(component, 'normalized_value'))}, "
                f"weighted_term={_format_float(_mapping_get(component, 'weighted_term'))}"
            )
    best_selection_lines.append("")

    start_goal_metadata = _mapping_get(results_single_plan, "start_goal_metadata", {})
    simulation_statistics = results_single_plan.sim_statistics
    if n_valid == 0:
        status = "failed_no_valid_trajectory"
        simulation_status = "skipped_no_valid_trajectory" if sim_backend != "none" else "not_requested"
    elif sim_backend == "isaaclab" and simulation_statistics is not None:
        status = "isaaclab_evaluated"
        simulation_status = "completed"
    else:
        status = "mpd_valid"
        simulation_status = "completed" if simulation_statistics is not None else "not_requested"

    lines = [
        "MPD INFERENCE REPORT",
        "====================",
        f"plan_index: {idx_sg}",
        f"status: {status}",
        f"environment: {_planning_env_name(planning_task)}",
        f"sim_backend: {sim_backend}",
        f"simulation_status: {simulation_status}",
        f"configured_total_trajectories: {n_total}",
        f"generated_total_trajectories: {n_generated}",
        "all_trajectory_rate_denominator: "
        f"{validity_denominator_label}={validity_denominator}",
        f"trajectory_horizon: {horizon}",
        "dense_validation_points: "
        f"{_format_float(_mapping_get(results_single_plan, 'dense_validation_points'), precision=0)}",
        "dense_validation_candidates_checked: "
        f"{_format_float(_mapping_get(results_single_plan, 'dense_validation_candidates_checked'), precision=0)}",
        "dense_validation_batches_evaluated: "
        f"{_format_float(_mapping_get(results_single_plan, 'dense_validation_batches_evaluated'), precision=0)}",
        "dense_validation_bucket_capacities: "
        f"{_mapping_get(results_single_plan, 'dense_validation_bucket_capacities', [])}",
        "dense_validation_padding_slots: "
        f"{_format_float(_mapping_get(results_single_plan, 'dense_validation_padding_slots'), precision=0)}",
        f"dense_validation_complete: {_mapping_get(results_single_plan, 'dense_validation_complete', 'N/A')}",
        "dense_validation_ranked_early_exit: "
        f"{_mapping_get(results_single_plan, 'dense_validation_ranked_early_exit', False)}",
        "",
        "[START/GOAL]",
        f"source: {_mapping_get(start_goal_metadata, 'source', 'unknown')}",
        f"selection: {_mapping_get(start_goal_metadata, 'selection', 'N/A')}",
        f"sample_index: {_mapping_get(start_goal_metadata, 'sample_index', 'N/A')}",
        f"sampling_attempt: {_mapping_get(start_goal_metadata, 'sampling_attempt', 'N/A')}",
        f"start_region_id: {_mapping_get(start_goal_metadata, 'start_region_id', 'N/A')}",
        f"goal_region_id: {_mapping_get(start_goal_metadata, 'goal_region_id', 'N/A')}",
        f"rotation_z_axis_deg: {_mapping_get(start_goal_metadata, 'rotation_z_axis_deg', 'N/A')}",
        f"rotate_with_environment: {_mapping_get(start_goal_metadata, 'rotate_with_environment', 'N/A')}",
        "",
        "[MPD VALIDITY]",
        f"valid_trajectories: {n_valid}",
        f"valid_rate: {n_valid / denominator:.6f} ({n_valid}/{validity_denominator})",
        f"colliding_trajectories: {n_collision}",
        f"collision_rate: {n_collision / denominator:.6f} ({n_collision}/{validity_denominator})",
        f"collision_waypoints: {n_collision_waypoints}",
        f"collision_waypoint_fraction: {n_collision_waypoints / max(validity_denominator * collision_horizon, 1):.6f} "
        f"({n_collision_waypoints}/{validity_denominator * collision_horizon})",
        f"joint_position_violation_trajectories: {int(joint_position_violation_mask.sum().item())}",
        f"joint_position_violation_rate: {joint_position_violation_mask.sum().item() / denominator:.6f}",
        f"joint_velocity_violation_trajectories: {int(joint_velocity_violation_mask.sum().item())}",
        f"joint_velocity_violation_rate: {joint_velocity_violation_mask.sum().item() / denominator:.6f}",
        f"joint_acceleration_violation_trajectories: {int(joint_acceleration_violation_mask.sum().item())}",
        f"joint_acceleration_violation_rate: {joint_acceleration_violation_mask.sum().item() / denominator:.6f}",
        "",
        "[FIRST COLLISION - MPD WAYPOINT INDEX]",
        f"first_collision_step_min: {_format_float(first_collision['min'], precision=3)}",
        f"first_collision_step_mean: {_format_float(first_collision['mean'], precision=3)}",
        f"first_collision_step_median: {_format_float(first_collision['median'], precision=3)}",
        f"first_collision_step_max: {_format_float(first_collision['max'], precision=3)}",
        f"first_collision_time_sec_min: {_format_float(first_collision['time_min'])}",
        f"first_collision_time_sec_mean: {_format_float(first_collision['time_mean'])}",
        f"first_collision_time_sec_median: {_format_float(first_collision['time_median'])}",
        f"first_collision_time_sec_max: {_format_float(first_collision['time_max'])}",
        f"first_collision_steps_by_trajectory: {first_collision['steps']}",
        "",
        "[PATH LENGTH - JOINT SPACE]",
        f"all_trajectories_path_length_mean: {_format_float(path_lengths_all.mean())}",
        f"all_trajectories_path_length_std: {_format_float(path_lengths_all.std())}",
        f"valid_trajectories_path_length_mean: "
        f"{_format_float(path_lengths_valid.mean() if path_lengths_valid is not None else None)}",
        f"valid_trajectories_path_length_std: "
        f"{_format_float(path_lengths_valid.std() if path_lengths_valid is not None else None)}",
        "",
        "[END-EFFECTOR GOAL ERROR - ALL TRAJECTORIES]",
        "position_error_mean_m: " f"{_format_float(metrics_all.get('ee_pose_goal_error_position_norm_mean'))}",
        "orientation_error_mean_deg: " f"{_format_float(metrics_all.get('ee_pose_goal_error_orientation_norm_mean'))}",
        "",
        "[END-EFFECTOR GOAL ERROR - VALID TRAJECTORIES]",
        "position_error_mean_m: " f"{_format_float(metrics_valid.get('ee_pose_goal_error_position_norm_mean'))}",
        "orientation_error_mean_deg: "
        f"{_format_float(metrics_valid.get('ee_pose_goal_error_orientation_norm_mean'))}",
        "",
        "[VALID TRAJECTORY METRICS - MEDIAN]",
        "ee_position_error_median_m: " f"{_format_float(metrics_valid.get('ee_pose_goal_error_position_norm_median'))}",
        "ee_orientation_error_median_deg: "
        f"{_format_float(metrics_valid.get('ee_pose_goal_error_orientation_norm_median'))}",
        f"path_length_median: {_format_float(metrics_valid.get('path_length_median'))}",
        f"smoothness_median: {_format_float(metrics_valid.get('smoothness_median'))}",
        "velocity_limit_utilization_median: "
        f"{_format_float(metrics_valid.get('velocity_limit_utilization_median'))}",
        "acceleration_limit_utilization_median: "
        f"{_format_float(metrics_valid.get('acceleration_limit_utilization_median'))}",
        "",
        *best_selection_lines,
        "[DDIM GUIDANCE SCHEDULE]",
        f"active_steps: {len(guidance_schedule)}",
        "ee_goal_position_weights: "
        f"[{', '.join(_format_float(value) for value in ee_goal_position_weight_schedule)}]",
        "ee_goal_orientation_weights: "
        f"[{', '.join(_format_float(value) for value in ee_goal_orientation_weight_schedule)}]",
        f"prior_weights: [{', '.join(_format_float(value) for value in prior_weight_schedule)}]",
        "",
        "[TIMING]",
        f"inference_total_sec: {_format_float(results_single_plan.t_inference_total)}",
        f"trajectory_ranking_sec: {_format_float(_mapping_get(results_single_plan, 'trajectory_ranking_time', 0.0))}",
        f"dense_validation_sec: {_format_float(_mapping_get(results_single_plan, 'dense_validation_time', 0.0))}",
        "inference_total_with_dense_validation_sec: "
        f"{_format_float(results_single_plan.t_inference_total + _mapping_get(results_single_plan, 'trajectory_ranking_time', 0.0) + _mapping_get(results_single_plan, 'dense_validation_time', 0.0))}",
        f"generator_sec: {_format_float(results_single_plan.t_generator)}",
        f"guide_sec: {_format_float(results_single_plan.t_guide)}",
        "",
        "[ISAAC LAB]",
    ]

    if sim_backend == "isaaclab" and simulation_statistics is not None:
        n_simulated = int(_mapping_get(simulation_statistics, "n_trajectories", 0))
        n_sim_collision = int(_mapping_get(simulation_statistics, "n_trajectories_collision", 0))
        n_sim_free = int(_mapping_get(simulation_statistics, "n_trajectories_free", 0))
        isaaclab_first_collision = _first_collision_summary(
            _mapping_get(simulation_statistics, "first_collision_step"),
            results_single_plan.timesteps,
        )
        lines.extend(
            [
                "status: completed",
                f"evaluated_trajectories: {n_simulated}",
                f"evaluated_rate_over_total_samples: {n_simulated / denominator:.6f} ({n_simulated}/{n_total})",
                f"collision_trajectories: {n_sim_collision}",
                f"collision_rate_over_total_samples: "
                f"{n_sim_collision / denominator:.6f} ({n_sim_collision}/{n_total})",
                f"free_trajectories: {n_sim_free}",
                f"free_rate_over_total_samples: {n_sim_free / denominator:.6f} ({n_sim_free}/{n_total})",
                f"first_collision_step_min: {_format_float(isaaclab_first_collision['min'], precision=3)}",
                f"first_collision_step_mean: {_format_float(isaaclab_first_collision['mean'], precision=3)}",
                f"first_collision_step_median: {_format_float(isaaclab_first_collision['median'], precision=3)}",
                f"first_collision_step_max: {_format_float(isaaclab_first_collision['max'], precision=3)}",
                f"first_collision_steps_by_trajectory: {isaaclab_first_collision['steps']}",
                f"contact_force_threshold: "
                f"{_format_float(_mapping_get(simulation_statistics, 'contact_force_threshold'))}",
            ]
        )
    else:
        isaaclab_status = simulation_status if sim_backend == "isaaclab" else "not_requested"
        lines.append(f"status: {isaaclab_status}")

    report_path = Path(results_dir) / f"inference-report-{idx_sg:03d}.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _run_isaaclab_evaluator(
    q_trajs_pos,
    q_trajs_pos_best,
    q_pos_goal,
    planning_task,
    idx_sg,
    results_dir,
    isaaclab_root,
    isaaclab_conda_env,
    isaaclab_device,
    isaaclab_headless,
    isaaclab_action_repeat,
    isaaclab_timeout_s,
    isaaclab_replay,
    isaaclab_replay_trajectory_index,
    isaaclab_replay_video_fps,
    isaaclab_replay_width,
    isaaclab_replay_height,
):
    if not isinstance(planning_task.robot, RobotPanda):
        raise NotImplementedError("The IsaacLab evaluator currently supports RobotPanda trajectories only.")
    env_name = _require_isaaclab_batch_support(planning_task)

    horizon = int(q_trajs_pos.shape[0])
    trajectory_duration = float(getattr(planning_task.parametric_trajectory, "trajectory_duration", 0.0))
    trajectory_dt = trajectory_duration / max(horizon - 1, 1) if trajectory_duration > 0.0 else 0.0

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = results_dir / f"isaaclab-trajectories-{idx_sg:03d}.pt"
    statistics_path = results_dir / f"isaaclab-statistics-{idx_sg:03d}.json"
    log_path = results_dir / f"isaaclab-evaluator-{idx_sg:03d}.log"
    replay_log_path = results_dir / f"isaaclab-replay-{idx_sg:03d}.log"
    replay_video_path = results_dir / f"isaaclab-replay-{idx_sg:03d}.mp4"
    replay_screenshot_path = results_dir / f"isaaclab-replay-{idx_sg:03d}.png"
    replay_json_path = results_dir / f"isaaclab-replay-{idx_sg:03d}.json"

    scene_payload = export_isaaclab_scene_payload(planning_task.env, include_boxes=True)
    if scene_payload["unsupported_obstacles"]:
        raise RuntimeError(
            f"IsaacLab {env_name} evaluation cannot export every obstacle. "
            f"Got unsupported={scene_payload['unsupported_obstacles']}."
        )
    if env_name in ISAACLAB_BOX_OBSTACLE_ENVS and not any(
        obstacle.get("type") == "box" for obstacle in scene_payload["obstacles"]
    ):
        raise RuntimeError(f"IsaacLab {env_name} evaluation requires exported box obstacles. Got scene={scene_payload}.")

    payload = {
        "q_trajs_pos": q_trajs_pos.detach().cpu(),
        "q_pos_starts": q_trajs_pos[0].detach().cpu(),
        "q_pos_goal": q_pos_goal.detach().cpu(),
        "robot_name": "panda",
        "env_name": env_name,
        "dt": trajectory_dt,
        "scene": scene_payload,
    }
    if q_trajs_pos_best is not None:
        payload["q_trajs_pos_best"] = q_trajs_pos_best.detach().cpu()
    torch.save(payload, trajectories_path)

    replay_request = None
    if isaaclab_replay:
        replay_request = {
            "trajectories_path": trajectories_path,
            "log_path": replay_log_path,
            "video_path": replay_video_path,
            "screenshot_path": replay_screenshot_path,
            "output_json_path": replay_json_path,
            "trajectory_source": "best",
            "trajectory_index": isaaclab_replay_trajectory_index,
            "isaaclab_root": isaaclab_root,
            "isaaclab_conda_env": isaaclab_conda_env,
            "isaaclab_device": isaaclab_device,
            "isaaclab_headless": isaaclab_headless,
            "isaaclab_action_repeat": isaaclab_action_repeat,
            "isaaclab_timeout_s": isaaclab_timeout_s,
            "video_fps": isaaclab_replay_video_fps,
            "width": isaaclab_replay_width,
            "height": isaaclab_replay_height,
        }

    statistics = run_isaaclab_evaluator_subprocess(
        trajectories_path=trajectories_path,
        statistics_path=statistics_path,
        log_path=log_path,
        isaaclab_root=isaaclab_root,
        isaaclab_conda_env=isaaclab_conda_env,
        isaaclab_device=isaaclab_device,
        isaaclab_headless=isaaclab_headless,
        isaaclab_action_repeat=isaaclab_action_repeat,
        isaaclab_timeout_s=isaaclab_timeout_s,
    )
    statistics["trajectories_path"] = trajectories_path.as_posix()
    statistics["statistics_path"] = statistics_path.as_posix()
    statistics["evaluator_log_path"] = log_path.as_posix()

    if replay_request is not None:
        statistics["replay"] = {
            "status": "queued_after_inference",
            "trajectory_source": "best",
            "trajectory_index": int(isaaclab_replay_trajectory_index),
            "device": str(isaaclab_device),
            "log_path": replay_log_path.as_posix(),
            "video_path": replay_video_path.as_posix(),
            "screenshot_path": replay_screenshot_path.as_posix(),
            "output_json_path": replay_json_path.as_posix(),
        }

    return statistics, replay_request


@single_experiment_yaml_optional_artifacts
def experiment(
    ########################################################################
    # Configuration path defining the model and the inference parameters
    # cfg_inference_path: str = './cfgs/config_EnvNarrowPassageDense2D-RobotPointMass2D_00.yaml',
    # cfg_inference_path: str = './cfgs/config_EnvPlanar2Link-RobotPlanar2Link_00.yaml',
    # cfg_inference_path: str = './cfgs/config_EnvPlanar4Link-RobotPlanar4Link_00.yaml',
    # cfg_inference_path: str = "./cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml",
    # cfg_inference_path: str = './cfgs/config_EnvSpheres3D-RobotPanda_00.yaml',
    cfg_inference_path: str = "./cfgs/config_EnvWarehouse-RobotPanda-config_file_v01_00.yaml",
    # Optional full-model filename under <model_dir>/checkpoints. Empty selects
    # ema_model_current.pth or model_current.pth from the training configuration.
    checkpoint: str = "",
    ########################################################################
    # Select the start and goal from the training or validation/test set.
    selection_start_goal: str = "validation",  # training, validation/test
    ########################################################################
    # number of start and goal states to evaluate
    n_start_goal_states: int = 1,
    ########################################################################
    ee_pose_goal_weight_override: float = -1.0,
    t_start_guide_steps_fraction_override: float = -1.0,
    n_guide_steps_override: int = -1,
    ddim_scale_grad_prior_override: float = -1.0,
    ee_pose_goal_weight_end_override: float = -1.0,
    ee_goal_position_weight_override: float = -1.0,
    ee_goal_position_weight_end_override: float = -1.0,
    ee_goal_orientation_weight_override: float = -1.0,
    ee_goal_orientation_weight_end_override: float = -1.0,
    ddim_scale_grad_prior_end_override: float = -1.0,
    # Benchmark-only runtime override. "config" preserves YAML, while true or
    # false toggles ranked fixed-bucket dense validation without duplicating a
    # complete environment config.
    dense_ranked_early_exit_override: str = "config",
    ########################################################################
    save_args_inference: bool = True,
    save_results_single_plan: bool = True,
    save_results_single_plan_low_mem: bool = False,
    lightweight_output: bool = False,
    ########################################################################
    # Visualization options
    render_joint_space_time_iters: bool = True,
    render_joint_space_env_iters: bool = False,
    render_env_robot_opt_iters: bool = False,
    render_env_robot_trajectories: bool = False,
    render_pybullet: bool = False,
    draw_collision_spheres: bool = False,
    sim_backend: str = "none",  # none, isaacgym, isaaclab
    run_evaluation_issac_gym: bool = False,
    run_evaluation_isaac_lab: bool = False,
    render_isaacgym_viewer: bool = False,
    render_isaacgym_movie: bool = False,
    isaaclab_root: str = os.environ.get("ISAACLAB_ROOT", "/home/eric/IsaacLab"),
    isaaclab_conda_env: str = os.environ.get("ISAACLAB_CONDA_ENV", "env_isaaclab"),
    isaaclab_device: str = "cuda:0",
    isaaclab_headless: bool = True,
    isaaclab_action_repeat: int = 4,
    isaaclab_timeout_s: int = 900,
    isaaclab_replay: bool = True,
    isaaclab_replay_trajectory_index: int = 0,
    isaaclab_replay_video_fps: float = 24.0,
    isaaclab_replay_width: int = 960,
    isaaclab_replay_height: int = 540,
    ########################################################################
    device: str = "cuda:0",  # cpu, cuda
    debug: bool = False,
    ########################################################################
    # MANDATORY
    seed: int = int(time.time()),
    # seed: int = 2,
    results_dir: str = "logs",
    ########################################################################
    **kwargs,
):
    sim_backend = _resolve_sim_backend(sim_backend, run_evaluation_issac_gym, run_evaluation_isaac_lab)

    # Set random seed for reproducibility
    fix_random_seed(seed)

    device = get_torch_device(device)
    tensor_args = {"device": device, "dtype": torch.float32}

    # Save and load the inference configuration
    cfg_inference_path_resolved = Path(os.path.expandvars(os.path.expanduser(str(cfg_inference_path))))
    if not cfg_inference_path_resolved.is_absolute():
        cfg_inference_path_resolved = (Path.cwd() / cfg_inference_path_resolved).resolve()
    args_inference = DotMap(load_params_from_yaml(cfg_inference_path_resolved.as_posix()))
    dense_override = str(dense_ranked_early_exit_override).strip().lower()
    if dense_override not in {"config", "true", "false"}:
        raise ValueError(
            "dense_ranked_early_exit_override must be config, true, or false."
        )
    if dense_override != "config":
        if not args_inference.get("dense_validation"):
            args_inference.dense_validation = DotMap()
        if not args_inference.dense_validation.get("ranked_early_exit"):
            args_inference.dense_validation.ranked_early_exit = DotMap()
        args_inference.dense_validation.ranked_early_exit.enabled = (
            dense_override == "true"
        )
    checkpoint_cli = checkpoint.strip()
    checkpoint_yaml = str(args_inference.get("checkpoint") or "").strip()
    args_inference.checkpoint = checkpoint_cli or checkpoint_yaml or None

    if ee_pose_goal_weight_override >= 0.0:
        for cost_key in (
            "CostTaskSpaceEEGoalPosition",
            "CostTaskSpaceEEGoalOrientation",
            "CostTaskSpaceEEGoalPose",
        ):
            if cost_key in args_inference.costs:
                args_inference.costs[cost_key].weight = ee_pose_goal_weight_override
    if ee_goal_position_weight_override >= 0.0:
        if "CostTaskSpaceEEGoalPosition" in args_inference.costs:
            args_inference.costs.CostTaskSpaceEEGoalPosition.weight = ee_goal_position_weight_override
        args_inference.ddim.ee_goal_position_weight_start = ee_goal_position_weight_override
    if ee_goal_orientation_weight_override >= 0.0:
        if "CostTaskSpaceEEGoalOrientation" in args_inference.costs:
            args_inference.costs.CostTaskSpaceEEGoalOrientation.weight = ee_goal_orientation_weight_override
        args_inference.ddim.ee_goal_orientation_weight_start = ee_goal_orientation_weight_override
    if t_start_guide_steps_fraction_override >= 0.0:
        args_inference.ddim.t_start_guide_steps_fraction = t_start_guide_steps_fraction_override
    if n_guide_steps_override >= 0:
        args_inference.ddim.n_guide_steps = n_guide_steps_override
    if ddim_scale_grad_prior_override >= 0.0:
        args_inference.ddim.ddim_scale_grad_prior = ddim_scale_grad_prior_override
    if ee_pose_goal_weight_override >= 0.0 and ee_pose_goal_weight_end_override >= 0.0:
        args_inference.ddim.ee_pose_goal_weight_start = ee_pose_goal_weight_override
        args_inference.ddim.ee_pose_goal_weight_end = ee_pose_goal_weight_end_override
    if ee_goal_position_weight_end_override >= 0.0:
        args_inference.ddim.ee_goal_position_weight_end = ee_goal_position_weight_end_override
    if ee_goal_orientation_weight_end_override >= 0.0:
        args_inference.ddim.ee_goal_orientation_weight_end = ee_goal_orientation_weight_end_override
    if ddim_scale_grad_prior_end_override >= 0.0:
        args_inference.ddim.ddim_scale_grad_prior_end = ddim_scale_grad_prior_end_override

    start_goal_regions_path = args_inference.get("start_goal_regions_path")
    if start_goal_regions_path:
        start_goal_regions_path = Path(os.path.expandvars(os.path.expanduser(str(start_goal_regions_path))))
        if not start_goal_regions_path.is_absolute():
            start_goal_regions_path = cfg_inference_path_resolved.parent / start_goal_regions_path
        args_inference.start_goal_regions_path = start_goal_regions_path.resolve().as_posix()

    if "cvae" in args_inference.planner_alg:
        if args_inference.model_selection == "bspline":
            args_inference.model_dir = args_inference.model_dir_cvae_bspline
        elif args_inference.model_selection == "waypoints":
            args_inference.model_dir = args_inference.model_dir_cvae_waypoints
        else:
            raise NotImplementedError
    else:
        if args_inference.model_selection == "bspline":
            args_inference.model_dir = args_inference.model_dir_ddpm_bspline
        elif args_inference.model_selection == "waypoints":
            args_inference.model_dir = args_inference.model_dir_ddpm_waypoints
        else:
            raise NotImplementedError

    args_inference.model_dir = os.path.expandvars(args_inference.model_dir)

    if save_args_inference:
        save_to_yaml(args_inference.toDict(), os.path.join(results_dir, "args_inference.yaml"))

    print(f"\n-------------------------------------------------------------------------------------------------")
    print(f"cfg_inference_path:\n{cfg_inference_path_resolved}")
    print(f"Model:\n{args_inference.model_dir}")
    print(f"--------------------------------------------------------------------------------------------------")

    ################################################################################################################
    # Load dataset, environment, robot and planning task.
    # Override training parameters.
    args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))
    args_train.update(
        **args_inference,
        gripper=True,
        reload_data=False,
        results_dir=results_dir,
        load_indices=True,
        tensor_args=tensor_args,
    )
    planning_task, train_subset, _, val_subset, _ = get_planning_task_and_dataset(**args_train)

    ################################################################################################################
    # Generator of evaluation samples
    evaluation_samples_kwargs = dict(args_inference)
    selection_start_goal = evaluation_samples_kwargs.pop("selection_start_goal", selection_start_goal)
    evaluation_samples_generator = EvaluationSamplesGenerator(
        planning_task,
        train_subset,
        val_subset,
        selection_start_goal=selection_start_goal,
        planner="RRTConnect",
        tensor_args=tensor_args,
        debug=debug,
        render_pybullet=render_pybullet,
        **evaluation_samples_kwargs,
    )

    ################################################################################################################
    # Load the generative model planner
    generative_optimization_planner = GenerativeOptimizationPlanner(
        planning_task,
        train_subset.dataset,
        args_train,
        args_inference,
        tensor_args,
        sampling_based_planner_fn=partial(
            evaluation_samples_generator.generate_data_ompl_worker.run,
            planner_allowed_time=10.0,
            interpolate_num=args_inference.num_T_pts,
            simplify_path=True,
        ),
        debug=debug,
    )

    ################################################################################################################
    # IsaacGym environment and motion planning controller
    motion_planning_isaac_env = None
    if sim_backend == "isaacgym":
        from torch_robotics.isaac_gym_envs.motion_planning_envs import (
            MotionPlanningControllerIsaacGym,
            MotionPlanningIsaacGymEnv,
        )

        robot_asset_file = planning_task.robot.robot_urdf_file
        if draw_collision_spheres:
            robot_asset_file = planning_task.robot.robot_urdf_collision_spheres_file
        motion_planning_isaac_env = MotionPlanningIsaacGymEnv(
            planning_task.env,
            planning_task.robot,
            asset_root=get_robot_path().as_posix(),
            robot_asset_file=robot_asset_file.replace(get_robot_path().as_posix() + "/", ""),
            num_envs=args_inference.n_trajectory_samples,
            # all_robots_in_one_env=True if n_start_goal_states == 1 else False,
            all_robots_in_one_env=True,
            render_isaacgym_viewer=render_isaacgym_viewer,
            render_camera_global=render_isaacgym_movie,
            render_camera_global_append_to_recorder=render_isaacgym_movie,
            sync_viewer_with_real_time=False,
            show_viewer=render_isaacgym_viewer,
            camera_global_from_top=True if planning_task.env.dim == 2 else False,
            add_ground_plane=False,
            viewer_time_between_steps=torch.diff(planning_task.parametric_trajectory.get_timesteps()[:2]).item(),
            draw_goal_configuration=True if not train_subset.dataset.context_ee_goal_pose else False,
            draw_ee_pose_goal=True if train_subset.dataset.context_ee_goal_pose else False,
            color_robots=False,
            draw_contact_forces=False,
            draw_end_effector_frame=False,
            draw_end_effector_path=True,
        )

        motion_planning_controller_isaac_gym = MotionPlanningControllerIsaacGym(motion_planning_isaac_env)

    ################################################################################################################
    # Metrics calculator
    planning_metrics_calculator = PlanningMetricsCalculator(planning_task)

    ################################################################################################################
    # Plan for several start and goal states sequentially
    start_goal_source = str(args_inference.get("start_goal_source", "auto")).lower()
    uses_dataset_start_goal = start_goal_source == "dataset" or (
        start_goal_source == "auto" and selection_start_goal in {"training", "validation"}
    )
    if uses_dataset_start_goal:
        dataset_subset = train_subset if selection_start_goal == "training" else val_subset
        idx_sample_l = np.random.choice(np.arange(len(dataset_subset)), n_start_goal_states)
    else:
        idx_sample_l = np.arange(n_start_goal_states)
    isaaclab_replay_requests = []
    reproducible_start_goal_states = []
    for idx_sg, idx_sample in enumerate(idx_sample_l):
        print(f"\n-------------------------------------------------------------------------------------------------")
        print(f"----------------PLANNING {idx_sg+1}/{n_start_goal_states}------------------")
        print(f"--------------------------------------------------------------------------------------------------")

        results_single_plan = DotMap(t_generator=0.0, t_guide=0.0)

        q_pos_start, q_pos_goal, ee_pose_goal = evaluation_samples_generator.get_data_sample(idx_sample)
        results_single_plan.start_goal_metadata = dict(evaluation_samples_generator.last_sample_metadata)
        reproducible_start_goal_states.append(
            {
                "q_pos_start": to_numpy(q_pos_start).tolist(),
                "q_pos_goal": to_numpy(q_pos_goal).tolist(),
                "ee_pose_goal": to_numpy(ee_pose_goal).reshape(3, 4).tolist(),
            }
        )

        print("\n----------------START AND GOAL states----------------")
        print(f"q_pos_start: {q_pos_start}")
        print(f"q_pos_goal: {q_pos_goal}")
        print(f"ee_pose_goal: {ee_pose_goal}")

        if debug:
            evaluation_samples_generator.add_start_goal_marker(q_pos_start, q_pos_goal)

        ############################################################################################################
        # Run motion planning inference
        print(f"\n----------------PLAN TRAJECTORIES----------------")
        print(f"Starting inference...")
        if generative_optimization_planner.cost_guide is not None:
            generative_optimization_planner.cost_guide.profile_context_index = idx_sg
        results_single_plan = generative_optimization_planner.plan_trajectory(
            q_pos_start,
            q_pos_goal,
            ee_pose_goal,
            q_vel_start=torch.zeros_like(q_pos_start),
            q_vel_goal=torch.zeros_like(q_pos_goal),
            q_acc_start=torch.zeros_like(q_pos_start),
            q_acc_goal=torch.zeros_like(q_pos_goal),
            results_ns=results_single_plan,
            debug=debug,
        )
        if (
            generative_optimization_planner.cost_guide is not None
            and generative_optimization_planner.cost_guide.guidance_profiler.records
        ):
            generative_optimization_planner.cost_guide.guidance_profiler.write_csv(
                Path(results_dir) / "active-statistics.csv"
            )
        print(f"...inference finished.")

        ############################################################################################################
        # Show in pybullet the best trajectory
        if render_pybullet and results_single_plan.q_trajs_pos_best is not None:
            time.sleep(3)
            ########################
            # Visualize in Pybullet
            q_pos_path = to_numpy(results_single_plan.q_trajs_pos_best)
            # add panda grippers to the path
            if (
                isinstance(planning_task.robot, RobotPanda)
                and q_pos_path.shape[1] == 7
                and evaluation_samples_generator.generate_data_ompl_worker.pbompl_interface.robot.num_dim == 9
            ):
                q_pos_path = np.concatenate((q_pos_path, np.zeros((q_pos_path.shape[0], 2))), axis=-1)
            evaluation_samples_generator.generate_data_ompl_worker.pbompl_interface.execute(
                q_pos_path, sleep_time=planning_task.parametric_trajectory.dt
            )

        ############################################################################################################
        # Evaluate with the selected simulator backend
        simulation_statistics = None
        results_single_plan.isaacgym_statistics = None
        results_single_plan.sim_statistics = None
        if sim_backend != "none" and results_single_plan.q_trajs_pos_valid is not None:
            if results_single_plan.q_trajs_pos_valid.shape[0] > 0:
                q_trajs_pos = results_single_plan.q_trajs_pos_valid.movedim(1, 0)  # horizon, batch, D
                if sim_backend == "isaacgym":
                    ########################
                    motion_planning_isaac_env.ee_pose_goal = planning_task.robot.get_EE_pose(
                        to_torch(q_pos_goal.unsqueeze(0), device), flatten_pos_quat=True, quat_xyzw=True
                    ).squeeze(0)

                    simulation_statistics = motion_planning_controller_isaac_gym.execute_trajectories(
                        q_trajs_pos,
                        q_pos_starts=q_trajs_pos[0],
                        q_pos_goal=q_trajs_pos[-1][0],  # add steps for better visualization
                        n_pre_steps=5 if render_isaacgym_viewer or render_isaacgym_movie else 0,
                        n_post_steps=5 if render_isaacgym_viewer or render_isaacgym_movie else 0,
                        stop_robot_if_in_contact=False,
                        make_video=render_isaacgym_movie,
                        video_duration=args_inference.trajectory_duration,
                        video_path=os.path.join(results_dir, f"isaacgym-{idx_sg:03d}.mp4"),
                        make_gif=False,
                    )
                elif sim_backend == "isaaclab":
                    simulation_statistics, replay_request = _run_isaaclab_evaluator(
                        q_trajs_pos=q_trajs_pos,
                        q_trajs_pos_best=results_single_plan.q_trajs_pos_best,
                        q_pos_goal=q_pos_goal,
                        planning_task=planning_task,
                        idx_sg=idx_sg,
                        results_dir=results_dir,
                        isaaclab_root=isaaclab_root,
                        isaaclab_conda_env=isaaclab_conda_env,
                        isaaclab_device=isaaclab_device,
                        isaaclab_headless=isaaclab_headless,
                        isaaclab_action_repeat=isaaclab_action_repeat,
                        isaaclab_timeout_s=isaaclab_timeout_s,
                        isaaclab_replay=isaaclab_replay,
                        isaaclab_replay_trajectory_index=isaaclab_replay_trajectory_index,
                        isaaclab_replay_video_fps=isaaclab_replay_video_fps,
                        isaaclab_replay_width=isaaclab_replay_width,
                        isaaclab_replay_height=isaaclab_replay_height,
                    )
                    if replay_request is not None:
                        isaaclab_replay_requests.append(replay_request)
            results_single_plan.isaacgym_statistics = simulation_statistics
            results_single_plan.sim_statistics = simulation_statistics

        ############################################################################################################
        # Compute motion planning metrics
        print(f"\n----------------METRICS----------------")
        results_single_plan.metrics = planning_metrics_calculator.compute_metrics(results_single_plan)

        print(f"t_inference_total: {results_single_plan.t_inference_total:.3f} sec")
        print(f"t_generator: {results_single_plan.t_generator:.3f} sec")
        print(f"t_guide: {results_single_plan.t_guide:.3f} sec")

        print(f"sim_statistics:")
        pprint(results_single_plan.sim_statistics)

        print(f"metrics:")
        pprint(results_single_plan.metrics)

        report_path = _write_inference_text_report(
            results_single_plan=results_single_plan,
            planning_task=planning_task,
            args_inference=args_inference,
            sim_backend=sim_backend,
            idx_sg=idx_sg,
            results_dir=results_dir,
        )
        results_single_plan.inference_report_path = report_path.as_posix()
        print(f"inference_report: {report_path}")

        if save_results_single_plan:
            results_single_plan_to_save = results_single_plan
            if save_results_single_plan_low_mem:
                results_single_plan_to_save = DotMap(
                    t_generator=results_single_plan.t_generator,
                    t_guide=results_single_plan.t_guide,
                    t_inference_total=results_single_plan.t_inference_total,
                    q_pos_start=q_pos_start,
                    q_pos_goal=q_pos_goal,
                    ee_pose_goal=ee_pose_goal,
                    start_goal_metadata=results_single_plan.start_goal_metadata,
                    control_points_iters=results_single_plan.control_points_iters,
                    metrics=results_single_plan.metrics,
                    best_trajectory_selection_details=results_single_plan.best_trajectory_selection_details,
                    isaacgym_statistics=results_single_plan.isaacgym_statistics,
                    sim_statistics=results_single_plan.sim_statistics,
                )
            torch.save(
                results_single_plan_to_save,
                os.path.join(results_dir, f"results_single_plan-{idx_sg:03d}.pt"),
                _use_new_zipfile_serialization=True,
            )

        ############################################################################################################
        # Render sampling results
        render_results(
            args_inference,
            planning_task,
            q_pos_start,
            q_pos_goal,
            results_single_plan,
            idx_sg,
            results_dir,
            render_joint_space_time_iters=render_joint_space_time_iters,
            render_joint_space_env_iters=render_joint_space_env_iters,
            render_planning_env_robot_opt_iters=render_env_robot_opt_iters,
            render_planning_env_robot_trajectories=render_env_robot_trajectories,
            debug=debug,
        )

        ############################################################################################################
        # empty memory
        del results_single_plan
        _release_torch_memory()

    reproducible_states_path = Path(results_dir) / "start-goal-states.yaml"
    save_to_yaml(reproducible_start_goal_states, reproducible_states_path)
    print(f"reproducible_start_goal_states: {reproducible_states_path}")

    ################################################################################################################
    # clean up inference resources before launching replay processes
    evaluation_samples_generator.generate_data_ompl_worker.terminate()
    if motion_planning_isaac_env is not None:
        motion_planning_isaac_env.clean_up()
        del motion_planning_isaac_env
        del motion_planning_controller_isaac_gym
    del generative_optimization_planner
    del planning_metrics_calculator
    del evaluation_samples_generator
    del planning_task
    del train_subset
    del val_subset
    _release_torch_memory()

    ################################################################################################################
    # Run queued visual replays only after all inference/evaluator files are saved and MPD CUDA memory is released.
    if isaaclab_replay_requests:
        print("\n----------------ISAACLAB REPLAY----------------")
        for replay_request in isaaclab_replay_requests:
            print(f"Replaying {replay_request['trajectories_path']} ...")
            replay_statistics = run_isaaclab_replay_subprocess(**replay_request)
            pprint(replay_statistics)
            _release_torch_memory()


if __name__ == "__main__":
    run_experiment(experiment)
