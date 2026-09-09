#!/usr/bin/env python3
"""Validate a Phase-1 Marvin Top-K artifact in Isaac Lab."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

from marvin_bimanual_asset import (
    CANONICAL_JOINT_NAMES,
    classify_contact_forces,
    load_inference_artifact,
    validate_marvin_urdf,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--asset-cache",
        type=Path,
        default=Path(__file__).resolve().parents[2] / ".cache/isaaclab/marvin_bimanual",
    )
    parser.add_argument("--robot-usd", type=Path, default=None)
    parser.add_argument("--force-usd-conversion", action="store_true")
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--contact-force-threshold", type=float, default=1.0)
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--fk-position-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--fk-orientation-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--graceful-shutdown", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (
        args.action_repeat < 1
        or args.physics_dt <= 0.0
        or args.contact_force_threshold < 0.0
        or args.fk_position_tolerance <= 0.0
        or args.fk_orientation_tolerance <= 0.0
    ):
        parser.error("repeat, dt, thresholds, and FK tolerances must be valid and positive")
    return args


args_cli = parse_args()
artifact = load_inference_artifact(args_cli.artifact)
urdf_gate = validate_marvin_urdf()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from marvin_bimanual_asset import (
    build_marvin_articulation_cfg,
    convert_marvin_urdf_to_usd,
    resolve_canonical_joint_ids,
    resolve_tcp_body_ids,
    spawn_scene_obstacles,
    tcp_poses_from_body_state,
    validate_marvin_usd,
)


def _torch(value):
    return value.torch if hasattr(value, "torch") else value


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _tcp_poses_numpy(robot, tcp_ids, env_origins):
    body_position = (
        _torch(robot.data.body_pos_w)[:, list(tcp_ids)] - _torch(env_origins)[:, None, :]
    ).detach().cpu().numpy()
    body_quaternion = (
        _torch(robot.data.body_quat_w)[:, list(tcp_ids)].detach().cpu().numpy()
    )
    positions, rotations = tcp_poses_from_body_state(body_position, body_quaternion)
    return np.concatenate((rotations, positions[..., None]), axis=-1)


def _pose_errors(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    position = np.linalg.norm(actual[..., :3, 3] - expected[..., :3, 3], axis=-1)
    relative = expected[..., :3, :3] @ np.swapaxes(actual[..., :3, :3], -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return position, np.arccos(cosine)


def _first_step(mask_history):
    history = np.stack(mask_history, axis=0)
    any_contact = history.any(axis=0)
    first = history.argmax(axis=0).astype(np.int64)
    first[~any_contact] = -1
    return any_contact, first


def _asset_metadata():
    if args_cli.robot_usd is None:
        return convert_marvin_urdf_to_usd(
            args_cli.asset_cache,
            force=args_cli.force_usd_conversion,
        )
    return validate_marvin_usd(args_cli.robot_usd)


def run_evaluation():
    asset = _asset_metadata()
    robot_cfg = build_marvin_articulation_cfg(asset["usd_path"])

    @configclass
    class MarvinEvaluationSceneCfg(InteractiveSceneCfg):
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=1800.0),
        )
        robot = robot_cfg
        contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )

    top_k = artifact.top_k_positions
    count, horizon, dof = top_k.shape
    if dof != 14:
        raise ValueError("Marvin evaluation requires exactly 14 trajectory joints")
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=args_cli.physics_dt, device=args_cli.device)
    )
    sim.set_camera_view([2.2, -2.2, 1.7], [0.3, 0.0, 0.5])
    scene = InteractiveScene(MarvinEvaluationSceneCfg(num_envs=count, env_spacing=3.0))
    obstacle_summary = spawn_scene_obstacles(sim_utils, artifact.scene)
    sim.reset()
    robot = scene["robot"]
    sensor = scene["contacts"]
    joint_ids = resolve_canonical_joint_ids(robot)
    tcp_ids = resolve_tcp_body_ids(robot)
    sim_dt = sim.get_physics_dt()
    trajectories = torch.as_tensor(top_k, dtype=torch.float32, device=sim.device)

    initial = robot.data.default_joint_pos.clone()
    initial[:, joint_ids] = trajectories[:, 0]
    robot.write_joint_state_to_sim(initial, torch.zeros_like(initial))
    robot.set_joint_position_target(initial)
    scene.reset()
    sensor.reset()
    scene.write_data_to_sim()
    sim.forward()
    scene.update(0.0)
    isaac_tcp_pose_start = _tcp_poses_numpy(robot, tcp_ids, scene.env_origins)
    sim.step(render=not args_cli.headless)
    scene.update(sim_dt)

    category_history = {
        key: []
        for key in (
            "contact",
            "left_world_contact",
            "right_world_contact",
            "interarm_contact",
            "base_contact",
        )
    }
    max_tracking_error = torch.zeros(count, device=sim.device)
    max_contact_force = np.zeros(count, dtype=np.float64)
    actual_joint_violation = np.zeros(count, dtype=bool)
    lower = np.asarray([urdf_gate["joint_limits"][name][0] for name in CANONICAL_JOINT_NAMES])
    upper = np.asarray([urdf_gate["joint_limits"][name][1] for name in CANONICAL_JOINT_NAMES])
    for waypoint in range(horizon):
        target = robot.data.joint_pos.clone()
        target[:, joint_ids] = trajectories[:, waypoint]
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        for _ in range(args_cli.action_repeat):
            sim.step(render=not args_cli.headless)
            scene.update(sim_dt)
        measured = _torch(robot.data.joint_pos)[:, joint_ids]
        max_tracking_error = torch.maximum(
            max_tracking_error,
            torch.amax(torch.abs(measured - trajectories[:, waypoint]), dim=-1),
        )
        forces = _torch(sensor.data.net_forces_w).detach().cpu().numpy()
        measured_cpu = measured.detach().cpu().numpy()
        actual_joint_violation |= ((measured_cpu < lower) | (measured_cpu > upper)).any(axis=1)
        max_contact_force = np.maximum(max_contact_force, np.linalg.norm(forces, axis=-1).max(axis=-1))
        classified = classify_contact_forces(
            sensor.body_names, forces, args_cli.contact_force_threshold
        )
        total = np.linalg.norm(forces, axis=-1).max(axis=-1) > args_cli.contact_force_threshold
        category_history["contact"].append(total)
        for key in category_history:
            if key != "contact":
                category_history[key].append(classified[key])

    contact_summary = {}
    for key, history in category_history.items():
        mask, first = _first_step(history)
        contact_summary[key] = mask
        contact_summary[f"first_{key}_waypoint"] = first

    target_joint_violation = ((top_k < lower) | (top_k > upper)).any(axis=(1, 2))

    tcp_position_error = np.full((count, 2), np.nan)
    tcp_orientation_error = np.full((count, 2), np.nan)
    isaac_tcp_pose_tracked_final = _tcp_poses_numpy(robot, tcp_ids, scene.env_origins)
    if artifact.ee_goal_pose is not None:
        tcp_position_error, tcp_orientation_error = _pose_errors(
            isaac_tcp_pose_tracked_final, artifact.ee_goal_pose[None]
        )

    kinematic_final = robot.data.default_joint_pos.clone()
    kinematic_final[:, joint_ids] = trajectories[:, -1]
    robot.write_joint_state_to_sim(kinematic_final, torch.zeros_like(kinematic_final))
    sim.forward()
    scene.update(0.0)
    isaac_tcp_pose_final = _tcp_poses_numpy(robot, tcp_ids, scene.env_origins)

    start_fk_position_error = np.full((count, 2), np.nan)
    start_fk_orientation_error = np.full((count, 2), np.nan)
    final_fk_position_error = np.full((count, 2), np.nan)
    final_fk_orientation_error = np.full((count, 2), np.nan)
    if artifact.mpd_tcp_pose_start is not None:
        start_fk_position_error, start_fk_orientation_error = _pose_errors(
            isaac_tcp_pose_start, artifact.mpd_tcp_pose_start[None]
        )
    if artifact.mpd_top_k_tcp_pose_final is not None:
        final_fk_position_error, final_fk_orientation_error = _pose_errors(
            isaac_tcp_pose_final, artifact.mpd_top_k_tcp_pose_final
        )

    isaac_collision = contact_summary["contact"]
    # Every saved Top-K candidate passed MPD's dense safety oracle.
    mpd_safety_available = artifact.result.get("validation", {}).get("backend") != "contract_stub"
    confusion = {
        "available": mpd_safety_available,
        "true_positive": 0 if mpd_safety_available else None,
        "false_positive": 0 if mpd_safety_available else None,
        "true_negative": int((~isaac_collision).sum()) if mpd_safety_available else None,
        "false_negative": int(isaac_collision.sum()) if mpd_safety_available else None,
    }
    trajectories_output = []
    for index in range(count):
        trajectories_output.append(
            {
                "top_k_index": index,
                "candidate_index": int(artifact.top_k_candidate_indices[index]),
                "contact": bool(contact_summary["contact"][index]),
                "first_contact_waypoint": int(contact_summary["first_contact_waypoint"][index]),
                "left_world_contact": bool(contact_summary["left_world_contact"][index]),
                "right_world_contact": bool(contact_summary["right_world_contact"][index]),
                "interarm_contact": bool(contact_summary["interarm_contact"][index]),
                "base_contact": bool(contact_summary["base_contact"][index]),
                "joint_limit_violation": bool(target_joint_violation[index]),
                "actual_joint_limit_violation": bool(actual_joint_violation[index]),
                "max_tracking_error_rad": float(max_tracking_error[index].item()),
                "max_contact_force_n": float(max_contact_force[index]),
                "left_tcp_position_error_m": float(tcp_position_error[index, 0]),
                "right_tcp_position_error_m": float(tcp_position_error[index, 1]),
                "left_tcp_orientation_error_rad": float(tcp_orientation_error[index, 0]),
                "right_tcp_orientation_error_rad": float(tcp_orientation_error[index, 1]),
                "start_mpd_isaac_position_error_m": start_fk_position_error[index].tolist(),
                "start_mpd_isaac_orientation_error_rad": start_fk_orientation_error[index].tolist(),
                "final_mpd_isaac_position_error_m": final_fk_position_error[index].tolist(),
                "final_mpd_isaac_orientation_error_rad": final_fk_orientation_error[index].tolist(),
            }
        )

    fk_values = np.concatenate(
        (
            start_fk_position_error.reshape(-1),
            start_fk_orientation_error.reshape(-1),
            final_fk_position_error.reshape(-1),
            final_fk_orientation_error.reshape(-1),
        )
    )
    fk_available = bool(np.isfinite(fk_values).all())
    fk_within_tolerance = bool(
        fk_available
        and np.nanmax(start_fk_position_error) <= args_cli.fk_position_tolerance
        and np.nanmax(final_fk_position_error) <= args_cli.fk_position_tolerance
        and np.nanmax(start_fk_orientation_error) <= args_cli.fk_orientation_tolerance
        and np.nanmax(final_fk_orientation_error) <= args_cli.fk_orientation_tolerance
    )

    return {
        "schema": "marvin_bimanual_isaaclab_evaluation/v1",
        "artifact": artifact.root.as_posix(),
        "robot": {
            "urdf": asset["urdf_path"].as_posix(),
            "urdf_sha256": asset["urdf_sha256"],
            "usd": asset["usd_path"].as_posix(),
            "usd_sha256": asset["usd_sha256"],
            "usd_bundle_sha256": asset["usd_bundle_sha256"],
            "asset_manifest_sha256": asset["asset_manifest_sha256"],
            "joint_names": list(CANONICAL_JOINT_NAMES),
            "isaac_native_joint_names": list(robot.joint_names),
            "canonical_joint_ids": list(joint_ids),
            "tcp_frame_names": list(asset["tcp_frame_names"]),
            "tcp_body_names": list(asset["tcp_body_names"]),
            "tcp_offsets_xyz": asset["tcp_offsets_xyz"],
            "tcp_calibrated": asset["tcp_calibrated"],
        },
        "artifact_hashes": artifact.hashes,
        "contact_classification": "body-side net force with balanced-pair interarm heuristic",
        "contact_classification_is_safety_authority": False,
        "trajectories": trajectories_output,
        "mpd_vs_isaac_collision_confusion": confusion,
        "safety_false_negative": bool(
            mpd_safety_available and confusion["false_negative"] > 0
        ),
        "mpd_isaac_fk": {
            "available": fk_available,
            "within_tolerance": fk_within_tolerance,
            "position_tolerance_m": args_cli.fk_position_tolerance,
            "orientation_tolerance_rad": args_cli.fk_orientation_tolerance,
        },
        "kinematic_mismatch": bool(fk_available and not fk_within_tolerance),
        "physics_dt": args_cli.physics_dt,
        "action_repeat": args_cli.action_repeat,
        "contact_force_threshold": args_cli.contact_force_threshold,
        **obstacle_summary,
    }


def main():
    exit_code = 0
    try:
        report = _jsonable(run_evaluation())
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
        if report["kinematic_mismatch"]:
            exit_code = 3
        elif report["safety_false_negative"]:
            exit_code = 2
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    if args_cli.graceful_shutdown:
        simulation_app.close(wait_for_replicator=False)
        raise SystemExit(exit_code)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
