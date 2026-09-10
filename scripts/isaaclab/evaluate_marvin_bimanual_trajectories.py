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
    MARVIN_CONTACT_BODY_PATHS,
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
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=0,
        help="Physics steps per waypoint; 0 follows artifact time_from_start.",
    )
    parser.add_argument("--contact-force-threshold", type=float, default=1.0)
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--fk-position-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--fk-orientation-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--graceful-shutdown", action="store_true")
    native_self = parser.add_mutually_exclusive_group()
    native_self.add_argument(
        "--enable-native-self-collisions",
        dest="native_self_collisions",
        action="store_true",
        help=(
            "Diagnostic only: enable unfiltered PhysX articulation self collisions; "
            "the imported Marvin collision meshes contain unsupported adjacent pairs."
        ),
    )
    native_self.add_argument(
        "--disable-native-self-collisions",
        dest="native_self_collisions",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(native_self_collisions=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (
        args.action_repeat < 0
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
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab_physx.physics import PhysxManager

from marvin_bimanual_asset import (
    build_marvin_articulation_cfg,
    convert_marvin_urdf_to_usd,
    resolve_canonical_joint_ids,
    resolve_tcp_body_ids,
    spawn_scene_obstacles,
    tcp_poses_from_body_state,
    trajectory_physics_step_schedule,
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
    # ContactSensor assumes all selected bodies are siblings.  Marvin's URDF
    # converter preserves a nested link tree, so use exact PhysX views instead.
    get_settings_manager().set_bool("/physics/disableContactProcessing", False)
    robot_cfg = build_marvin_articulation_cfg(
        asset["usd_path"],
        enabled_self_collisions=args_cli.native_self_collisions,
    )

    @configclass
    class MarvinEvaluationSceneCfg(InteractiveSceneCfg):
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=1800.0),
        )
        robot = robot_cfg

    top_k = artifact.top_k_positions
    count, horizon, dof = top_k.shape
    if dof != 14:
        raise ValueError("Marvin evaluation requires exactly 14 trajectory joints")
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=args_cli.physics_dt, device=args_cli.device)
    )
    sim.set_camera_view([2.2, -2.2, 1.7], [0.3, 0.0, 0.5])
    scene = InteractiveScene(
        MarvinEvaluationSceneCfg(num_envs=count, env_spacing=3.0)
    )
    obstacle_summary = spawn_scene_obstacles(sim_utils, artifact.scene)
    # ``activate_contact_sensors`` stops walking below its first rigid body.
    # That only activates Marvin's root base because the imported articulation
    # keeps rigid links nested.  Apply the reporter to every exact link before
    # PhysX creates its tensor views.
    for env_index in range(count):
        for _, body_path in MARVIN_CONTACT_BODY_PATHS:
            sim_utils.activate_contact_sensors(
                f"/World/envs/env_{env_index}/Robot{body_path}"
            )
    sim.reset()
    robot = scene["robot"]
    joint_ids = resolve_canonical_joint_ids(robot)
    tcp_ids = resolve_tcp_body_ids(robot)
    sim_dt = sim.get_physics_dt()
    body_path_by_name = dict(MARVIN_CONTACT_BODY_PATHS)
    missing_contact_paths = [
        name for name in robot.body_names if name not in body_path_by_name
    ]
    if missing_contact_paths:
        raise RuntimeError(
            f"Marvin rigid bodies missing contact-view paths: {missing_contact_paths}"
        )
    # PhysX may merge fixed bodies into their parent articulation link.  Build
    # views for the actual articulation bodies rather than assuming all URDF
    # fixed links remain independently addressable.
    contact_body_names = list(robot.body_names)
    physics_view = PhysxManager.get_physics_sim_view()
    contact_views = [
        physics_view.create_rigid_contact_view(
            "/World/envs/env_*/Robot" + body_path_by_name[name]
        )
        for name in contact_body_names
    ]
    trajectories = torch.as_tensor(top_k, dtype=torch.float32, device=sim.device)

    initial = robot.data.default_joint_pos.clone()
    initial[:, joint_ids] = trajectories[:, 0]
    robot.write_joint_state_to_sim(initial, torch.zeros_like(initial))
    robot.set_joint_position_target(initial)
    scene.reset()
    scene.write_data_to_sim()
    sim.forward()
    scene.update(0.0)
    isaac_tcp_pose_start = _tcp_poses_numpy(robot, tcp_ids, scene.env_origins)
    sim.step(render=not args_cli.headless)
    scene.update(sim_dt)
    step_schedule = trajectory_physics_step_schedule(
        artifact.time_from_start, sim_dt, args_cli.action_repeat
    )

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
    max_contact_force_by_body = np.zeros(
        (count, len(contact_body_names)), dtype=np.float64
    )
    actual_joint_violation = np.zeros(count, dtype=bool)
    lower = np.asarray([urdf_gate["joint_limits"][name][0] for name in CANONICAL_JOINT_NAMES])
    upper = np.asarray([urdf_gate["joint_limits"][name][1] for name in CANONICAL_JOINT_NAMES])
    for waypoint in range(horizon):
        target = robot.data.joint_pos.clone()
        target[:, joint_ids] = trajectories[:, waypoint]
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        for _ in range(int(step_schedule[waypoint])):
            sim.step(render=not args_cli.headless)
            scene.update(sim_dt)
        measured = _torch(robot.data.joint_pos)[:, joint_ids]
        max_tracking_error = torch.maximum(
            max_tracking_error,
            torch.amax(torch.abs(measured - trajectories[:, waypoint]), dim=-1),
        )
        forces = np.concatenate(
            [
                wp.to_torch(
                    view.get_net_contact_forces(dt=sim_dt).reshape((count, 1, 3))
                )
                .detach()
                .cpu()
                .numpy()
                for view in contact_views
            ],
            axis=1,
        )
        measured_cpu = measured.detach().cpu().numpy()
        actual_joint_violation |= ((measured_cpu < lower) | (measured_cpu > upper)).any(axis=1)
        force_norms = np.linalg.norm(forces, axis=-1)
        max_contact_force = np.maximum(max_contact_force, force_norms.max(axis=-1))
        max_contact_force_by_body = np.maximum(
            max_contact_force_by_body, force_norms
        )
        classified = classify_contact_forces(
            contact_body_names,
            forces,
            args_cli.contact_force_threshold,
            infer_interarm=args_cli.native_self_collisions,
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
        max_body_index = int(np.argmax(max_contact_force_by_body[index]))
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
                "max_contact_body": contact_body_names[max_body_index],
                "max_contact_body_force_n": float(
                    max_contact_force_by_body[index, max_body_index]
                ),
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
        "contact_classification": "exact per-rigid-body net force; native self collisions disabled by default",
        "contact_detection_scope": "robot-versus-world",
        "self_collision_authority": "MPD production validator intraarm/interarm pairs",
        "contact_classification_is_safety_authority": not args_cli.native_self_collisions,
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
        "timing_mode": "fixed_repeat" if args_cli.action_repeat > 0 else "artifact_timestamps",
        "physics_step_schedule": step_schedule.tolist(),
        "contact_force_threshold": args_cli.contact_force_threshold,
        "native_self_collisions_enabled": args_cli.native_self_collisions,
        "contact_body_names": contact_body_names,
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
