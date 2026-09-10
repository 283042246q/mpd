#!/usr/bin/env python3
"""Replay one Marvin trajectory with both TCP paths and goal frames."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

from marvin_bimanual_asset import load_inference_artifact, validate_marvin_urdf


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--evaluation", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
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
    parser.add_argument("--video-fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument(
        "--camera-eye",
        nargs=3,
        type=float,
        default=(4.8, 0.0, 2.2),
        help="Camera position; default is centered in the aisle between the shelves.",
    )
    parser.add_argument(
        "--camera-target",
        nargs=3,
        type=float,
        default=(0.45, 0.0, 0.0),
        help="Camera look-at target in the bimanual workspace.",
    )
    parser.add_argument("--graceful-shutdown", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.action_repeat < 0 or args.video_fps <= 0.0 or args.width < 1 or args.height < 1:
        parser.error(
            "action repeat must be non-negative; video fps, width, and height must be positive"
        )
    args.enable_cameras = args.output_video is not None or args.screenshot is not None
    return args


args_cli = parse_args()
artifact = load_inference_artifact(args_cli.artifact)
urdf_gate = validate_marvin_urdf()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.camera import Camera
from isaaclab.utils import configclass
from isaacsim.core.experimental.utils.app import enable_extension

from marvin_bimanual_asset import (
    build_marvin_articulation_cfg,
    convert_marvin_urdf_to_usd,
    resolve_canonical_joint_ids,
    resolve_tcp_body_ids,
    sha256_file,
    spawn_scene_obstacles,
    tcp_poses_from_body_state,
    trajectory_physics_step_schedule,
    validate_marvin_usd,
)


enable_extension("isaacsim.util.debug_draw")
from isaacsim.util.debug_draw import _debug_draw as omni_debug_draw  # noqa: E402


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


def _asset_metadata():
    if args_cli.robot_usd is None:
        return convert_marvin_urdf_to_usd(
            args_cli.asset_cache, force=args_cli.force_usd_conversion
        )
    return validate_marvin_usd(args_cli.robot_usd)


def _evaluation_trajectory(index):
    if args_cli.evaluation is None:
        return None
    report = json.loads(args_cli.evaluation.read_text())
    if report.get("schema") != "marvin_bimanual_isaaclab_evaluation/v1":
        raise ValueError("--evaluation is not a Marvin bimanual Isaac Lab report")
    if report.get("artifact_hashes") != artifact.hashes:
        raise ValueError("--evaluation was produced from a different inference artifact")
    matching = [
        item for item in report.get("trajectories", [])
        if int(item.get("top_k_index", -1)) == index
    ]
    if len(matching) != 1:
        raise IndexError("evaluation JSON has no matching Top-K trajectory")
    entry = matching[0]
    if int(entry.get("candidate_index", -1)) != int(artifact.top_k_candidate_indices[index]):
        raise ValueError("evaluation candidate index differs from inference artifact")
    return entry


def _write_video(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), args_cli.video_fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _capture(camera: Camera, dt):
    camera.update(dt=dt)
    rgb = _torch(camera.data.output["rgb"])[0, ..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _draw_path(draw, points, color):
    if len(points) < 2:
        return
    starts = [tuple(map(float, point)) for point in points[:-1]]
    ends = [tuple(map(float, point)) for point in points[1:]]
    draw.draw_lines(starts, ends, [color] * len(starts), [4.0] * len(starts))


def _draw_goal_frames(draw, poses, axis_length=0.10):
    colors = (
        (1.0, 0.1, 0.1, 1.0),
        (0.1, 1.0, 0.1, 1.0),
        (0.1, 0.3, 1.0, 1.0),
    )
    starts, ends, line_colors = [], [], []
    for pose in np.asarray(poses):
        origin = pose[:3, 3]
        for axis in range(3):
            starts.append(tuple(map(float, origin)))
            ends.append(tuple(map(float, origin + axis_length * pose[:3, axis])))
            line_colors.append(colors[axis])
    draw.draw_lines(starts, ends, line_colors, [5.0] * len(starts))


def _draw_collision_crosses(draw, points, radius=0.045):
    starts, ends = [], []
    for point in np.asarray(points):
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = radius
            starts.append(tuple(map(float, point - offset)))
            ends.append(tuple(map(float, point + offset)))
    draw.draw_lines(
        starts,
        ends,
        [(1.0, 0.0, 0.0, 1.0)] * len(starts),
        [8.0] * len(starts),
    )


def _tcp_positions(robot, tcp_ids, env_origin):
    body_positions = _torch(robot.data.body_pos_w)[0, list(tcp_ids)].detach().cpu().numpy()
    body_positions -= _torch(env_origin).detach().cpu().numpy()[None]
    body_quaternions = _torch(robot.data.body_quat_w)[0, list(tcp_ids)].detach().cpu().numpy()
    positions, _ = tcp_poses_from_body_state(body_positions, body_quaternions)
    return positions


def run_replay():
    index = args_cli.trajectory_index
    if index < 0 or index >= artifact.top_k_positions.shape[0]:
        raise IndexError(
            f"--trajectory-index must be in [0,{artifact.top_k_positions.shape[0] - 1}]"
        )
    trajectory_cpu = artifact.top_k_positions[index]
    evaluation_entry = _evaluation_trajectory(index)
    collision_waypoint = (
        int(evaluation_entry.get("first_contact_waypoint", -1))
        if evaluation_entry is not None
        else -1
    )
    asset = _asset_metadata()
    robot_cfg = build_marvin_articulation_cfg(
        asset["usd_path"], enabled_self_collisions=False
    )

    @configclass
    class MarvinReplaySceneCfg(InteractiveSceneCfg):
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=1800.0),
        )
        robot = robot_cfg
        if args_cli.enable_cameras:
            camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/ReplayCamera",
                update_period=0.0,
                height=args_cli.height,
                width=args_cli.width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.05, 100.0),
                ),
            )

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    )
    scene = InteractiveScene(MarvinReplaySceneCfg(num_envs=1, env_spacing=3.0))
    obstacle_summary = spawn_scene_obstacles(sim_utils, artifact.scene)
    sim.reset()
    robot = scene["robot"]
    camera = scene["camera"] if args_cli.enable_cameras else None
    joint_ids = resolve_canonical_joint_ids(robot)
    tcp_ids = resolve_tcp_body_ids(robot)
    trajectory = torch.as_tensor(trajectory_cpu, dtype=torch.float32, device=sim.device)
    sim_dt = sim.get_physics_dt()
    step_schedule = trajectory_physics_step_schedule(
        artifact.time_from_start, sim_dt, args_cli.action_repeat
    )

    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor([args_cli.camera_eye], dtype=torch.float32, device=sim.device),
            torch.tensor([args_cli.camera_target], dtype=torch.float32, device=sim.device),
        )
    initial = robot.data.default_joint_pos.clone()
    initial[:, joint_ids] = trajectory[0]
    robot.write_joint_state_to_sim(initial, torch.zeros_like(initial))
    robot.set_joint_position_target(initial)
    scene.reset()
    scene.write_data_to_sim()
    sim.step(render=args_cli.enable_cameras)
    scene.update(sim_dt)

    tcp_paths = [[], []]
    for waypoint in trajectory:
        state = robot.data.default_joint_pos.clone()
        state[:, joint_ids] = waypoint[None]
        robot.write_joint_state_to_sim(state, torch.zeros_like(state))
        sim.forward()
        scene.update(0.0)
        positions = _tcp_positions(robot, tcp_ids, scene.env_origins[0])
        for arm in range(2):
            tcp_paths[arm].append(positions[arm].copy())
    tcp_paths = [np.asarray(points) for points in tcp_paths]

    draw = omni_debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()
    _draw_path(draw, tcp_paths[0], (0.15, 0.55, 1.0, 1.0))
    _draw_path(draw, tcp_paths[1], (0.95, 0.35, 0.15, 1.0))

    if artifact.ee_goal_pose is not None:
        _draw_goal_frames(draw, artifact.ee_goal_pose)

    initial[:, joint_ids] = trajectory[0]
    robot.write_joint_state_to_sim(initial, torch.zeros_like(initial))
    robot.set_joint_position_target(initial)
    frames = [_capture(camera, sim_dt)] if camera is not None else []
    for waypoint_index, waypoint in enumerate(trajectory):
        target = robot.data.joint_pos.clone()
        target[:, joint_ids] = waypoint[None]
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        for _ in range(int(step_schedule[waypoint_index])):
            sim.step(render=args_cli.enable_cameras)
            scene.update(sim_dt)
        if waypoint_index == collision_waypoint:
            _draw_collision_crosses(
                draw,
                np.stack((tcp_paths[0][waypoint_index], tcp_paths[1][waypoint_index])),
            )
        if camera is not None:
            frame = _capture(camera, sim_dt)
            if waypoint_index == collision_waypoint:
                cv2.putText(
                    frame,
                    f"FIRST COLLISION waypoint {waypoint_index}",
                    (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 30, 30),
                    2,
                    cv2.LINE_AA,
                )
            frames.append(frame)

    if args_cli.output_video is not None:
        _write_video(args_cli.output_video, frames)
    if args_cli.screenshot is not None:
        args_cli.screenshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args_cli.screenshot), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not create {args_cli.screenshot}")
    summary = {
        "schema": "marvin_bimanual_isaaclab_replay/v1",
        "artifact": artifact.root.as_posix(),
        "trajectory_index": index,
        "candidate_index": int(artifact.top_k_candidate_indices[index]),
        "horizon": int(trajectory.shape[0]),
        "native_self_collisions_enabled": False,
        "timing_mode": "fixed_repeat" if args_cli.action_repeat > 0 else "artifact_timestamps",
        "physics_dt": sim_dt,
        "action_repeat": args_cli.action_repeat,
        "physics_step_schedule": step_schedule.tolist(),
        "camera_eye": list(args_cli.camera_eye),
        "camera_target": list(args_cli.camera_target),
        "capture_enabled": bool(args_cli.enable_cameras),
        "joint_names": list(urdf_gate["joint_names"]),
        "isaac_native_joint_names": list(robot.joint_names),
        "canonical_joint_ids": list(joint_ids),
        "tcp_frame_names": list(urdf_gate["tcp_frame_names"]),
        "tcp_body_names": list(urdf_gate["tcp_body_names"]),
        "tcp_offsets_xyz": urdf_gate["tcp_offsets_xyz"],
        "left_tcp_path_points": int(tcp_paths[0].shape[0]),
        "right_tcp_path_points": int(tcp_paths[1].shape[0]),
        "first_collision_waypoint": collision_waypoint,
        "usd_path": asset["usd_path"].as_posix(),
        "usd_sha256": asset["usd_sha256"],
        "usd_bundle_sha256": asset["usd_bundle_sha256"],
        "artifact_hashes": artifact.hashes,
        "evaluation": evaluation_entry,
        "output_video": args_cli.output_video.as_posix() if args_cli.output_video else None,
        "screenshot": args_cli.screenshot.as_posix() if args_cli.screenshot else None,
        **obstacle_summary,
    }
    summary["output_hashes"] = {
        "video_sha256": sha256_file(args_cli.output_video) if args_cli.output_video else None,
        "screenshot_sha256": sha256_file(args_cli.screenshot) if args_cli.screenshot else None,
    }
    if args_cli.output_json is not None:
        args_cli.output_json.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output_json.write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return summary


def main():
    exit_code = 0
    try:
        run_replay()
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
