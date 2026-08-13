"""Replay one MPD trajectory in IsaacLab and export visual evidence.

Launch with IsaacLab's Python wrapper, for example:

    /home/eric/IsaacLab/isaaclab.sh -p scripts/isaaclab/replay_mpd_trajectory.py \
        --input scripts/inference/logs/run/isaaclab-trajectories-000.pt \
        --trajectory_index 0 \
        --output_video scripts/inference/logs/run/replay-000.mp4 \
        --screenshot_path scripts/inference/logs/run/replay-000.png \
        --headless --enable_cameras
"""

from __future__ import annotations

"""Launch Isaac Sim before importing IsaacLab runtime modules."""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one MPD trajectory with IsaacLab camera output.")
    parser.add_argument("--input", type=Path, required=True, help="Torch file written by MPD inference.")
    parser.add_argument(
        "--trajectory_source",
        choices=("best", "batch"),
        default="best",
        help="Replay q_trajs_pos_best when available, or replay q_trajs_pos[trajectory_index].",
    )
    parser.add_argument("--trajectory_index", type=int, default=0, help="Trajectory index inside q_trajs_pos batch.")
    parser.add_argument("--output_video", type=Path, default=None, help="Path to write an mp4 replay.")
    parser.add_argument("--screenshot_path", type=Path, default=None, help="Path to write the final RGB frame.")
    parser.add_argument("--output_json", type=Path, default=None, help="Optional replay metadata JSON path.")
    parser.add_argument(
        "--robot_usd",
        default=None,
        help="Optional Panda USD path or URL. Defaults to the Isaac Sim FrankaPanda asset.",
    )
    parser.add_argument("--action_repeat", type=int, default=4, help="Physics steps per trajectory waypoint.")
    parser.add_argument("--video_fps", type=float, default=24.0, help="Output mp4 frame rate.")
    parser.add_argument("--width", type=int, default=960, help="Camera image width.")
    parser.add_argument("--height", type=int, default=540, help="Camera image height.")
    parser.add_argument(
        "--camera_eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Optional world-space camera position. Must be used together with --camera_target.",
    )
    parser.add_argument(
        "--camera_target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Optional world-space point observed by the camera. Must be used together with --camera_eye.",
    )
    parser.add_argument(
        "--graceful_shutdown",
        action="store_true",
        help="Run Isaac Sim cleanup on exit. Default is fast process exit after outputs are written.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output_video is None and args.screenshot_path is None:
        parser.error("At least one of --output_video or --screenshot_path is required.")
    if args.action_repeat < 1:
        parser.error("--action_repeat must be >= 1.")
    if (args.camera_eye is None) != (args.camera_target is None):
        parser.error("--camera_eye and --camera_target must be provided together.")
    args.enable_cameras = True
    return args


args_cli = parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.camera import Camera
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


def _log(message: str) -> None:
    print(f"[mpd-replay] {message}", flush=True)


def _load_payload(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        return payload.float().contiguous(), {}
    if not isinstance(payload, dict) or "q_trajs_pos" not in payload:
        raise TypeError(f"{path} must contain a tensor or dict with q_trajs_pos.")
    q_trajs_pos = payload["q_trajs_pos"]
    if not isinstance(q_trajs_pos, torch.Tensor):
        raise TypeError(f"q_trajs_pos must be a torch.Tensor, got {type(q_trajs_pos)!r}.")
    metadata = {key: value for key, value in payload.items() if key != "q_trajs_pos"}
    return q_trajs_pos.float().contiguous(), metadata


def _metadata_tensor(metadata: dict[str, Any], key: str) -> torch.Tensor | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.float().contiguous()


def _normalize_best_trajectory(q_trajs_pos_best: torch.Tensor) -> torch.Tensor:
    if q_trajs_pos_best.ndim == 2:
        return q_trajs_pos_best
    if q_trajs_pos_best.ndim == 3:
        if q_trajs_pos_best.shape[1] == 1:
            return q_trajs_pos_best[:, 0]
        if q_trajs_pos_best.shape[0] == 1:
            return q_trajs_pos_best[0]
    raise ValueError(f"q_trajs_pos_best must have shape [H, D], got {tuple(q_trajs_pos_best.shape)}.")


PANDA_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
PANDA_CFG.spawn.usd_path = (
    args_cli.robot_usd or f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
)
PANDA_CFG.spawn.activate_contact_sensors = True
PANDA_CFG.spawn.rigid_props.disable_gravity = True


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    """Single-env Panda scene for visual replay."""

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot = PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    replay_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ReplayCamera",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1.0e5),
        ),
    )


def _spawn_scene_obstacles(scene_payload: dict[str, Any] | None) -> dict[str, Any]:
    scene_payload = scene_payload or {}
    obstacles = scene_payload.get("obstacles") or []
    obstacle_types = []

    for obstacle_idx, obstacle in enumerate(obstacles):
        obstacle_type = obstacle.get("type")
        obstacle_types.append(str(obstacle_type))
        prim_path = f"/World/envs/env_.*/MpdObstacle_{obstacle_idx:03d}"
        position = tuple(float(value) for value in obstacle.get("position", [0.0, 0.0, 0.0]))
        orientation = tuple(float(value) for value in obstacle.get("orientation", [1.0, 0.0, 0.0, 0.0]))
        common_cfg = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            "mass_props": sim_utils.MassPropertiesCfg(mass=1.0),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
        }
        if obstacle_type == "sphere":
            obstacle_cfg = sim_utils.SphereCfg(
                radius=float(obstacle["radius"]),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.58, 0.12), metallic=0.0),
                **common_cfg,
            )
        elif obstacle_type == "box":
            obstacle_cfg = sim_utils.CuboidCfg(
                size=tuple(float(value) for value in obstacle["size"]),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.47, 0.50), metallic=0.0),
                **common_cfg,
            )
        else:
            raise NotImplementedError(f"Unsupported replay obstacle type: {obstacle_type!r}")
        obstacle_cfg.func(prim_path, obstacle_cfg, translation=position, orientation=orientation)

    return {"n_obstacles": len(obstacles), "obstacle_types": sorted(set(obstacle_types))}


def _resolve_panda_joint_ids(robot, trajectory_dof: int) -> tuple[list[int] | slice, list[int]]:
    arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]
    finger_joint_ids = robot.find_joints(["panda_finger_joint.*"])[0]
    if trajectory_dof == len(arm_joint_ids):
        return arm_joint_ids, finger_joint_ids
    if trajectory_dof == robot.num_joints:
        return slice(None), finger_joint_ids
    raise ValueError(f"Trajectory DoF {trajectory_dof} does not match Panda joints.")


def _apply_joint_targets(robot, q_pos: torch.Tensor, joint_ids: list[int] | slice, finger_joint_ids: list[int]) -> None:
    target = robot.data.joint_pos.clone()
    target[:, joint_ids] = q_pos
    if finger_joint_ids:
        target[:, finger_joint_ids] = 0.04
    robot.set_joint_position_target(target)


def _reset_robot(robot, scene: InteractiveScene, q_pos_start: torch.Tensor, joint_ids, finger_joint_ids) -> None:
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    joint_pos[:, joint_ids] = q_pos_start
    if finger_joint_ids:
        joint_pos[:, finger_joint_ids] = 0.04
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    scene.reset()


def _position_camera(camera: Camera, sim, env_name: str, camera_eye=None, camera_target=None):
    if camera_eye is not None:
        eye_values = camera_eye
        target_values = camera_target
    elif env_name.startswith("EnvWarehouse"):
        eye_values = [2.0, -2.0, 1.45]
        target_values = [0.45, 0.0, 0.35]
    else:
        eye_values = [2.4, -2.4, 1.85]
        target_values = [0.15, 0.0, 0.45]
    eye = torch.tensor([eye_values], dtype=torch.float32, device=sim.device)
    target = torch.tensor([target_values], dtype=torch.float32, device=sim.device)
    camera.set_world_poses_from_view(eye, target)
    return camera, list(eye_values), list(target_values)


def _capture_rgb(camera: Camera, sim_dt: float) -> np.ndarray:
    camera.update(dt=sim_dt)
    rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _write_screenshot(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write screenshot to {path}")


def run_replay() -> dict[str, Any]:
    _log(f"loading payload: {args_cli.input}")
    q_trajs_pos_cpu, metadata = _load_payload(args_cli.input)
    horizon, batch, dof = q_trajs_pos_cpu.shape

    q_trajs_pos_best_cpu = _metadata_tensor(metadata, "q_trajs_pos_best")
    if args_cli.trajectory_source == "best" and q_trajs_pos_best_cpu is not None:
        q_traj_cpu = _normalize_best_trajectory(q_trajs_pos_best_cpu)
        q_pos_start_cpu = q_traj_cpu[0]
        trajectory_source = "best"
        trajectory_index = None
    else:
        if args_cli.trajectory_source == "best":
            _log("q_trajs_pos_best not found in payload; falling back to q_trajs_pos batch trajectory")
        if args_cli.trajectory_index < 0 or args_cli.trajectory_index >= batch:
            raise IndexError(f"--trajectory_index must be in [0, {batch - 1}], got {args_cli.trajectory_index}.")

        q_pos_starts_cpu = _metadata_tensor(metadata, "q_pos_starts")
        if q_pos_starts_cpu is None:
            q_pos_start_cpu = q_trajs_pos_cpu[0, args_cli.trajectory_index]
        else:
            q_pos_start_cpu = q_pos_starts_cpu[args_cli.trajectory_index]

        q_traj_cpu = q_trajs_pos_cpu[:, args_cli.trajectory_index]
        trajectory_source = "batch"
        trajectory_index = int(args_cli.trajectory_index)

    horizon, dof = q_traj_cpu.shape
    env_name = str(metadata.get("env_name", ""))

    _log(
        f"creating simulation: env={env_name}, source={trajectory_source}, horizon={horizon}, batch={batch}, dof={dof}"
    )
    sim_cfg = sim_utils.SimulationCfg(dt=float(metadata.get("dt", 0.005)), device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.4, -2.4, 1.8], [0.2, 0.0, 0.45])

    _log("creating scene")
    scene_cfg = ReplaySceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    _log("spawning obstacles")
    obstacle_summary = _spawn_scene_obstacles(metadata.get("scene"))
    _log("resetting simulation")
    sim.reset()

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    joint_ids, finger_joint_ids = _resolve_panda_joint_ids(robot, dof)
    q_traj = q_traj_cpu.to(device=sim.device)
    q_pos_start = q_pos_start_cpu.reshape(1, -1).to(device=sim.device)

    _log("positioning camera")
    camera, camera_eye, camera_target = _position_camera(
        scene["replay_camera"],
        sim,
        env_name,
        camera_eye=args_cli.camera_eye,
        camera_target=args_cli.camera_target,
    )
    _log("resetting robot")
    _reset_robot(robot, scene, q_pos_start, joint_ids, finger_joint_ids)
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim_dt)

    _log("capturing trajectory")
    frames = [_capture_rgb(camera, sim_dt)]
    for step_idx in range(horizon):
        _apply_joint_targets(robot, q_traj[step_idx].reshape(1, -1), joint_ids, finger_joint_ids)
        scene.write_data_to_sim()
        for _ in range(args_cli.action_repeat):
            sim.step(render=True)
            scene.update(sim_dt)
        frames.append(_capture_rgb(camera, sim_dt))

    if args_cli.screenshot_path is not None:
        _log(f"writing screenshot: {args_cli.screenshot_path}")
        _write_screenshot(args_cli.screenshot_path, frames[-1])
    if args_cli.output_video is not None:
        _log(f"writing video: {args_cli.output_video}")
        _write_video(args_cli.output_video, frames, args_cli.video_fps)

    summary = {
        "input": args_cli.input.as_posix(),
        "trajectory_source": trajectory_source,
        "trajectory_index": trajectory_index,
        "env_name": env_name,
        "robot_usd": PANDA_CFG.spawn.usd_path,
        "trajectory_horizon": int(horizon),
        "trajectory_batch": int(batch),
        "trajectory_dof": int(dof),
        "n_frames": len(frames),
        "output_video": args_cli.output_video.as_posix() if args_cli.output_video else None,
        "camera_eye": camera_eye,
        "camera_target": camera_target,
        "screenshot_path": args_cli.screenshot_path.as_posix() if args_cli.screenshot_path else None,
        **obstacle_summary,
    }
    if args_cli.output_json is not None:
        args_cli.output_json.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("replay complete")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    exit_code = 0
    try:
        run_replay()
    except BaseException:
        exit_code = 1
        traceback.print_exc()

    if args_cli.graceful_shutdown:
        _log("closing simulation app")
        simulation_app.close(wait_for_replicator=False)
        raise SystemExit(exit_code)

    _log(f"fast process exit: {exit_code}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
