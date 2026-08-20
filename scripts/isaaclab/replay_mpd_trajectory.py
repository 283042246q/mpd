"""Replay one MPD trajectory or a Phase-4 replan timeline in IsaacLab.

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
    parser = argparse.ArgumentParser(description="Replay MPD trajectories with IsaacLab camera output.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Legacy Torch file written by MPD inference.")
    source.add_argument(
        "--manifest",
        type=Path,
        help="Phase-4 mpd_dynamic_replay JSON manifest containing multiple NPZ plans.",
    )
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
        "--prediction_horizon_s",
        type=float,
        default=3.0,
        help="Future dynamic-obstacle envelope horizon [s].",
    )
    parser.add_argument(
        "--prediction_samples",
        type=int,
        default=8,
        help="Number of translucent envelope samples per dynamic object.",
    )
    parser.add_argument(
        "--trajectory_history_s",
        type=float,
        default=30.0,
        help="How long superseded trajectories remain visible in gray [s].",
    )
    parser.add_argument(
        "--trajectory_line_width",
        type=float,
        default=4.0,
        help="Rendered trajectory line width in pixels.",
    )
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
    if args.prediction_horizon_s <= 0.0 or args.prediction_samples < 1:
        parser.error("prediction horizon and sample count must be positive.")
    if args.trajectory_history_s < 0.0 or args.trajectory_line_width <= 0.0:
        parser.error("trajectory history must be non-negative and line width must be positive.")
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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.camera import Camera
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from isaacsim.core.experimental.utils.app import enable_extension  # isort: skip

from dynamic_replay_timeline import (  # isort: skip
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
    DynamicReplayManifest,
    PlanRecord,
    active_plan_at,
    brake_event_at,
    latest_pending_plan_at,
    load_dynamic_replay_manifest,
    plan_base_color,
    predict_object,
    robot_position_at,
    segment_color,
    world_snapshot_at,
)

enable_extension("isaacsim.util.debug_draw")
from isaacsim.util.debug_draw import _debug_draw as omni_debug_draw  # noqa: E402, PLC0415


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
PANDA_CFG.spawn.usd_path = args_cli.robot_usd or f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
PANDA_CFG.spawn.activate_contact_sensors = True
PANDA_CFG.spawn.rigid_props.disable_gravity = True


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    """Single-env Panda scene for visual replay."""

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.75, 0.75, 0.75)),
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


_CURRENT_MARKER_INDEX = {"sphere": 0, "box": 1, "capsule": 2}
_HANDOFF_MARKER_INDEX = 3


def _create_dynamic_markers() -> VisualizationMarkers:
    current_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.95, 0.28, 0.04),
        emissive_color=(0.12, 0.02, 0.0),
        roughness=0.35,
        opacity=0.92,
    )
    handoff_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=COLOR_PURPLE[:3],
        emissive_color=(0.25, 0.02, 0.35),
        roughness=0.2,
        opacity=1.0,
    )
    cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/MpdDynamicReplay",
        markers={
            "current_sphere": sim_utils.SphereCfg(radius=0.5, visual_material=current_material),
            "current_box": sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0), visual_material=current_material),
            "current_capsule": sim_utils.CapsuleCfg(
                radius=0.5,
                height=1.0,
                axis="Z",
                visual_material=current_material,
            ),
            "handoff": sim_utils.SphereCfg(radius=0.045, visual_material=handoff_material),
        },
    )
    return VisualizationMarkers(cfg)


def _current_marker_scale(local_sdf: dict[str, Any]) -> np.ndarray:
    shape_type = local_sdf["type"]
    if shape_type == "sphere":
        diameter = 2.0 * float(local_sdf["radius"])
        return np.full(3, diameter, dtype=np.float32)
    if shape_type == "box":
        return np.asarray(local_sdf["size_xyz"], dtype=np.float32)
    diameter = 2.0 * float(local_sdf["radius"])
    return np.asarray([diameter, diameter, float(local_sdf["length"])], dtype=np.float32)


def _object_bounding_radius(local_sdf: dict[str, Any], inflation_m: float) -> float:
    shape_type = local_sdf["type"]
    if shape_type == "sphere":
        radius = float(local_sdf["radius"])
    elif shape_type == "box":
        radius = float(np.linalg.norm(np.asarray(local_sdf["size_xyz"], dtype=np.float64) * 0.5))
    else:
        radius = float(local_sdf["radius"]) + 0.5 * float(local_sdf["length"])
    return radius + float(inflation_m)


def _handoff_positions(
    manifest: DynamicReplayManifest,
    time_s: float,
    ee_paths: dict[str, np.ndarray],
) -> list[np.ndarray]:
    positions = []
    plans = {plan.plan_id: plan for plan in manifest.plans}
    for event in manifest.events:
        if event.event_type != "handoff" or time_s < event.time_s - 1.0:
            continue
        if event.position is not None:
            positions.append(np.asarray(event.position, dtype=np.float32))
            continue
        if event.plan_id is None or event.plan_id not in plans or event.plan_id not in ee_paths:
            continue
        plan = plans[event.plan_id]
        local_time = np.clip(
            event.time_s - plan.start_s,
            plan.trajectory.times_s[0],
            plan.trajectory.times_s[-1],
        )
        path = ee_paths[event.plan_id]
        positions.append(
            np.asarray(
                [np.interp(local_time, plan.trajectory.times_s, path[:, axis]) for axis in range(3)],
                dtype=np.float32,
            )
        )
    return positions


def _update_dynamic_markers(
    markers: VisualizationMarkers,
    manifest: DynamicReplayManifest,
    time_s: float,
    ee_paths: dict[str, np.ndarray],
) -> int:
    snapshot = world_snapshot_at(manifest, time_s)
    translations = []
    orientations = []
    scales = []
    indices = []
    for item in snapshot.objects:
        current = predict_object(item, snapshot.time_s, time_s)
        translations.append(current.position)
        orientations.append(current.orientation_xyzw)
        scales.append(_current_marker_scale(current.local_sdf))
        indices.append(_CURRENT_MARKER_INDEX[current.local_sdf["type"]])

    handoffs = _handoff_positions(manifest, time_s, ee_paths)
    for position in handoffs:
        translations.append(position)
        orientations.append(np.asarray([0.0, 0.0, 0.0, 1.0]))
        scales.append(np.ones(3, dtype=np.float32))
        indices.append(_HANDOFF_MARKER_INDEX)

    if not translations:
        markers.set_visibility(False)
        return 0
    markers.set_visibility(True)
    markers.visualize(
        translations=np.asarray(translations, dtype=np.float32),
        orientations=np.asarray(orientations, dtype=np.float32),
        scales=np.asarray(scales, dtype=np.float32),
        marker_indices=np.asarray(indices, dtype=np.int32),
    )
    return len(snapshot.objects)


def _append_wire_sphere(
    starts: list,
    ends: list,
    colors: list,
    widths: list,
    center: np.ndarray,
    radius: float,
) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 25, dtype=np.float64)
    for fixed_axis in range(3):
        varying_axes = [axis for axis in range(3) if axis != fixed_axis]
        points = np.repeat(np.asarray(center, dtype=np.float64)[None, :], len(angles), axis=0)
        points[:, varying_axes[0]] += radius * np.cos(angles)
        points[:, varying_axes[1]] += radius * np.sin(angles)
        for index in range(len(points) - 1):
            starts.append(points[index].tolist())
            ends.append(points[index + 1].tolist())
            colors.append(COLOR_YELLOW)
            widths.append(max(1.0, args_cli.trajectory_line_width * 0.5))


def _append_prediction_envelopes(
    starts: list,
    ends: list,
    colors: list,
    widths: list,
    manifest: DynamicReplayManifest,
    time_s: float,
) -> int:
    snapshot = world_snapshot_at(manifest, time_s)
    prediction_end = min(time_s + args_cli.prediction_horizon_s, snapshot.valid_until_s)
    if prediction_end <= time_s:
        return 0
    count = 0
    for item in snapshot.objects:
        for prediction_time in np.linspace(
            time_s,
            prediction_end,
            args_cli.prediction_samples + 1,
            dtype=np.float64,
        )[1:]:
            predicted = predict_object(item, snapshot.time_s, float(prediction_time))
            radius = _object_bounding_radius(predicted.local_sdf, predicted.inflation_m)
            _append_wire_sphere(starts, ends, colors, widths, predicted.position, radius)
            count += 1
    return count


def _draw_trajectory_lines(
    draw_interface,
    manifest: DynamicReplayManifest,
    time_s: float,
    ee_paths: dict[str, np.ndarray],
) -> tuple[int, int]:
    starts = []
    ends = []
    colors = []
    widths = []
    for plan in manifest.plans:
        color = plan_base_color(plan, manifest, time_s)
        if color is None:
            continue
        if (
            plan.active_until_s is not None
            and time_s - plan.active_until_s > args_cli.trajectory_history_s
            and color == COLOR_GRAY
        ):
            continue
        path = ee_paths[plan.plan_id]
        for index in range(len(path) - 1):
            starts.append(path[index].tolist())
            ends.append(path[index + 1].tolist())
            colors.append(
                segment_color(
                    plan,
                    float(plan.trajectory.times_s[index]),
                    color,
                )
            )
            widths.append(float(args_cli.trajectory_line_width))
    trajectory_segment_count = len(starts)
    prediction_count = _append_prediction_envelopes(
        starts,
        ends,
        colors,
        widths,
        manifest,
        time_s,
    )
    draw_interface.clear_lines()
    if starts:
        draw_interface.draw_lines(starts, ends, colors, widths)
    return trajectory_segment_count, prediction_count


def _rgba_to_rgb8(color: tuple[float, ...]) -> tuple[int, int, int]:
    return tuple(int(round(255.0 * component)) for component in color[:3])


def _overlay_dynamic_hud(
    frame: np.ndarray,
    manifest: DynamicReplayManifest,
    time_s: float,
) -> np.ndarray:
    output = frame.copy()
    snapshot = world_snapshot_at(manifest, time_s)
    active = active_plan_at(manifest, time_s)
    pending = latest_pending_plan_at(manifest, time_s)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def text(value: str, position: tuple[int, int], scale: float = 0.55, color=(245, 245, 245)):
        cv2.putText(output, value, (position[0] + 2, position[1] + 2), font, scale, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(output, value, position, font, scale, color, 1, cv2.LINE_AA)

    text(f"Phase 4 dynamic replay   t={time_s:05.2f}s   world={snapshot.world_version}", (22, 30), 0.62)
    text(f"active={active.plan_id if active else '-'}   latest={pending.plan_id if pending else '-'}", (22, 55))
    legend = (
        ("obsolete", COLOR_GRAY),
        ("active", COLOR_BLUE),
        ("latest", COLOR_GREEN),
        ("prediction", COLOR_YELLOW),
        ("rejected/collision", COLOR_RED),
        ("handoff", COLOR_PURPLE),
    )
    x, y = 22, output.shape[0] - 24
    for label, color in legend:
        label_width = cv2.getTextSize(label, font, 0.42, 1)[0][0]
        item_width = label_width + 48
        if x + item_width > output.shape[1] - 20:
            x = 22
            y -= 25
        box_color = _rgba_to_rgb8(color)
        cv2.rectangle(output, (x, y - 13), (x + 16, y + 3), box_color, thickness=-1)
        text(label, (x + 22, y + 1), 0.42)
        x += item_width

    brake = brake_event_at(manifest, time_s)
    if brake is not None and int((time_s - brake.time_s) * 8.0) % 2 == 0:
        cv2.rectangle(output, (5, 5), (output.shape[1] - 6, output.shape[0] - 6), (255, 0, 0), 12)
        reason = f"  {brake.reason}" if brake.reason else ""
        text(f"SAFETY BRAKE{reason}", (22, 88), 0.9, (255, 40, 40))
    return output


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


def _teleport_robot(robot, q_pos: torch.Tensor, joint_ids, finger_joint_ids) -> None:
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    joint_pos[:, joint_ids] = q_pos
    if finger_joint_ids:
        joint_pos[:, finger_joint_ids] = 0.04
    robot.write_joint_state_to_sim(position=joint_pos, velocity=joint_vel)
    robot.set_joint_position_target(joint_pos)


def _body_positions_tensor(robot):
    body_positions = robot.data.body_pos_w
    return body_positions.torch if hasattr(body_positions, "torch") else body_positions


def _compute_ee_paths(
    robot,
    scene: InteractiveScene,
    sim,
    manifest: DynamicReplayManifest,
    joint_ids,
    finger_joint_ids,
) -> dict[str, np.ndarray]:
    hand_indices = robot.find_bodies("panda_hand")[0]
    if not hand_indices:
        raise RuntimeError("IsaacLab Panda asset has no panda_hand body")
    hand_index = hand_indices[0]
    result = {}
    for plan in manifest.plans:
        points = []
        for q_pos_cpu in plan.trajectory.positions:
            q_pos = torch.as_tensor(q_pos_cpu, dtype=torch.float32, device=sim.device).reshape(1, -1)
            _teleport_robot(robot, q_pos, joint_ids, finger_joint_ids)
            sim.forward()
            scene.update(0.0)
            points.append(_body_positions_tensor(robot)[0, hand_index].detach().cpu().numpy().copy())
        result[plan.plan_id] = np.asarray(points, dtype=np.float32)
    return result


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


def _run_legacy_replay() -> dict[str, Any]:
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


def _open_video_writer(path: Path, fps: float, width: int, height: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def _run_dynamic_replay() -> dict[str, Any]:
    _log(f"loading dynamic replay manifest: {args_cli.manifest}")
    manifest = load_dynamic_replay_manifest(args_cli.manifest)
    dof = int(manifest.initial_q.shape[0])
    _log(
        "creating dynamic simulation: "
        f"env={manifest.env_name}, duration={manifest.duration_s:.2f}s, "
        f"plans={len(manifest.plans)}, dof={dof}"
    )
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.4, -2.4, 1.8], [0.2, 0.0, 0.45])

    _log("creating scene and dynamic visualization markers")
    scene_cfg = ReplaySceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    obstacle_summary = _spawn_scene_obstacles(manifest.static_scene)
    markers = _create_dynamic_markers()
    draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    sim.reset()

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    joint_ids, finger_joint_ids = _resolve_panda_joint_ids(robot, dof)
    camera, camera_eye, camera_target = _position_camera(
        scene["replay_camera"],
        sim,
        manifest.env_name,
        camera_eye=args_cli.camera_eye,
        camera_target=args_cli.camera_target,
    )

    _log("precomputing end-effector paths for colored trajectory overlays")
    ee_paths = _compute_ee_paths(
        robot,
        scene,
        sim,
        manifest,
        joint_ids,
        finger_joint_ids,
    )
    initial_q = torch.as_tensor(
        manifest.initial_q,
        dtype=torch.float32,
        device=sim.device,
    ).reshape(1, -1)
    _reset_robot(robot, scene, initial_q, joint_ids, finger_joint_ids)
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim_dt)

    frame_times = (
        np.arange(
            int(np.floor(manifest.duration_s * args_cli.video_fps)) + 1,
            dtype=np.float64,
        )
        / args_cli.video_fps
    )
    video_writer = None
    if args_cli.output_video is not None:
        _log(f"streaming video frames to: {args_cli.output_video}")
        video_writer = _open_video_writer(
            args_cli.output_video,
            args_cli.video_fps,
            args_cli.width,
            args_cli.height,
        )

    final_frame = None
    peak_dynamic_objects = 0
    peak_prediction_markers = 0
    peak_trajectory_segments = 0
    try:
        for frame_index, replay_time_s in enumerate(frame_times):
            q_pos = torch.as_tensor(
                robot_position_at(manifest, float(replay_time_s)),
                dtype=torch.float32,
                device=sim.device,
            ).reshape(1, -1)
            _teleport_robot(robot, q_pos, joint_ids, finger_joint_ids)
            n_objects = _update_dynamic_markers(
                markers,
                manifest,
                float(replay_time_s),
                ee_paths,
            )
            n_segments, n_predictions = _draw_trajectory_lines(
                draw_interface,
                manifest,
                float(replay_time_s),
                ee_paths,
            )
            peak_dynamic_objects = max(peak_dynamic_objects, n_objects)
            peak_prediction_markers = max(peak_prediction_markers, n_predictions)
            peak_trajectory_segments = max(peak_trajectory_segments, n_segments)

            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim_dt)
            frame = _capture_rgb(camera, sim_dt)
            final_frame = _overlay_dynamic_hud(frame, manifest, float(replay_time_s))
            if video_writer is not None:
                video_writer.write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))
            if frame_index and frame_index % max(1, int(args_cli.video_fps * 2.0)) == 0:
                _log(f"rendered {frame_index + 1}/{len(frame_times)} frames")
    finally:
        if video_writer is not None:
            video_writer.release()
        draw_interface.clear_lines()

    if final_frame is None:
        raise RuntimeError("Dynamic replay produced no frames")
    if args_cli.screenshot_path is not None:
        _log(f"writing screenshot: {args_cli.screenshot_path}")
        _write_screenshot(args_cli.screenshot_path, final_frame)

    summary = {
        "mode": "dynamic_replay",
        "manifest": manifest.path.as_posix(),
        "schema": "mpd_dynamic_replay",
        "schema_version": 1,
        "env_name": manifest.env_name,
        "frame_id": manifest.frame_id,
        "robot_usd": PANDA_CFG.spawn.usd_path,
        "duration_s": manifest.duration_s,
        "video_fps": args_cli.video_fps,
        "n_frames": len(frame_times),
        "n_plans": len(manifest.plans),
        "n_world_snapshots": len(manifest.world_snapshots),
        "n_handoff_events": sum(event.event_type == "handoff" for event in manifest.events),
        "n_brake_events": sum(event.event_type == "brake" for event in manifest.events),
        "peak_dynamic_objects": peak_dynamic_objects,
        "peak_prediction_markers": peak_prediction_markers,
        "peak_trajectory_segments": peak_trajectory_segments,
        "prediction_horizon_s": args_cli.prediction_horizon_s,
        "prediction_samples": args_cli.prediction_samples,
        "output_video": args_cli.output_video.as_posix() if args_cli.output_video else None,
        "screenshot_path": args_cli.screenshot_path.as_posix() if args_cli.screenshot_path else None,
        "camera_eye": camera_eye,
        "camera_target": camera_target,
        "colors": {
            "obsolete": COLOR_GRAY,
            "active": COLOR_BLUE,
            "latest": COLOR_GREEN,
            "prediction": COLOR_YELLOW,
            "rejected_or_collision": COLOR_RED,
            "handoff": COLOR_PURPLE,
        },
        **obstacle_summary,
    }
    if args_cli.output_json is not None:
        args_cli.output_json.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("dynamic replay complete")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_replay() -> dict[str, Any]:
    """Dispatch without changing the existing single-trajectory entry point."""

    if args_cli.manifest is not None:
        return _run_dynamic_replay()
    return _run_legacy_replay()


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
