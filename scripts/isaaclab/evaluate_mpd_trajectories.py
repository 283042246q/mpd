"""Evaluate MPD joint-space trajectories in IsaacLab.

This script intentionally runs outside the MPD Python environment. Launch it
with IsaacLab's Python wrapper, for example:

    /home/eric/IsaacLab_ori/isaaclab.sh -p scripts/isaaclab/evaluate_mpd_trajectories.py \
        --input logs/trajectories.pt \
        --output logs/isaaclab_statistics.json \
        --headless
"""

from __future__ import annotations

"""Launch Isaac Sim before importing IsaacLab runtime modules."""

import argparse
import json
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MPD trajectories with IsaacLab.")
    parser.add_argument("--input", type=Path, help="Path to a torch file containing q_trajs_pos.")
    parser.add_argument("--output", type=Path, help="Path to write IsaacLab statistics JSON.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of IsaacLab environments. Defaults to B.")
    parser.add_argument("--robot_name", default="panda", choices=("panda",), help="Robot asset to use.")
    parser.add_argument("--action_repeat", type=int, default=4, help="Physics steps per MPD trajectory waypoint.")
    parser.add_argument("--contact_force_threshold", type=float, default=1.0, help="Contact force norm threshold.")
    parser.add_argument("--stop_robot_if_in_contact", action="store_true", help="Hold collided envs at current state.")
    parser.add_argument("--make_video", action="store_true", help="Reserved for the video stage; currently ignored.")
    parser.add_argument("--video_path", type=Path, default=None, help="Reserved video output path.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.input is None:
        parser.error("--input is required")
    if args.output is None:
        parser.error("--output is required")
    return args


args_cli = parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


def _load_payload(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        q_trajs_pos = payload
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict):
        if "q_trajs_pos" not in payload:
            raise KeyError(f"{path} does not contain required key 'q_trajs_pos'")
        q_trajs_pos = payload["q_trajs_pos"]
        metadata = {key: value for key, value in payload.items() if key != "q_trajs_pos"}
    else:
        raise TypeError(f"Unsupported input payload type: {type(payload)!r}")

    if not isinstance(q_trajs_pos, torch.Tensor):
        raise TypeError(f"'q_trajs_pos' must be a torch.Tensor, got {type(q_trajs_pos)!r}")
    if q_trajs_pos.ndim != 3:
        raise ValueError(f"'q_trajs_pos' must have shape [H, B, D], got {tuple(q_trajs_pos.shape)}")
    if not torch.is_floating_point(q_trajs_pos):
        q_trajs_pos = q_trajs_pos.float()

    return q_trajs_pos.contiguous(), metadata


def _metadata_tensor(metadata: dict[str, Any], key: str) -> torch.Tensor | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.float().contiguous()


PANDA_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
PANDA_CFG.spawn.activate_contact_sensors = True
PANDA_CFG.spawn.rigid_props.disable_gravity = True


@configclass
class MpdTrajectorySceneCfg(InteractiveSceneCfg):
    """Scene used for MPD trajectory replay."""

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot = PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )


def _resolve_panda_joint_ids(robot, trajectory_dof: int) -> tuple[list[int] | slice, list[int]]:
    arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]
    finger_joint_ids = robot.find_joints(["panda_finger_joint.*"])[0]

    if trajectory_dof == len(arm_joint_ids):
        return arm_joint_ids, finger_joint_ids
    if trajectory_dof == robot.num_joints:
        return slice(None), finger_joint_ids

    raise ValueError(
        f"Trajectory DoF {trajectory_dof} does not match Panda arm DoF {len(arm_joint_ids)} "
        f"or full joint DoF {robot.num_joints}."
    )


def _apply_joint_targets(robot, q_pos: torch.Tensor, joint_ids: list[int] | slice, finger_joint_ids: list[int]) -> None:
    target = robot.data.joint_pos.clone()
    target[:, joint_ids] = q_pos
    if finger_joint_ids:
        target[:, finger_joint_ids] = 0.04
    robot.set_joint_position_target(target)


def _reset_robot(robot, scene: InteractiveScene, q_pos_starts: torch.Tensor, joint_ids, finger_joint_ids) -> None:
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    joint_pos[:, joint_ids] = q_pos_starts
    if finger_joint_ids:
        joint_pos[:, finger_joint_ids] = 0.04
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    scene.reset()


def _contact_mask(contact_sensor, threshold: float, num_trajectories: int) -> torch.Tensor:
    forces = contact_sensor.data.net_forces_w[:num_trajectories]
    if forces.numel() == 0:
        return torch.zeros(num_trajectories, dtype=torch.bool, device=forces.device)
    return torch.linalg.norm(forces, dim=-1).amax(dim=-1) > threshold


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def run_evaluation() -> dict[str, Any]:
    q_trajs_pos_cpu, metadata = _load_payload(args_cli.input)
    horizon, batch, dof = q_trajs_pos_cpu.shape

    num_envs = args_cli.num_envs or batch
    if num_envs != batch:
        raise ValueError(f"--num_envs must match trajectory batch B={batch}; got {num_envs}.")
    if args_cli.action_repeat < 1:
        raise ValueError("--action_repeat must be >= 1.")

    q_pos_starts_cpu = _metadata_tensor(metadata, "q_pos_starts")
    if q_pos_starts_cpu is None:
        q_pos_starts_cpu = q_trajs_pos_cpu[0]
    if tuple(q_pos_starts_cpu.shape) != (batch, dof):
        raise ValueError(f"'q_pos_starts' must have shape {(batch, dof)}, got {tuple(q_pos_starts_cpu.shape)}")

    sim_cfg = sim_utils.SimulationCfg(dt=float(metadata.get("dt", 0.005)), device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, -2.5, 2.0], [0.0, 0.0, 0.5])

    scene_cfg = MpdTrajectorySceneCfg(num_envs=num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    contact_forces = scene["contact_forces"]
    sim_dt = sim.get_physics_dt()

    joint_ids, finger_joint_ids = _resolve_panda_joint_ids(robot, dof)
    device = torch.device(sim.device)
    q_trajs_pos = q_trajs_pos_cpu.to(device=device)
    q_pos_starts = q_pos_starts_cpu.to(device=device)

    _reset_robot(robot, scene, q_pos_starts, joint_ids, finger_joint_ids)
    contact_forces.reset()
    scene.write_data_to_sim()
    sim.step(render=not args_cli.headless)
    scene.update(sim_dt)

    collision_mask = torch.zeros(batch, dtype=torch.bool, device=device)
    first_collision_step = torch.full((batch,), -1, dtype=torch.long, device=device)

    for step_idx in range(horizon):
        q_step = q_trajs_pos[step_idx]
        if args_cli.stop_robot_if_in_contact and collision_mask.any():
            q_step = q_step.clone()
            if isinstance(joint_ids, slice):
                q_step[collision_mask] = robot.data.joint_pos[:batch, joint_ids][collision_mask]
            else:
                q_step[collision_mask] = robot.data.joint_pos[:batch, joint_ids][collision_mask]

        _apply_joint_targets(robot, q_step, joint_ids, finger_joint_ids)
        scene.write_data_to_sim()

        for _ in range(args_cli.action_repeat):
            sim.step(render=not args_cli.headless)
            scene.update(sim_dt)
            step_contact_mask = _contact_mask(contact_forces, args_cli.contact_force_threshold, batch)
            newly_collided = step_contact_mask & ~collision_mask
            first_collision_step[newly_collided] = step_idx
            collision_mask |= step_contact_mask

    n_collision = int(collision_mask.sum().item())
    n_free = int(batch - n_collision)
    stats = {
        "backend": "isaaclab",
        "robot_name": str(metadata.get("robot_name", args_cli.robot_name)),
        "env_name": str(metadata.get("env_name", "")),
        "num_envs": int(num_envs),
        "n_trajectories": int(batch),
        "trajectory_horizon": int(horizon),
        "trajectory_dof": int(dof),
        "n_trajectories_collision": n_collision,
        "n_trajectories_free": n_free,
        "n_trajectories_free_fraction": float(n_free / batch) if batch else 0.0,
        "collision_mask": collision_mask.detach().cpu().tolist(),
        "first_collision_step": first_collision_step.detach().cpu().tolist(),
        "contact_force_threshold": float(args_cli.contact_force_threshold),
        "action_repeat": int(args_cli.action_repeat),
        "device": str(args_cli.device),
        "video_path": args_cli.video_path.as_posix() if args_cli.make_video and args_cli.video_path else None,
        "video_note": "video capture is reserved for a later migration stage" if args_cli.make_video else None,
    }

    joint_names = getattr(robot.data, "joint_names", None)
    if joint_names is not None:
        stats["joint_names"] = _to_jsonable(joint_names)

    return stats


def main() -> None:
    try:
        stats = run_evaluation()
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
