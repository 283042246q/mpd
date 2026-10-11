"""IsaacLab kinematic video replay of frozen Warehouse static/dynamic worlds.

Launch through isaaclab.sh -p; MPD planning packages are not imported here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--video_fps", type=float, default=24.)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--robot_usd")
    parser.add_argument("--camera_eye", type=float, nargs=3, default=[-2.2, -2.2, 2.15])
    parser.add_argument("--camera_target", type=float, nargs=3, default=[.35, 0., .45])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.video_fps, args.width, args.height) <= 0 or (args.limit is not None and args.limit < 1):
        parser.error("resolution, fps and limit must be positive")
    args.enable_cameras = True
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch
from pxr import Gf, UsdGeom
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from isaacsim.core.experimental.utils.app import enable_extension

from warehouse_spacetime_payload import frame_times, interpolate, load_case, validate_suite

enable_extension("isaacsim.util.debug_draw")
from isaacsim.util.debug_draw import _debug_draw


COLORS = [(1., .24, .12, 1.), (.04, .72, .9, 1.), (1., .72, .08, 1.)]
SHAPE_NAMES = ["sphere", "box", "capsule"]
PANDA_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
PANDA_CFG.spawn.usd_path = args_cli.robot_usd or f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
PANDA_CFG.spawn.rigid_props.disable_gravity = True
PANDA_CFG.spawn.articulation_props.enabled_self_collisions = False


@configclass
class WarehouseReplaySceneCfg(InteractiveSceneCfg):
    dome = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1200., color=(.85, .88, .95)))
    robot = PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera", update_period=0.,
                       height=args_cli.height, width=args_cli.width, data_types=["rgb"],
                       spawn=sim_utils.PinholeCameraCfg(focal_length=18., horizontal_aperture=20.955,
                                                       clipping_range=(.05, 100.)))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".inprogress")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def spawn_static(payload):
    if payload.get("unsupported_obstacles"):
        raise ValueError("static scene has unsupported geometry")
    for index, obj in enumerate(payload["obstacles"]):
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.18, .22, .28), roughness=.8)
        if obj["type"] == "box":
            config = sim_utils.CuboidCfg(size=tuple(obj["size"]), visual_material=material)
        elif obj["type"] == "sphere":
            config = sim_utils.SphereCfg(radius=obj["radius"], visual_material=material)
        else:
            raise ValueError("unsupported static shape")
        # Visual-only primitives: no rigid body/collider alters prescribed motion.
        config.func("/World/Warehouse/Object_%03d" % index, config,
                    translation=tuple(obj["position"]), orientation=tuple(obj["orientation"]))


class DynamicObjects:
    """Exact sphere, box and z-capsule dimensions [m], animated without physics."""

    def __init__(self, stage, count):
        self.stage, self.count = stage, count
        self.transforms, self.geometries = [], []
        for obj in range(count):
            parent = UsdGeom.Xform.Define(stage, "/World/Moving/Object_%02d" % obj)
            self.transforms.append(parent.AddTranslateOp())
            self.geometries.append(None)

    def configure(self, shapes, sizes):
        if len(shapes) != self.count:
            raise ValueError("a render suite must share the dynamic object count")
        for obj, (shape, size) in enumerate(zip(shapes, sizes)):
            path = "/World/Moving/Object_%02d/Geometry" % obj
            self.stage.RemovePrim(path)
            if shape == 0:
                geometry = UsdGeom.Sphere.Define(self.stage, path)
                geometry.GetRadiusAttr().Set(float(size[0]))
            elif shape == 1:
                geometry = UsdGeom.Cube.Define(self.stage, path)
                geometry.GetSizeAttr().Set(2.)
                geometry.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
            else:
                geometry = UsdGeom.Capsule.Define(self.stage, path)
                geometry.GetRadiusAttr().Set(float(size[0]))
                # USD capsule height is the cylindrical segment, excluding caps.
                geometry.GetHeightAttr().Set(2. * float(size[1]))
                geometry.GetAxisAttr().Set("Z")
            geometry.CreateDisplayColorAttr([Gf.Vec3f(*COLORS[obj % 3][:3])])
            self.geometries[obj] = geometry

    def update(self, centers):
        for transform, center in zip(self.transforms, centers):
            transform.Set(Gf.Vec3d(*map(float, center)))


def draw_paths(draw, case, instant):
    draw.clear_lines()
    starts, ends, colors, widths = [], [], [], []
    def add(path, color, width):
        starts.extend(path[:-1].tolist())
        ends.extend(path[1:].tolist())
        colors.extend([color] * (len(path) - 1))
        widths.extend([width] * (len(path) - 1))
    add(case["ee_path"], (.2, .5, 1., .75), 2.)
    times = case["forecast_time"]
    # Recent and upcoming motion, rather than a huge full-horizon line offscreen.
    trail = (times >= max(0., instant - 1.2)) & (times <= min(times[-1], instant + .8))
    for obj in range(len(case["shape"])):
        color = COLORS[obj % 3]
        add(case["obstacle_positions"][trail, obj], color[:3] + (.65,), 2.)
    if starts:
        draw.draw_lines(starts, ends, colors, widths)


def hud(rgb, case, instant):
    frame = np.ascontiguousarray(rgb.copy())
    metadata = case["metadata"]
    scale = args_cli.width / 1280.
    panel_height = int(112 * scale)
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (args_cli.width, panel_height), (15, 20, 27), -1)
    frame = cv2.addWeighted(shade, .72, frame, .28, 0)
    def text(value, row, color=(245, 245, 245)):
        cv2.putText(frame, value, (int(18*scale), int(row*scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    .64*scale, color, max(1, int(2*scale)), cv2.LINE_AA)
    text("WAREHOUSE  |  %s  |  variant %d  |  t %.2f / %.2f s" %
         (metadata["mode"], metadata["variant"], instant, case["execution_time"][-1]), 29)
    text("Panda joint trajectory + frozen moving obstacles  |  %s" % metadata["backend"], 56)
    text("Static/object overlaps ignored  |  Kinematic replay, no physics execution claim", 83)
    y = args_cli.height - int(70*scale)
    for obj, shape in enumerate(case["shape"]):
        label = "%s: workspace event %.2fs" % (SHAPE_NAMES[int(shape)], case["event_times"][obj])
        color = tuple(int(255*x) for x in COLORS[obj % 3][:3])
        cv2.putText(frame, label, (int(18*scale), y + int(22*obj*scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    .5*scale, color, max(1, int(scale)), cv2.LINE_AA)
    return frame


def render_suite():
    suite = json.loads(args_cli.suite.read_text())
    validate_suite(suite)
    all_entries = suite["cases"]
    entries = all_entries[:args_cli.limit]
    if not entries:
        raise ValueError("empty suite")
    output = args_cli.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    static_path = Path(suite["static_scene"])
    payload = json.loads(static_path.read_text())
    static_hash = hashlib.sha256(static_path.read_bytes()).hexdigest()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1. / 240., device=args_cli.device))
    scene = InteractiveScene(WarehouseReplaySceneCfg(num_envs=1, env_spacing=2.5))
    spawn_static(payload)
    first = load_case(entries[0]["payload"])
    objects = DynamicObjects(omni.usd.get_context().get_stage(), len(first["shape"]))
    objects.configure(first["shape"], first["size"])
    objects.update(first["obstacle_positions"][0])
    draw = _debug_draw.acquire_debug_draw_interface()
    sim.reset()
    robot, camera = scene["robot"], scene["camera"]
    arm_ids = robot.find_joints(["panda_joint%d" % j for j in range(1, 8)], preserve_order=True)[0]
    finger_ids = robot.find_joints(["panda_finger_joint.*"])[0]
    hand_id = robot.find_bodies("panda_hand")[0][0]
    if len(arm_ids) != 7:
        raise ValueError("Panda must have seven ordered arm joints")
    camera.set_world_poses_from_view(torch.tensor([args_cli.camera_eye], device=sim.device),
                                     torch.tensor([args_cli.camera_target], device=sim.device))
    summaries = []
    for entry in entries:
        case = load_case(entry["payload"])
        case_id = entry["case_id"]
        video, stats = output / (case_id + ".mp4"), output / (case_id + ".json")
        identity = {"payload_sha256": hashlib.sha256(Path(entry["payload"]).read_bytes()).hexdigest(),
                    "static_sha256": static_hash, "width": args_cli.width, "height": args_cli.height,
                    "fps": args_cli.video_fps, "robot_usd": PANDA_CFG.spawn.usd_path,
                    "camera_eye": args_cli.camera_eye, "camera_target": args_cli.camera_target,
                    "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "payload_helper_sha256": hashlib.sha256(Path(__file__).with_name("warehouse_spacetime_payload.py").read_bytes()).hexdigest()}
        if args_cli.resume and stats.exists() and video.exists():
            saved = json.loads(stats.read_text())
            if saved.get("identity") == identity and saved.get("complete") and video.stat().st_size == saved["bytes"]:
                summaries.append(saved)
                print("[warehouse-render] reuse " + case_id, flush=True)
                continue
        if video.exists() and not args_cli.resume:
            raise FileExistsError("video exists; use --resume: " + str(video))
        objects.configure(case["shape"], case["size"])
        times = frame_times(float(case["execution_time"][-1]), args_cli.video_fps)
        temporary = video.with_name(video.stem + ".inprogress.mp4")
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), args_cli.video_fps,
                                 (args_cli.width, args_cli.height))
        if not writer.isOpened():
            raise RuntimeError("video writer failed")
        preview = None
        ee_error_max = 0.
        event_center = float(np.median(case["event_times"]))
        try:
            for frame_index, instant in enumerate(times):
                joints = robot.data.default_joint_pos.clone()
                joints[:, arm_ids] = torch.as_tensor(interpolate(case["execution_time"], case["q"], instant),
                                                     device=sim.device, dtype=joints.dtype)
                velocities = torch.zeros_like(robot.data.default_joint_vel)
                if finger_ids:
                    joints[:, finger_ids] = .04
                robot.write_joint_position_to_sim_index(position=joints)
                robot.write_joint_velocity_to_sim_index(velocity=velocities)
                robot.set_joint_position_target_index(target=joints)
                objects.update(interpolate(case["forecast_time"], case["obstacle_positions"], instant))
                draw_paths(draw, case, instant)
                # Flush PhysX/Fabric using one zero-velocity step, then impose
                # the exact recorded pose again. Clip time is the frozen time,
                # independent of simulator wall-clock or controller tracking.
                scene.write_data_to_sim()
                sim.step(render=False)
                robot.write_joint_position_to_sim_index(position=joints)
                robot.write_joint_velocity_to_sim_index(velocity=velocities)
                sim.forward()
                scene.update(sim.get_physics_dt())
                body_positions = robot.data.body_link_pos_w
                if hasattr(body_positions, "torch"):
                    body_positions = body_positions.torch
                expected_ee = interpolate(case["execution_time"], case["ee_positions"], instant)
                error = float(np.linalg.norm(body_positions[0, hand_id].detach().cpu().numpy() - expected_ee))
                ee_error_max = max(ee_error_max, error)
                if error > .01:
                    raise RuntimeError("rendered Panda FK differs from MPD by %.4f m" % error)
                # Warm up RTX after a new scene/geometry, without advancing q-t.
                for _ in range(8 if frame_index == 0 else 1):
                    sim.render()
                camera.update(1. / args_cli.video_fps, force_recompute=True)
                rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                frame = hud(rgb, case, float(instant))
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if preview is None or abs(instant - event_center) < abs(preview[0] - event_center):
                    preview = (float(instant), frame.copy())
                if frame_index and frame_index % int(max(1, args_cli.video_fps * 2)) == 0:
                    print("[warehouse-render] %s %d/%d" % (case_id, frame_index + 1, len(times)), flush=True)
        finally:
            writer.release()
        capture = cv2.VideoCapture(str(temporary))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if frames != len(times) or temporary.stat().st_size < 1000:
            raise RuntimeError("incomplete encoded video: " + case_id)
        os.replace(temporary, video)
        screenshot = output / (case_id + ".png")
        if not cv2.imwrite(str(screenshot), cv2.cvtColor(preview[1], cv2.COLOR_RGB2BGR)):
            raise RuntimeError("preview write failed")
        summary = {"case_id": case_id, "mode": entry["mode"], "variant": entry["variant"], "complete": True,
                   "video": str(video), "preview": str(screenshot), "frames": frames, "identity": identity,
                   "duration_s": float(times[-1]), "encoded_duration_s": frames / args_cli.video_fps,
                   "preview_time_s": preview[0], "bytes": video.stat().st_size,
                   "seed": entry["seed"], "source_row": entry["source_row"], "backend": case["metadata"]["backend"],
                   "static_objects": len(payload["obstacles"]), "dynamic_objects": len(case["shape"]),
                   "quality": case["metadata"]["quality"], "kinematic_replay": True,
                   "rendered_hand_error_max_m": ee_error_max,
                   "obstacle_static_overlaps": "ignored", "physics_execution_validated": False}
        atomic_json(stats, summary)
        summaries.append(summary)
        publish_report(output, suite, summaries)
        print("[warehouse-render] complete " + case_id, flush=True)
    draw.clear_lines()
    publish_report(output, suite, summaries)


def publish_report(output, suite, summaries):
    rows = []
    for case in suite["cases"]:
        path = output / (case["case_id"] + ".json")
        if path.exists():
            rows.append(json.loads(path.read_text()))
    counts = {mode: sum(r["mode"] == mode and r["complete"] for r in rows) for mode in suite["modes"]}
    atomic_json(output / "report.json", {"expected": suite["expected_cases"], "completed": sum(counts.values()),
                                         "mode_counts": counts, "videos": rows})
    tiles = []
    markdown = ["# Warehouse spacetime IsaacLab videos", "", "Kinematic replay of validated trajectories; obstacle/static overlaps ignored.", ""]
    for mode in suite["modes"]:
        mode_rows = sorted([r for r in rows if r["mode"] == mode], key=lambda r: r["variant"])
        if not mode_rows:
            continue
        markdown.extend(["## " + mode, ""])
        images = []
        for row in mode_rows:
            name = row["case_id"]
            markdown.extend(["- [%s video](%s.mp4) · [preview](%s.png) · %.2f s" % (name, name, name, row["duration_s"])])
            preview = cv2.imread(row["preview"])
            if preview is not None:
                images.append(cv2.resize(preview, (384, 216)))
        markdown.append("")
        while len(images) < suite["variants_per_mode"]:
            images.append(np.zeros((216, 384, 3), dtype=np.uint8))
        tiles.append(np.concatenate(images, axis=1))
    temporary = output / "INDEX.md.inprogress"
    temporary.write_text("\n".join(markdown))
    os.replace(temporary, output / "INDEX.md")
    if tiles:
        cv2.imwrite(str(output / "contact-sheet.jpg"), np.concatenate(tiles, axis=0))


if __name__ == "__main__":
    exit_code = 0
    try:
        render_suite()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Avoid native teardown crashes after the video writer and reports commit.
    os._exit(exit_code)
