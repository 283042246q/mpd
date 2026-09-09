"""Pure artifact/asset gates plus lazy Isaac Lab configuration for Marvin."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_DIR = REPO_ROOT / "mpd/torch_robotics/torch_robotics/data/urdf/robots/marvin"
MARVIN_URDF = ROBOT_DIR / "marvin_pika_bimanual_mpd.urdf"
ASSET_LOCK = ROBOT_DIR / "pika_assets.lock.yaml"
CANONICAL_JOINT_NAMES = tuple(
    [f"Joint{index}_L" for index in range(1, 8)]
    + [f"Joint{index}_R" for index in range(1, 8)]
)
TCP_FRAME_NAMES = ("left_pika_gripper_tcp", "right_pika_gripper_tcp")
# The massless TCP links are USD Xforms after URDF conversion, not rigid bodies.
# Isaac therefore exposes the two gripper bases in ``body_names``; the TCP pose
# is reconstructed with these verified fixed transforms.
TCP_BODY_NAMES = ("left_gripper_base_link", "right_gripper_base_link")
TCP_OFFSETS_XYZ = ((0.0, 0.0, 0.21), (0.0, 0.0, 0.21))
TCP_OFFSETS_RPY = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash a USD bundle, including payloads referenced by the top-level USD."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"USD bundle directory not found: {root}")
    digest = hashlib.sha256()
    files = sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name != "marvin_bimanual_asset.json"
    )
    if not files:
        raise ValueError(f"USD bundle directory is empty: {root}")
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _parse_vector(text: str | None, *, length: int, label: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in (text or "").split())
    if len(values) != length or not np.isfinite(values).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return values


def validate_marvin_urdf(path: Path = MARVIN_URDF) -> dict:
    """Reject asset drift before Isaac Sim starts or commands any joints."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Marvin URDF not found: {path}")
    lock = yaml.safe_load(ASSET_LOCK.read_text())
    root = ET.parse(path).getroot()
    movable = tuple(
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("type") != "fixed"
    )
    links = {link.get("name") for link in root.findall("link")}
    if movable != CANONICAL_JOINT_NAMES:
        raise ValueError("Marvin URDF movable joints are not canonical left-then-right 14D")
    if not set(TCP_FRAME_NAMES) <= links or not set(TCP_BODY_NAMES) <= links:
        raise ValueError("Marvin URDF is missing one or both Pika TCP links")
    if tuple(lock.get("joint_names", ())) != CANONICAL_JOINT_NAMES:
        raise ValueError("pika_assets.lock.yaml joint order drift")
    if tuple(lock.get("ee_links", ())) != TCP_FRAME_NAMES:
        raise ValueError("pika_assets.lock.yaml TCP link drift")
    canonical_key = MARVIN_URDF.relative_to(REPO_ROOT).as_posix()
    expected_urdf_hash = lock.get("files", {}).get(canonical_key)
    actual_urdf_hash = sha256_file(path)
    if not expected_urdf_hash or actual_urdf_hash != expected_urdf_hash:
        raise ValueError(
            "Marvin URDF hash differs from pika_assets.lock.yaml: "
            f"expected={expected_urdf_hash}, actual={actual_urdf_hash}"
        )
    mesh_hashes = {}
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename or "://" in filename or Path(filename).is_absolute():
            raise ValueError(f"Marvin URDF mesh must be a local relative path: {filename!r}")
        mesh_path = (path.parent / filename).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Marvin mesh not found: {mesh_path}")
        try:
            mesh_key = mesh_path.relative_to(REPO_ROOT).as_posix()
        except ValueError as error:
            raise ValueError(f"Marvin mesh is outside the repository: {mesh_path}") from error
        expected_mesh_hash = lock.get("files", {}).get(mesh_key)
        actual_mesh_hash = sha256_file(mesh_path)
        if not expected_mesh_hash or actual_mesh_hash != expected_mesh_hash:
            raise ValueError(
                f"Marvin mesh hash drift for {mesh_key}: "
                f"expected={expected_mesh_hash}, actual={actual_mesh_hash}"
            )
        mesh_hashes[mesh_key] = actual_mesh_hash

    tcp_joints = {
        joint.get("name"): joint
        for joint in root.findall("joint")
        if joint.get("type") == "fixed"
    }
    for arm, frame, body, xyz, rpy in zip(
        ("left", "right"),
        TCP_FRAME_NAMES,
        TCP_BODY_NAMES,
        TCP_OFFSETS_XYZ,
        TCP_OFFSETS_RPY,
    ):
        joint = tcp_joints.get(f"{arm}_pika_gripper_tcp_joint")
        if joint is None:
            raise ValueError(f"Marvin URDF is missing the {arm} TCP fixed joint")
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        if parent is None or parent.get("link") != body:
            raise ValueError(f"{arm} TCP must be parented to {body}")
        if child is None or child.get("link") != frame:
            raise ValueError(f"{arm} TCP fixed joint must target {frame}")
        if origin is None:
            raise ValueError(f"{arm} TCP fixed joint has no origin")
        actual_xyz = _parse_vector(origin.get("xyz"), length=3, label=f"{arm} TCP xyz")
        actual_rpy = _parse_vector(origin.get("rpy"), length=3, label=f"{arm} TCP rpy")
        if not np.allclose(actual_xyz, xyz, atol=1e-12, rtol=0.0):
            raise ValueError(f"{arm} TCP offset drift: {actual_xyz} != {xyz}")
        if not np.allclose(actual_rpy, rpy, atol=1e-12, rtol=0.0):
            raise ValueError(f"{arm} TCP rotation drift: {actual_rpy} != {rpy}")
    limits = {}
    for joint in root.findall("joint"):
        if joint.get("name") not in CANONICAL_JOINT_NAMES:
            continue
        limit = joint.find("limit")
        limits[joint.get("name")] = (
            float(limit.get("lower")),
            float(limit.get("upper")),
        )
    return {
        "urdf_path": path,
        "urdf_sha256": actual_urdf_hash,
        "asset_manifest_sha256": lock["asset_sha256"],
        "joint_names": CANONICAL_JOINT_NAMES,
        "tcp_frame_names": TCP_FRAME_NAMES,
        "tcp_body_names": TCP_BODY_NAMES,
        "tcp_offsets_xyz": TCP_OFFSETS_XYZ,
        "tcp_offsets_rpy": TCP_OFFSETS_RPY,
        "joint_limits": limits,
        "mesh_hashes": mesh_hashes,
        "tcp_calibrated": bool(lock.get("tcp_calibrated", False)),
    }


def validate_scene_payload(scene: dict) -> dict:
    """Hard-gate the static Warehouse payload used by Isaac Lab."""
    if scene.get("schema") != "mpd_isaaclab_scene" or scene.get("schema_version") != 1:
        raise ValueError("scene.json must use mpd_isaaclab_scene schema version 1")
    if scene.get("frame_id") != "world":
        raise ValueError("scene.json frame_id must be 'world'")
    if scene.get("env_name") != "EnvWarehouseMarvinBimanual":
        raise ValueError("scene.json must describe EnvWarehouseMarvinBimanual")
    unsupported = scene.get("unsupported_obstacles", [])
    if unsupported:
        raise ValueError(f"Warehouse scene has unsupported obstacles: {unsupported}")
    obstacles = scene.get("obstacles")
    if not isinstance(obstacles, list):
        raise ValueError("scene obstacles must be a list")
    names = set()
    counts = {"sphere": 0, "box": 0}
    for index, obstacle in enumerate(obstacles):
        if not isinstance(obstacle, dict):
            raise ValueError(f"scene obstacle {index} must be an object")
        name = obstacle.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"scene obstacle {index} has an invalid or duplicate name")
        names.add(name)
        kind = obstacle.get("type")
        if kind not in counts:
            raise ValueError(f"unsupported Warehouse primitive {kind!r}")
        counts[kind] += 1
        position = np.asarray(obstacle.get("position"), dtype=np.float64)
        orientation = np.asarray(obstacle.get("orientation"), dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"scene obstacle {name!r} position must be finite [3]")
        if orientation.shape != (4,) or not np.isfinite(orientation).all():
            raise ValueError(f"scene obstacle {name!r} orientation must be finite wxyz [4]")
        if not np.isclose(np.linalg.norm(orientation), 1.0, atol=1e-5):
            raise ValueError(f"scene obstacle {name!r} quaternion is not normalized")
        if kind == "box":
            size = np.asarray(obstacle.get("size"), dtype=np.float64)
            if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
                raise ValueError(f"scene box {name!r} size must be positive finite [3]")
        else:
            radius = obstacle.get("radius")
            if not isinstance(radius, (int, float)) or not np.isfinite(radius) or radius <= 0.0:
                raise ValueError(f"scene sphere {name!r} radius must be positive and finite")
    return {
        "scene_sha256": sha256_json(scene),
        "n_obstacles": len(obstacles),
        "n_spheres": counts["sphere"],
        "n_boxes": counts["box"],
        "frame_id": "world",
    }


@dataclass(frozen=True)
class MarvinInferenceArtifact:
    root: Path
    result: dict
    scene: dict
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    time_from_start: np.ndarray
    top_k_positions: np.ndarray
    top_k_velocities: np.ndarray
    top_k_accelerations: np.ndarray
    top_k_candidate_indices: np.ndarray
    ee_goal_pose: np.ndarray | None
    active_ee_mask: np.ndarray | None
    mpd_tcp_pose_start: np.ndarray | None
    mpd_top_k_tcp_pose_final: np.ndarray | None
    hashes: dict


def _resolve_artifact_paths(path: Path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        root, result_path = path, path / "result.json"
    elif path.name == "result.json" or path.suffix == ".json":
        root, result_path = path.parent, path
    elif path.suffix == ".npz":
        root, result_path = path.parent, path.parent / "result.json"
    else:
        raise ValueError("artifact must be a directory, result JSON, or trajectory NPZ")
    return root, result_path, root / "trajectory.npz", root / "scene.json"


def load_inference_artifact(path: Path) -> MarvinInferenceArtifact:
    """Load a Phase-1 artifact without importing Torch or Isaac Lab."""
    root, result_path, trajectory_path, scene_path = _resolve_artifact_paths(path)
    for required in (result_path, trajectory_path, scene_path):
        if not required.is_file():
            raise FileNotFoundError(f"Incomplete Marvin artifact; missing {required}")
    result = json.loads(result_path.read_text())
    scene = json.loads(scene_path.read_text())
    if result.get("schema") != "marvin_bimanual_result/v2":
        raise ValueError("artifact result schema must be marvin_bimanual_result/v2")
    if result.get("status") != "success":
        raise ValueError(f"artifact result status is {result.get('status')!r}")
    if tuple(result.get("joint_names", ())) != CANONICAL_JOINT_NAMES:
        raise ValueError("result joint order does not match canonical Marvin order")
    scene_gate = validate_scene_payload(scene)

    with np.load(trajectory_path, allow_pickle=False) as payload:
        required_arrays = (
            "positions",
            "velocities",
            "accelerations",
            "time_from_start",
            "joint_names",
        )
        missing = [key for key in required_arrays if key not in payload]
        if missing:
            raise ValueError(f"trajectory.npz is missing {missing}")
        positions = np.asarray(payload["positions"], dtype=np.float64)
        velocities = np.asarray(payload["velocities"], dtype=np.float64)
        accelerations = np.asarray(payload["accelerations"], dtype=np.float64)
        times = np.asarray(payload["time_from_start"], dtype=np.float64)
        names = tuple(str(item) for item in payload["joint_names"].tolist())
        top_positions = np.asarray(
            payload["top_k_positions"] if "top_k_positions" in payload else positions[None],
            dtype=np.float64,
        )
        top_velocities = np.asarray(
            payload["top_k_velocities"] if "top_k_velocities" in payload else velocities[None],
            dtype=np.float64,
        )
        top_accelerations = np.asarray(
            payload["top_k_accelerations"] if "top_k_accelerations" in payload else accelerations[None],
            dtype=np.float64,
        )
        top_indices = np.asarray(
            payload["top_k_candidate_indices"]
            if "top_k_candidate_indices" in payload
            else [0],
            dtype=np.int64,
        )
        ee_goal_pose = (
            np.asarray(payload["ee_goal_pose"], dtype=np.float64)
            if "ee_goal_pose" in payload
            else None
        )
        active_ee_mask = (
            np.asarray(payload["active_ee_mask"], dtype=np.float64)
            if "active_ee_mask" in payload
            else None
        )
        mpd_tcp_pose_start = (
            np.asarray(payload["mpd_tcp_pose_start"], dtype=np.float64)
            if "mpd_tcp_pose_start" in payload
            else None
        )
        mpd_top_k_tcp_pose_final = (
            np.asarray(payload["mpd_top_k_tcp_pose_final"], dtype=np.float64)
            if "mpd_top_k_tcp_pose_final" in payload
            else None
        )

    horizon = positions.shape[0] if positions.ndim == 2 else -1
    if positions.shape != (horizon, 14) or horizon < 2:
        raise ValueError("positions must have shape [T,14], T>=2")
    if velocities.shape != positions.shape or accelerations.shape != positions.shape:
        raise ValueError("velocity/acceleration arrays must match positions")
    if times.shape != (horizon,) or times[0] != 0.0 or not np.all(np.diff(times) > 0):
        raise ValueError("time_from_start must start at zero and increase strictly")
    if names != CANONICAL_JOINT_NAMES:
        raise ValueError("trajectory.npz joint order is not canonical")
    expected_top_shape = (top_positions.shape[0], horizon, 14)
    if top_positions.shape != expected_top_shape or top_positions.shape[0] < 1:
        raise ValueError("top_k_positions must have shape [K,T,14]")
    if top_velocities.shape != expected_top_shape or top_accelerations.shape != expected_top_shape:
        raise ValueError("Top-K derivative arrays must match top_k_positions")
    if top_indices.shape != (top_positions.shape[0],):
        raise ValueError("top_k_candidate_indices must have shape [K]")
    if not all(
        np.isfinite(value).all()
        for value in (
            positions,
            velocities,
            accelerations,
            times,
            top_positions,
            top_velocities,
            top_accelerations,
        )
    ):
        raise ValueError("trajectory artifact contains NaN or Inf")
    if not np.allclose(top_positions[0], positions, atol=1e-8, rtol=0.0):
        raise ValueError("Top-K slot zero must be the selected best trajectory")
    if not np.allclose(np.asarray(result["positions"]), positions, atol=1e-8, rtol=0.0):
        raise ValueError("result.json positions differ from trajectory.npz")
    if ee_goal_pose is not None and ee_goal_pose.shape != (2, 3, 4):
        raise ValueError("ee_goal_pose must have shape [2,3,4]")
    if active_ee_mask is not None and active_ee_mask.shape != (2,):
        raise ValueError("active_ee_mask must have shape [2]")
    if mpd_tcp_pose_start is not None and mpd_tcp_pose_start.shape != (2, 3, 4):
        raise ValueError("mpd_tcp_pose_start must have shape [2,3,4]")
    if (
        mpd_top_k_tcp_pose_final is not None
        and mpd_top_k_tcp_pose_final.shape != (top_positions.shape[0], 2, 3, 4)
    ):
        raise ValueError("mpd_top_k_tcp_pose_final must have shape [K,2,3,4]")
    optional_arrays = (
        ee_goal_pose,
        active_ee_mask,
        mpd_tcp_pose_start,
        mpd_top_k_tcp_pose_final,
    )
    if any(value is not None and not np.isfinite(value).all() for value in optional_arrays):
        raise ValueError("trajectory pose metadata contains NaN or Inf")
    is_stub = result.get("validation", {}).get("backend") == "contract_stub"
    if not is_stub and (mpd_tcp_pose_start is None or mpd_top_k_tcp_pose_final is None):
        raise ValueError("real MPD artifacts must include MPD start/final TCP FK poses")
    result_scene_hash = result.get("scene", {}).get("scene_sha256")
    if result_scene_hash is not None and result_scene_hash != scene_gate["scene_sha256"]:
        raise ValueError("result.json scene hash differs from canonical scene.json content")

    return MarvinInferenceArtifact(
        root=root,
        result=result,
        scene=scene,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        time_from_start=times,
        top_k_positions=top_positions,
        top_k_velocities=top_velocities,
        top_k_accelerations=top_accelerations,
        top_k_candidate_indices=top_indices,
        ee_goal_pose=ee_goal_pose,
        active_ee_mask=active_ee_mask,
        mpd_tcp_pose_start=mpd_tcp_pose_start,
        mpd_top_k_tcp_pose_final=mpd_top_k_tcp_pose_final,
        hashes={
            "result_sha256": sha256_file(result_path),
            "trajectory_sha256": sha256_file(trajectory_path),
            "scene_file_sha256": sha256_file(scene_path),
            "scene_sha256": scene_gate["scene_sha256"],
        },
    )


def convert_marvin_urdf_to_usd(
    output_dir: Path,
    *,
    urdf_path: Path = MARVIN_URDF,
    force: bool = False,
) -> dict:
    """Convert after SimulationApp launch and return a content-addressed gate."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    asset = validate_marvin_urdf(urdf_path)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = UrdfConverterCfg(
        asset_path=asset["urdf_path"].as_posix(),
        usd_dir=output_dir.as_posix(),
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        force_usd_conversion=bool(force),
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=400.0, damping=40.0
            ),
            target_type="position",
        ),
    )
    converter = UrdfConverter(config)
    usd_path = Path(converter.usd_path).resolve()
    if not usd_path.is_file():
        raise RuntimeError(f"Isaac Lab converter did not create {usd_path}")
    metadata = {
        "schema": "marvin_bimanual_isaaclab_asset/v1",
        **asset,
        "usd_path": usd_path,
        "usd_sha256": sha256_file(usd_path),
        "usd_bundle_sha256": sha256_tree(usd_path.parent),
        "converter": "IsaacLab UrdfConverter",
        "fix_base": True,
        "merge_fixed_joints": False,
    }
    serializable = {
        key: value.as_posix() if isinstance(value, Path) else value
        for key, value in metadata.items()
    }
    (output_dir / "marvin_bimanual_asset.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def validate_marvin_usd(path: Path) -> dict:
    """Validate a previously converted local bundle against its conversion record."""
    asset = validate_marvin_urdf()
    usd_path = Path(path).expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"Marvin USD not found: {usd_path}")
    metadata_candidates = (
        usd_path.parent / "marvin_bimanual_asset.json",
        usd_path.parent.parent / "marvin_bimanual_asset.json",
    )
    metadata_path = next((item for item in metadata_candidates if item.is_file()), None)
    if metadata_path is None:
        raise FileNotFoundError(
            "Preconverted Marvin USD needs marvin_bimanual_asset.json beside its bundle"
        )
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != "marvin_bimanual_isaaclab_asset/v1":
        raise ValueError("Preconverted Marvin USD metadata schema is invalid")
    actual_usd_hash = sha256_file(usd_path)
    actual_bundle_hash = sha256_tree(usd_path.parent)
    expected = {
        "urdf_sha256": asset["urdf_sha256"],
        "asset_manifest_sha256": asset["asset_manifest_sha256"],
        "usd_sha256": actual_usd_hash,
        "usd_bundle_sha256": actual_bundle_hash,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Preconverted Marvin USD {key} mismatch: "
                f"metadata={metadata.get(key)!r}, actual={value!r}"
            )
    return {
        **asset,
        "schema": metadata["schema"],
        "usd_path": usd_path,
        "usd_sha256": actual_usd_hash,
        "usd_bundle_sha256": actual_bundle_hash,
        "converter": metadata.get("converter", "preconverted"),
        "metadata_path": metadata_path,
    }


def build_marvin_articulation_cfg(usd_path: Path):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    import isaaclab.sim as sim_utils

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=Path(usd_path).as_posix(),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=["Joint[1-7]_[LR]"],
                stiffness=400.0,
                damping=40.0,
            )
        },
    )


def resolve_canonical_joint_ids(robot) -> list[int]:
    names = tuple(robot.joint_names)
    missing = [name for name in CANONICAL_JOINT_NAMES if name not in names]
    unexpected = [name for name in names if name not in CANONICAL_JOINT_NAMES]
    if missing or unexpected or len(names) != 14:
        raise ValueError(
            f"Isaac articulation joint mismatch; missing={missing}, unexpected={unexpected}"
        )
    return [names.index(name) for name in CANONICAL_JOINT_NAMES]


def resolve_tcp_body_ids(robot) -> tuple[int, int]:
    names = tuple(robot.body_names)
    missing = [name for name in TCP_BODY_NAMES if name not in names]
    if missing:
        raise ValueError(f"Isaac articulation is missing TCP bodies: {missing}")
    return tuple(names.index(name) for name in TCP_BODY_NAMES)


def tcp_poses_from_body_state(body_positions, body_quaternions_xyzw):
    """Compose the two fixed TCP offsets with Isaac rigid-body poses."""
    positions = np.asarray(body_positions, dtype=np.float64)
    quaternions = np.asarray(body_quaternions_xyzw, dtype=np.float64)
    if positions.shape[-2:] != (2, 3) or quaternions.shape[-2:] != (2, 4):
        raise ValueError("body positions/quaternions must end in [2,3] and [2,4]")
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(quaternions, -1, 0)
    rotations = np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*quaternions.shape[:-1], 3, 3)
    offsets = np.asarray(TCP_OFFSETS_XYZ, dtype=np.float64)
    tcp_positions = positions + np.einsum("...aij,aj->...ai", rotations, offsets)
    return tcp_positions, rotations


def classify_contact_forces(body_names, net_forces, threshold: float) -> dict:
    """Classify arm contacts; balanced simultaneous arm forces indicate inter-arm contact."""
    forces = np.asarray(net_forces, dtype=np.float64)
    if forces.ndim != 3 or forces.shape[1] != len(body_names) or forces.shape[2] != 3:
        raise ValueError("net_forces must have shape [N,B,3]")
    sides = []
    for name in body_names:
        if name.startswith("left_") or "_L" in name:
            sides.append("left")
        elif name.startswith("right_") or "_R" in name:
            sides.append("right")
        else:
            sides.append("base")
    norm = np.linalg.norm(forces, axis=-1)
    masks = {side: np.asarray([value == side for value in sides]) for side in ("left", "right", "base")}
    active = {
        side: (norm[:, mask].max(axis=1) > threshold if mask.any() else np.zeros(forces.shape[0], dtype=bool))
        for side, mask in masks.items()
    }
    left_sum = forces[:, masks["left"]].sum(axis=1)
    right_sum = forces[:, masks["right"]].sum(axis=1)
    balance = np.linalg.norm(left_sum + right_sum, axis=-1) / np.maximum(
        np.linalg.norm(left_sum, axis=-1) + np.linalg.norm(right_sum, axis=-1),
        1e-12,
    )
    interarm = active["left"] & active["right"] & (balance < 0.25)
    return {
        "left_contact": active["left"],
        "right_contact": active["right"],
        "base_contact": active["base"],
        "interarm_contact": interarm,
        "left_world_contact": active["left"] & ~interarm,
        "right_world_contact": active["right"] & ~interarm,
        "interarm_balance_ratio": balance,
        "classification_method": "body-side net force with balanced-pair interarm test",
    }


def spawn_scene_obstacles(sim_utils, scene_payload: dict) -> dict:
    obstacles = scene_payload.get("obstacles") or []
    for index, obstacle in enumerate(obstacles):
        common = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            "visual_material": sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.47, 0.50)
            ),
        }
        if obstacle["type"] == "sphere":
            config = sim_utils.SphereCfg(radius=float(obstacle["radius"]), **common)
        elif obstacle["type"] == "box":
            config = sim_utils.CuboidCfg(
                size=tuple(float(value) for value in obstacle["size"]), **common
            )
        else:
            raise ValueError(f"Unsupported Warehouse primitive {obstacle['type']!r}")
        config.func(
            f"/World/envs/env_.*/MpdObstacle_{index:03d}",
            config,
            translation=tuple(float(value) for value in obstacle["position"]),
            orientation=tuple(float(value) for value in obstacle["orientation"]),
        )
    return {"n_obstacles": len(obstacles), "unsupported": scene_payload.get("unsupported_obstacles", [])}
