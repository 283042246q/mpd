"""Offline asset/kinematics gate; optional MARVIN_ROS_ROOT enables live drift check.

No ROS node, hardware interface or external network is used by the offline tests.
"""
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from scripts.robots.build_marvin_pika_assets import freeze_fingers, stl_vertices


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "mpd/torch_robotics/torch_robotics/data"
MODEL = DATA / "urdf/robots/marvin"
CONFIG = DATA / "configs/marvin/pika"
JOINTS = [f"Joint{i}_{side}" for side in ("L", "R") for i in range(1, 8)]
TCP = [f"{side}_pika_gripper_tcp" for side in ("left", "right")]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_yaml(path):
    return yaml.safe_load(path.read_text())


@pytest.fixture(scope="module")
def manifest():
    return read_yaml(MODEL / "pika_assets.lock.yaml")


@pytest.fixture(scope="module")
def planning():
    return ET.parse(MODEL / "marvin_pika_bimanual_mpd.urdf").getroot()


@pytest.fixture(scope="module")
def reference():
    return ET.parse(MODEL / "sources/ros_marvin_pika_expanded.urdf").getroot()


def assert_file_hash(path, expected):
    assert path.is_file(), f"Missing asset: {path}"
    assert digest(path.read_bytes()) == expected, f"Asset drift: {path}"


def test_manifest_and_source_integrity(manifest, planning):
    assert manifest["schema"] == "marvin_pika_assets/v1"
    assert manifest["joint_names"] == JOINTS
    assert manifest["ee_links"] == TCP
    assert manifest["tcp_calibrated"] is False  # Upstream still labels TCP a placeholder.
    files = manifest["files"]
    assert digest(json.dumps(files, sort_keys=True).encode()) == manifest["asset_sha256"]
    for relative, expected in files.items():
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        assert_file_hash(ROOT / relative, expected)
    for name, source in manifest["packages"].items():
        package = MODEL / "sources" / name
        assert len(source["commit"]) == 40 and source["repository"].startswith("https://")
        for relative, expected in source["source_sha256"].items():
            assert_file_hash(package / relative, expected)
        declared = [e.text for e in ET.parse(package / "package.xml").findall("license")]
        assert declared == source["declared_license"]
    mesh_paths = {mesh.get("filename") for mesh in planning.findall(".//mesh")}
    assert mesh_paths == set(manifest["mesh_sources"])
    for relative, source in manifest["mesh_sources"].items():
        assert_file_hash(MODEL / relative, source["sha256"])
    for name in ("Apache-2.0.txt", "pika_ros-BSD-3-Clause.txt", "SOURCES.md"):
        path = MODEL / "licenses" / name
        assert path.relative_to(ROOT).as_posix() in files
    assert "Tixiao Shan" in (MODEL / "licenses/pika_ros-BSD-3-Clause.txt").read_text()
    assert (MODEL / "sources/marvin_description/LICENSE").is_file()
    assert (MODEL / "sources/pika_gripper_description/docs/MESH_SOURCES.md").is_file()


@pytest.mark.parametrize("kind", ["mesh", "config", "license"])
def test_integrity_gate_rejects_modified_assets(tmp_path, manifest, kind):
    candidates = {"mesh": "meshes/", "config": "configs/marvin/pika/", "license": "licenses/"}
    relative = next(p for p in manifest["files"] if candidates[kind] in p)
    # Only mutate a disposable copy, never the actual ROS or MPD assets.
    copy = tmp_path / "asset"
    copy.write_bytes((ROOT / relative).read_bytes() + b"\nchanged")
    with pytest.raises(AssertionError, match="Asset drift"):
        assert_file_hash(copy, manifest["files"][relative])


def test_check_mode_fails_without_rewriting_stale_output(tmp_path, monkeypatch):
    from scripts.robots import build_marvin_pika_assets as exporter
    relative = Path("generated.urdf")
    target = tmp_path / relative
    target.write_bytes(b"stale")
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    monkeypatch.setattr(exporter, "build", lambda *_: {relative: b"fresh"})
    monkeypatch.setattr("sys.argv", ["build", "--ros-root", str(tmp_path), "--check"])
    with pytest.raises(SystemExit, match="ROS/MPD asset drift"):
        exporter.main()
    assert target.read_bytes() == b"stale"


def test_planning_urdf_is_self_contained_fixed_gripper_tree(planning, manifest):
    links = [link.get("name") for link in planning.findall("link")]
    joints = planning.findall("joint")
    assert len(links) == len(set(links))
    assert len(joints) == len({j.get("name") for j in joints})
    assert [j.get("name") for j in joints if j.get("type") != "fixed"] == JOINTS
    parents = {j.find("child").get("link"): j.find("parent").get("link") for j in joints}
    assert len(parents) == len(joints) == len(links) - 1
    assert set(links) - parents.keys() == {"world"}
    for link in links:
        seen = set()
        while link in parents:
            assert link not in seen
            seen.add(link)
            link = parents[link]
        assert link == "world"
    assert not planning.findall("ros2_control")
    assert set(TCP) <= set(links)
    assert len(manifest["frozen_fingers"]) == 4
    for entry in manifest["frozen_fingers"].values():
        joint = planning.find(f"joint[@name='{entry['joint']}']")
        assert joint.get("type") == "fixed"
        assert all(joint.find(tag) is None for tag in ("axis", "mimic", "limit"))
        assert entry["q"] == pytest.approx(0.045)
    for mesh in planning.findall(".//mesh"):
        filename = mesh.get("filename")
        assert "://" not in filename and not Path(filename).is_absolute()
        assert ".." not in Path(filename).parts and (MODEL / filename).is_file()


def test_arm_geometry_and_limits_preserved(planning, reference):
    legacy = ET.parse(MODEL / "marvin_bimanual_mpd.urdf").getroot()
    for joint in legacy.findall("joint"):
        name = joint.get("name")
        actual = planning.find(f"joint[@name='{name}']")
        assert ET.tostring(actual).strip() == ET.tostring(joint).strip()
    for name in JOINTS:
        actual = planning.find(f"joint[@name='{name}']")
        original = reference.find(f"joint[@name='{name}']")
        assert ET.tostring(actual).strip() == ET.tostring(original).strip()
    assert read_yaml(CONFIG / "joint_limits.yaml") == read_yaml(CONFIG.parent / "joint_limits.yaml")
    old_spheres = read_yaml(CONFIG.parent / "collision_spheres.yaml")
    new_spheres = read_yaml(CONFIG / "collision_spheres.yaml")
    for name, entries in old_spheres.items():
        if name == "self_collision":
            for parent, peers in entries.items():
                assert new_spheres[name][parent] == peers
        else:
            assert new_spheres[name] == entries


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def origin_transform(origin):
    result = np.eye(4)
    if origin is None:
        return result
    roll, pitch, yaw = map(float, origin.get("rpy", "0 0 0").split())
    result[:3, :3] = (axis_rotation([0, 0, 1], yaw) @ axis_rotation([0, 1, 0], pitch)
                       @ axis_rotation([1, 0, 0], roll))
    result[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    return result


def independent_fk(root, values):
    """NumPy URDF evaluator, independent of TorchKin and finger-freezing code."""
    poses = {"world": np.eye(4)}
    remaining = list(root.findall("joint"))
    while remaining:
        progressed = False
        for joint in list(remaining):
            parent = joint.find("parent").get("link")
            if parent not in poses:
                continue
            relative = origin_transform(joint.find("origin"))
            if joint.get("type") != "fixed":
                q = values[joint.get("name")]
                mimic = joint.find("mimic")
                if mimic is not None:
                    q = values[mimic.get("joint")] * float(mimic.get("multiplier", 1))
                    q += float(mimic.get("offset", 0))
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                motion = np.eye(4)
                if joint.get("type") == "prismatic":
                    motion[:3, 3] = q * axis
                else:
                    motion[:3, :3] = axis_rotation(axis, q)
                relative = relative @ motion
            poses[joint.find("child").get("link")] = poses[parent] @ relative
            remaining.remove(joint)
            progressed = True
        assert progressed, "Disconnected/cyclic URDF"
    return poses


def test_frozen_link_frames_match_expanded_ros(planning, reference, manifest):
    qmap = dict(zip(JOINTS, np.linspace(-0.2, 0.3, 14)))
    qmap.update({e["joint"]: e["q"] for e in manifest["frozen_fingers"].values()})
    actual = independent_fk(planning, qmap)
    expected = independent_fk(reference, qmap)
    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=0)


def test_mesh_and_full_finger_travel_inside_collision_envelopes(planning, reference, manifest):
    envelopes = manifest["collision"]["link_envelopes"]
    assert len(envelopes) == 8
    spheres = read_yaml(CONFIG / "collision_spheres.yaml")
    fixed_pose = independent_fk(planning, dict.fromkeys(JOINTS, 0.0))
    for travel in (0.0, 0.0225, 0.045):
        values = dict.fromkeys(JOINTS, 0.0)
        values.update({e["joint"]: travel for e in manifest["frozen_fingers"].values()})
        source_pose = independent_fk(reference, values)
        for name, envelope in envelopes.items():
            low, high = np.array(envelope["lower"]), np.array(envelope["upper"])
            link = reference.find(f"link[@name='{name}']")
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                vertices = stl_vertices((MODEL / mesh.get("filename")).read_bytes())
                vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                transform = (np.linalg.inv(fixed_pose[name]) @ source_pose[name]
                             @ origin_transform(collision.find("origin")))
                local = vertices @ transform[:3, :3].T + transform[:3, 3]
                assert np.all(local >= low - 1e-9), name
                assert np.all(local <= high + 1e-9), name
            # Every grid cell's eight corners lie inside its sphere: the full
            # mesh volume and continuous finger sweep are covered, not only vertices.
            counts = np.maximum(1, np.ceil((high-low)/manifest["collision"]["cell_size_m"]).astype(int))
            step = (high-low)/counts
            cells = list(itertools.product(*(range(n) for n in counts)))
            assert len(spheres[name]) == len(cells)
            for cell, sphere in zip(cells, spheres[name]):
                corners = low + (np.array(cell) + np.array(list(itertools.product((0, 1), repeat=3)))) * step
                assert np.all(np.linalg.norm(corners - sphere[:3], axis=1) <= sphere[3] + 1e-12)


def test_parent_bounds_and_cross_arm_pairs(manifest):
    spheres = read_yaml(CONFIG / "collision_spheres.yaml")
    bounds = read_yaml(CONFIG / "collision_parent_bounds.yaml")["parent_bounds"]
    assert set(bounds) == set(spheres) - {"self_collision"}
    for name, entries in bounds.items():
        covered = []
        for bound in entries:
            ids = bound["source_sphere_indices"]
            covered.extend(ids)
            fine = np.array(spheres[name])[ids]
            assert np.all(np.linalg.norm(fine[:, :3]-bound["center"], axis=1) + fine[:, 3]
                          <= bound["radius"] + 1e-12)
        assert sorted(covered) == list(range(len(spheres[name])))
    pair_list = read_yaml(CONFIG / "self_collision_pairs.yaml")["pairs"]
    assert pair_list == [[a, b] for a, peers in spheres["self_collision"].items() for b in peers]
    pairs = {frozenset(pair) for pair in pair_list}
    assert len(pairs) == len(pair_list)
    tools = set(manifest["collision"]["link_envelopes"])
    for tool in tools:
        side = tool.split("_")[0]
        for other in set(spheres) - {"self_collision", tool}:
            rigid = other.startswith(side + "_") or other == "Link7_" + ("L" if side == "left" else "R")
            assert (frozenset((tool, other)) in pairs) is (not rigid), (tool, other)


@pytest.mark.parametrize("travel", [-0.001, 0.046, float("nan"), float("inf")])
def test_exporter_rejects_invalid_travel(reference, travel):
    import copy
    with pytest.raises(ValueError, match="outside"):
        freeze_fingers(copy.deepcopy(reference), travel)


@pytest.fixture(scope="module")
def robot():
    import torch
    from torch_robotics.robots.robot_marvin_bimanual import RobotMarvinBimanual
    instance = RobotMarvinBimanual(tensor_args={"device": "cpu", "dtype": torch.float64})
    yield instance
    instance.cleanup()


def test_runtime_tcp_matches_ros_at_100_legal_configurations(robot, reference, manifest):
    import torch
    limits = read_yaml(CONFIG / "joint_limits.yaml")
    low, high = (np.array([limits[n][key] for n in JOINTS]) for key in ("qmin", "qmax"))
    values = np.random.default_rng(2309).uniform(low, high, (100, 14))
    q = torch.tensor(values, dtype=torch.float64)
    actual = [robot.fk_left(q).detach().numpy(), robot.fk_right(q).detach().numpy()]
    for i, arm_q in enumerate(values):
        qmap = dict(zip(JOINTS, arm_q))
        qmap.update({e["joint"]: e["q"] for e in manifest["frozen_fingers"].values()})
        poses = independent_fk(reference, qmap)
        for side, name in enumerate(TCP):
            np.testing.assert_allclose(actual[side][i], poses[name][:3], atol=1e-6, rtol=0)
    torch.testing.assert_close(robot.get_EE_pose(q), robot.fk_left(q))
    assert robot.q_dim == 14 and robot.asset_hash == manifest["asset_sha256"]
    assert robot.tcp_calibrated is False


def test_runtime_canonical_jacobians_and_parent_sphere_path(robot):
    import torch
    q = torch.linspace(-0.3, 0.4, 14, dtype=torch.float64).unsqueeze(0)
    eps = 1e-6
    jp, pp = robot.jfk_s_collision_spheres_parent_links(q)
    js, ps = robot.jfk_s_collision_spheres(q)
    torch.testing.assert_close(torch.stack(jp), torch.stack(js), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(torch.stack(pp), torch.stack(ps), atol=1e-9, rtol=1e-9)
    selected = [i for i, name in enumerate(robot.collision_sphere_parent_links)
                if name.startswith(("left_", "right_"))][::30]
    jac = torch.stack(js)[selected, 0]
    positions = torch.stack(ps)[selected, 0, :, 3]
    # Spatial twist is referenced to the world origin; convert to point velocity.
    linear = jac[:, :3] + torch.linalg.cross(jac[:, 3:].transpose(-1, -2),
                                            positions[:, None, :], dim=-1).transpose(-1, -2)
    for column in range(14):
        delta = torch.zeros_like(q)
        delta[0, column] = eps
        plus = torch.stack(robot.fk_collision_spheres(q + delta))[selected, 0, :, 3]
        minus = torch.stack(robot.fk_collision_spheres(q - delta))[selected, 0, :, 3]
        torch.testing.assert_close(linear[:, :, column], (plus-minus)/(2*eps), atol=1e-7, rtol=1e-6)
    for fk, jfk, offset in ((robot.fk_left, robot.jfk_left, 0), (robot.fk_right, robot.jfk_right, 7)):
        jacobian, _ = jfk(q)
        for column in range(7):
            delta = torch.zeros_like(q)
            delta[0, offset+column] = eps
            finite_difference = (fk(q+delta)[..., 3] - fk(q-delta)[..., 3])/(2*eps)
            torch.testing.assert_close(jacobian[:, :3, column], finite_difference, atol=1e-7, rtol=1e-6)


def test_runtime_subset_and_cached_jacobians_keep_canonical_order(robot):
    import torch
    from mpd.inference.active_jacobian import ActiveJacobianComputer
    from mpd.inference.collision_risk_selector import FineSphereScanCache
    q = torch.linspace(-0.2, 0.3, 14, dtype=torch.float64).reshape(1, 14)
    names = robot.collision_sphere_unique_parent_links
    parents = [i for i, name in enumerate(names) if name.startswith(("left_", "right_"))]
    indices = torch.tensor(parents)
    spheres = torch.where((robot.collision_sphere_parent_indices[:, None] == indices).any(dim=1))[0]
    computer = ActiveJacobianComputer(robot, use_parent_link_kinematics=True)
    expected_j, expected_p = robot.jfk_s_collision_spheres_parent_links(q)
    expected = (torch.stack(expected_j).transpose(0, 1)[:, spheres],
                torch.stack(expected_p).transpose(0, 1)[:, spheres])
    subset = computer._evaluate_parent_subset(q, parents, spheres)
    related = robot.fk_collision_parent_pose_cache(q)
    cache = FineSphereScanCache(
        related_link_ids=robot.collision_parent_related_link_ids,
        related_poses=tuple(p.reshape(1, 1, 3, 4) for p in related),
        sphere_poses=torch.stack(expected_p).transpose(0, 1).unsqueeze(1),
    )
    cached = computer._evaluate_parent_subset_from_pose_cache(
        cache, torch.tensor([0]), torch.tensor([[0]]), parents, spheres)
    for actual in (subset, cached):
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)


def test_runtime_pika_obstacle_and_zero_pose_self_collision(robot):
    import torch
    q = torch.zeros(1, 14, dtype=torch.float64)
    centers = robot.fk_map_collision(q)[0]
    radii = robot.link_collision_spheres_radii
    is_tool = torch.tensor([name.startswith(("left_", "right_"))
                            for name in robot.collision_sphere_parent_links])
    tool_centers = centers[is_tool]
    distance_to_arms = (torch.cdist(tool_centers, centers[~is_tool])-radii[~is_tool]).min(dim=1).values
    # A small sphere at this tool center is missed by the arm-only sphere model.
    point = tool_centers[torch.argmax(distance_to_arms)]
    assert torch.max(distance_to_arms) > 0.01
    assert torch.min(torch.linalg.norm(centers-point, dim=1)-radii-0.005) < 0
    pairs = torch.tensor([[a, b] for a, b, _, _ in robot.link_self_collision_tuples])
    clearance = (torch.linalg.norm(centers[pairs[:, 0]]-centers[pairs[:, 1]], dim=1)
                 - radii[pairs[:, 0]]-radii[pairs[:, 1]])
    assert torch.min(clearance) >= -1e-8


def test_runtime_legacy_arm_only_option():
    import torch
    from torch_robotics.robots.robot_marvin_bimanual import RobotMarvinBimanual
    instance = RobotMarvinBimanual(with_pika=False, tensor_args={"device": "cpu", "dtype": torch.float64})
    try:
        assert instance.q_dim == 14 and instance.link_name_ee_left == "flange_L"
        assert instance.link_name_ee_right == "flange_R"
        assert not any(name.startswith(("left_", "right_"))
                       for name in instance.collision_sphere_unique_parent_links)
    finally:
        instance.cleanup()


def test_pybullet_loads_self_contained_14_joint_model():
    import pybullet as pb
    client = pb.connect(pb.DIRECT)
    try:
        robot_id = pb.loadURDF(str(MODEL / "marvin_pika_bimanual_mpd.urdf"),
                              useFixedBase=True, physicsClientId=client)
        names = [pb.getJointInfo(robot_id, i, physicsClientId=client)[1].decode()
                 for i in range(pb.getNumJoints(robot_id, physicsClientId=client))
                 if pb.getJointInfo(robot_id, i, physicsClientId=client)[2] != pb.JOINT_FIXED]
        assert names == JOINTS
    finally:
        pb.disconnect(client)


@pytest.mark.skipif(not os.environ.get("MARVIN_ROS_ROOT"), reason="Set MARVIN_ROS_ROOT for live ROS/MPD gate")
def test_live_ros_sources_match_generated_assets():
    ros = Path(os.environ["MARVIN_ROS_ROOT"]).resolve()
    subprocess.run(["pixi", "run", "--manifest-path", str(ros / "pixi.toml"),
                    "python", "-m", "scripts.robots.build_marvin_pika_assets",
                    "--ros-root", str(ros), "--check"], cwd=ROOT, check=True, timeout=90)
