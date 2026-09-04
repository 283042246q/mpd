from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "mpd" / "torch_robotics" / "torch_robotics" / "data" / "configs" / "marvin"


def test_joint_order_and_limits_contract():
    expected = [f"Joint{i}_L" for i in range(1, 8)] + [f"Joint{i}_R" for i in range(1, 8)]
    limits = yaml.safe_load((CONFIG / "joint_limits.yaml").read_text())
    assert list(limits) == expected
    assert all(set(value) >= {"qmin", "qmax", "qdot_max", "qddot_max", "qddd_max"} for value in limits.values())


def test_collision_model_contains_both_arms_and_interarm_pairs():
    spheres = yaml.safe_load((CONFIG / "collision_spheres.yaml").read_text())
    assert all(f"Link{i}_{side}" in spheres for side in ("L", "R") for i in range(1, 8))
    assert any("Link3_R" in peers for peers in spheres["self_collision"].values())
    pairs = yaml.safe_load((CONFIG / "self_collision_pairs.yaml").read_text())["pairs"]
    assert pairs
    assert all(left in spheres and right in spheres for left, right in pairs)


def test_exported_urdf_has_local_meshes_and_canonical_joints():
    import xml.etree.ElementTree as ET

    urdf = ROOT / "mpd" / "torch_robotics" / "torch_robotics" / "data" / "urdf" / "robots" / "marvin" / "marvin_bimanual_mpd.urdf"
    root = ET.parse(urdf).getroot()
    movable = [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib["type"] != "fixed"]
    assert movable == [f"Joint{i}_L" for i in range(1, 8)] + [f"Joint{i}_R" for i in range(1, 8)]
    mesh_files = [mesh.attrib["filename"] for mesh in root.findall(".//mesh")]
    assert mesh_files
    assert all(not filename.startswith("package://") for filename in mesh_files)
