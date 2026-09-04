from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "mpd" / "torch_robotics" / "torch_robotics" / "data" / "configs" / "marvin"


def test_joint_order_and_limits_contract():
    expected = [f"Joint{i}_L" for i in range(1, 8)] + [f"Joint{i}_R" for i in range(1, 8)]
    limits = yaml.safe_load((CONFIG / "joint_limits.yaml").read_text())
    assert list(limits) == expected
    assert all(set(value) >= {"qdot_max", "qddot_max", "qddd_max"} for value in limits.values())


def test_collision_model_contains_both_arms_and_interarm_pairs():
    spheres = yaml.safe_load((CONFIG / "collision_spheres.yaml").read_text())
    assert all(f"link{i}_{side}" in spheres for side in ("L", "R") for i in range(1, 7))
    assert any("link3_R" in peers for peers in spheres["self_collision"].values())

