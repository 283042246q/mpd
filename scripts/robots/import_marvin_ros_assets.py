#!/usr/bin/env python3
"""Import Marvin ROS2 collision and joint assets into MPD formats."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


JOINT_NAMES = tuple(
    [f"Joint{i}_L" for i in range(1, 8)]
    + [f"Joint{i}_R" for i in range(1, 8)]
)
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))


def convert_joint_limits(source: Path) -> dict:
    native = yaml.safe_load(source.read_text())
    converted = {}
    for side in ("L", "R"):
        for index, native_name in enumerate(ARM_JOINTS, 1):
            limit = native[native_name]["limit"]
            converted[f"Joint{index}_{side}"] = {
                "qmin": float(limit["lower"]),
                "qmax": float(limit["upper"]),
                "qdot_max": float(limit["velocity"]),
                "qddot_max": 8.0,
                "qddd_max": 20.0,
            }
    return converted


def convert_collision(source: Path) -> tuple[dict, dict]:
    native = yaml.safe_load(source.read_text())
    kinematics = native["kinematics"]
    spheres = {
        name: [
            [float(value) for value in entry["center"]]
            + [float(entry["radius"])]
            for entry in entries
        ]
        for name, entries in kinematics["collision_spheres"].items()
    }
    ignored = {
        name: set(values)
        for name, values in kinematics.get("self_collision_ignore", {}).items()
    }
    names = [name for name in kinematics["collision_link_names"] if name in spheres]
    self_collision = {}
    for index, name in enumerate(names):
        values = []
        for other in names[index + 1 :]:
            if other in ignored.get(name, set()) or name in ignored.get(other, set()):
                continue
            values.append(other)
        if values:
            self_collision[name] = values
    spheres["self_collision"] = self_collision
    return spheres, self_collision


def convert(source_root: Path, mpd_root: Path) -> None:
    config_root = source_root / "config"
    output_config = mpd_root / "mpd/torch_robotics/torch_robotics/data/configs/marvin"
    output_config.mkdir(parents=True, exist_ok=True)
    (output_config / "joint_limits.yaml").write_text(
        yaml.safe_dump(convert_joint_limits(config_root / "joint_limits.yaml"), sort_keys=False)
    )
    collision, self_collision = convert_collision(config_root / "curobo/marvin.yml")
    (output_config / "collision_spheres.yaml").write_text(
        yaml.safe_dump(collision, sort_keys=False, width=120)
    )
    pairs = [[name, other] for name, others in self_collision.items() for other in others]
    (output_config / "self_collision_pairs.yaml").write_text(
        yaml.safe_dump({"pairs": pairs}, sort_keys=False, width=120)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--mpd-root", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source_root, args.mpd_root)


if __name__ == "__main__":
    main()
