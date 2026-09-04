#!/usr/bin/env python3
"""Export a ROS Marvin xacro to an MPD URDF with provenance.

This utility is not invoked automatically by ``RobotMarvinBimanual``.  When an
asset root is supplied it also copies Marvin meshes and localizes URDF paths.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

EXPECTED_JOINT_NAMES = tuple(
    [f"Joint{i}_L" for i in range(1, 8)] + [f"Joint{i}_R" for i in range(1, 8)]
)


def _localize_meshes(destination: Path, asset_root: Path) -> None:
    text = destination.read_text()
    marker = "package://marvin_description/meshes/"
    mesh_root = asset_root / "meshes"
    while marker in text:
        start = text.index(marker) + len(marker)
        end = start
        while end < len(text) and text[end] not in '\"<':
            end += 1
        relative = Path(text[start:end])
        source_mesh = mesh_root / relative
        if not source_mesh.is_file():
            raise FileNotFoundError(source_mesh)
        target_mesh = destination.parent / "meshes" / relative
        target_mesh.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_mesh, target_mesh)
        text = text[: start - len(marker)] + f"meshes/{relative.as_posix()}" + text[end:]
    destination.write_text(text)


def export(source: Path, destination: Path, asset_root: Path | None = None) -> Path:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".urdf":
        shutil.copyfile(source, destination)
    elif source.suffix in {".xacro", ".xml"}:
        try:
            result = subprocess.run(
                ["xacro", str(source), "ros2_control:=false"],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "xacro was not found. Run from the ROS2/pixi environment, for example "
                "`cd physical_ai_runtime && source install/setup.bash`, then retry."
            ) from error
        except subprocess.CalledProcessError as error:
            details = (error.stderr or error.stdout or "").strip()
            raise RuntimeError(
                "xacro failed while expanding the model. Source the ROS2 overlay "
                "(`source physical_ai_runtime/install/setup.bash`) so package:// "
                f"dependencies are discoverable. Details: {details}"
            ) from error
        destination.write_text(result.stdout)
    else:
        raise ValueError(f"source must be .xacro, .xml, or .urdf; got {source}")
    if asset_root is not None:
        _localize_meshes(destination, asset_root)
    root = ET.parse(destination).getroot()
    joint_names = tuple(
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("type") != "fixed"
    )
    if joint_names != EXPECTED_JOINT_NAMES:
        raise RuntimeError(
            "The exported model must contain exactly Marvin's 14 arm joints in canonical order; "
            f"found {len(joint_names)} joints: {joint_names}. "
            "Use marvin_description/urdf/marvin.urdf.xacro (arm-only), not the Pika/gripper bringup xacro."
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(f"{digest}  {source}\n")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="marvin_description package root used to copy and localize meshes",
    )
    args = parser.parse_args(argv)
    print(export(args.source, args.destination, args.asset_root))


if __name__ == "__main__":
    main()
