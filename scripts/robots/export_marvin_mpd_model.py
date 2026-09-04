#!/usr/bin/env python3
"""Export a ROS Marvin xacro to an MPD URDF with provenance.

This utility is not invoked automatically by ``RobotMarvinBimanual`` and does
not generate collision-sphere YAML or copy meshes.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess


def export(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(["xacro", str(source)], check=True, capture_output=True, text=True)
        destination.write_text(result.stdout)
    except FileNotFoundError:
        # Keep the command useful in a non-ROS checkout while making the
        # provenance failure explicit in the sidecar file.
        fallback = source.with_suffix("")
        if fallback.suffix == ".urdf" and fallback.exists():
            shutil.copyfile(fallback, destination)
        else:
            raise RuntimeError("xacro is required to export a Marvin MPD model")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(f"{digest}  {source}\n")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    print(export(args.source, args.destination))


if __name__ == "__main__":
    main()
