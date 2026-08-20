#!/usr/bin/env python3
"""Export one selectable MPD static scene for dynamic replay recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.isaaclab.scene_payload import export_isaaclab_scene_payload
from torch_robotics.environments import EnvOpenDrawerShelf


def _build_environment(profile: str):
    tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}
    if profile == "to_drawer":
        return EnvOpenDrawerShelf(tensor_args=tensor_args)
    raise ValueError(f"unsupported replay scene profile: {profile!r}")


def export_scene(profile: str) -> dict:
    environment = _build_environment(profile)
    return export_isaaclab_scene_payload(environment, include_boxes=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("to_drawer",), default="to_drawer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = export_scene(args.profile)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
