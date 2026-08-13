#!/usr/bin/env python3
"""Convert FR3 joint positions to the Cartesian pose used by MPD."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from mpd.utils.patches import numpy_monkey_patch

numpy_monkey_patch()

import numpy as np
import torch

from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.torch_kinematics_tree.geometrics.quaternion import q_to_euler


JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
ROBOT_MODEL = "franka_fr3"
REFERENCE_FRAME = "fr3_link0"
END_EFFECTOR_FRAME = "fr3_hand"
INPUT_KEYS = ("joint_positions", "q", "q_pos", "q_pos_goal", "positions")


class InputError(ValueError):
    """Raised when joint-position input does not satisfy the FK contract."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one or more FR3 joint-space configurations to Cartesian " "poses using the same kinematics as MPD."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--joints",
        nargs=7,
        type=float,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
        help="Seven joint positions in radians.",
    )
    source.add_argument(
        "--input",
        type=Path,
        help="JSON/text input file. Use '-' to read standard input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally write the JSON result to this file.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: 2; use 0 for compact output).",
    )
    return parser.parse_args()


def _extract_json_positions(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in INPUT_KEYS:
        if key in value:
            return value[key]
    accepted = ", ".join(INPUT_KEYS)
    raise InputError(f"JSON object must contain one of these keys: {accepted}.")


def _normalize_positions(value: Any) -> np.ndarray:
    try:
        positions = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InputError(f"Joint positions must be numeric: {error}") from error

    if positions.ndim == 1:
        if positions.shape != (7,):
            raise InputError(f"Expected exactly 7 joint positions, got shape {positions.shape}.")
        positions = positions.reshape(1, 7)
    elif positions.ndim == 2:
        if positions.shape[0] == 0 or positions.shape[1] != 7:
            raise InputError(
                "A batch must have shape [N, 7] with at least one configuration; " f"got {positions.shape}."
            )
    else:
        raise InputError(f"Joint positions must have shape [7] or [N, 7], got {positions.shape}.")

    if not np.isfinite(positions).all():
        raise InputError("Joint positions cannot contain NaN or infinity.")
    return positions


def _parse_text_positions(text: str) -> np.ndarray:
    normalized = text
    for character in "[],;":
        normalized = normalized.replace(character, " ")
    values = np.fromstring(normalized, dtype=np.float64, sep=" ")
    if values.size == 0:
        raise InputError("No joint positions were found in the input.")
    if values.size % 7 != 0:
        raise InputError(f"Plain-text input must contain a multiple of 7 values; got {values.size}.")
    return _normalize_positions(values.reshape(-1, 7))


def _parse_input_text(text: str) -> np.ndarray:
    if not text.strip():
        raise InputError("Input is empty.")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return _parse_text_positions(text)
    return _normalize_positions(_extract_json_positions(value))


def _read_positions(args: argparse.Namespace) -> np.ndarray:
    if args.joints is not None:
        return _normalize_positions(args.joints)

    if args.input is not None:
        if args.input.as_posix() == "-":
            return _parse_input_text(sys.stdin.read())
        try:
            return _parse_input_text(args.input.read_text(encoding="utf-8"))
        except OSError as error:
            raise InputError(f"Cannot read {args.input}: {error}") from error

    if sys.stdin.isatty():
        raise InputError("Provide --joints, --input FILE, or joint positions on stdin.")
    return _parse_input_text(sys.stdin.read())


def _finite_list(values: np.ndarray) -> list[Any]:
    if not all(math.isfinite(float(item)) for item in values.reshape(-1)):
        raise RuntimeError("Forward kinematics produced a non-finite result.")
    return values.tolist()


def compute_forward_kinematics(positions: np.ndarray) -> dict[str, Any]:
    """Compute MPD-compatible FR3 hand poses for a validated [N, 7] array."""
    tensor_args = {"device": "cpu", "dtype": torch.float64}

    # RobotBase prints URDF diagnostics during construction. Keep stdout reserved
    # for the machine-readable JSON result.
    with redirect_stdout(io.StringIO()):
        robot = RobotPanda(gripper=True, tensor_args=tensor_args)

    q = torch.as_tensor(positions, **tensor_args)
    with torch.no_grad():
        transforms_3x4 = robot.get_EE_pose(q)
        quaternions_wxyz = robot.get_EE_orientation(q, rotation_matrix=False)
        rpy_rad = q_to_euler(quaternions_wxyz)

    transforms = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(q), axis=0)
    transforms[:, :3, :4] = transforms_3x4.detach().cpu().numpy()
    quaternions_wxyz_np = quaternions_wxyz.detach().cpu().numpy()
    quaternions_xyzw_np = quaternions_wxyz_np[:, [1, 2, 3, 0]]
    rpy_rad_np = rpy_rad.detach().cpu().numpy()

    poses = []
    for index in range(len(positions)):
        transform = transforms[index]
        poses.append(
            {
                "index": index,
                "joint_positions_rad": _finite_list(positions[index]),
                "position_m": _finite_list(transform[:3, 3]),
                "quaternion_xyzw": _finite_list(quaternions_xyzw_np[index]),
                "quaternion_wxyz": _finite_list(quaternions_wxyz_np[index]),
                "rpy_rad": _finite_list(rpy_rad_np[index]),
                "rpy_deg": _finite_list(np.rad2deg(rpy_rad_np[index])),
                "transform_matrix": _finite_list(transform),
            }
        )

    return {
        "schema_version": 1,
        "robot_model": ROBOT_MODEL,
        "reference_frame": REFERENCE_FRAME,
        "end_effector_frame": END_EFFECTOR_FRAME,
        "joint_names": list(JOINT_NAMES),
        "units": {
            "joint_positions": "rad",
            "position": "m",
            "quaternion": "unitless",
            "rpy": "rad and deg",
        },
        "poses": poses,
    }


def _serialize_json(payload: dict[str, Any], indent: int) -> str:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=None if indent == 0 else indent,
        )
        + "\n"
    )


def _atomic_write(path: Path, serialized: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(serialized)
        stream.flush()
    temporary_path.replace(path)


def main() -> int:
    args = _parse_args()
    if args.indent < 0:
        print("error: --indent must be non-negative.", file=sys.stderr)
        return 2

    try:
        positions = _read_positions(args)
        payload = compute_forward_kinematics(positions)
        serialized = _serialize_json(payload, args.indent)
        if args.output is not None:
            _atomic_write(args.output, serialized)
        sys.stdout.write(serialized)
    except (InputError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
