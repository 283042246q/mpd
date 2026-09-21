#!/usr/bin/env python3
"""Double a Marvin Warehouse v3 dataset with time-reversed trajectories.

The output is a new, self-contained dataset directory.  Forward/reverse rows
are adjacent and deliberately share the source ``task_id`` so the existing
task-ID-based train/validation split keeps each pair in the same subset.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch
import yaml

from scripts.generate_data.generate_marvin_warehouse_bimanual import file_sha256
from scripts.generate_data.merge_marvin_warehouse_datasets import (
    REQUIRED_FIELDS,
    validate_sources,
)


DIRECTION_REVERSE = {
    "random_to_placement": "placement_to_random",
    "placement_to_random": "random_to_placement",
    "placement_to_placement": "placement_to_placement",
    "random_to_random": "random_to_random",
}


def _decode_strings(values):
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def reverse_directions(values):
    decoded = _decode_strings(values)
    unknown = sorted(set(decoded).difference(DIRECTION_REVERSE))
    if unknown:
        raise ValueError(f"unsupported trajectory directions: {unknown}")
    return np.asarray([DIRECTION_REVERSE[value] for value in decoded], dtype=object)


def reverse_bspline_parameters(knots, coefficients, degree):
    """Reverse a batch of SciPy-style ``(t, c, k)`` B-splines.

    For ``S(u)`` on ``[a,b]``, the returned parameters represent
    ``S(a + b - u)``.  Marvin stores coefficients as ``[row, joint, point]``.
    """
    knots = np.asarray(knots)
    coefficients = np.asarray(coefficients)
    degree = np.asarray(degree)
    if knots.ndim != 2 or coefficients.ndim != 3 or degree.ndim != 1:
        raise ValueError("expected knots [N,K], coefficients [N,14,C], degree [N]")
    if knots.shape[0] != coefficients.shape[0] or knots.shape[0] != degree.shape[0]:
        raise ValueError("B-spline parameter batch dimensions differ")
    if coefficients.shape[1] != 14:
        raise ValueError(f"Marvin B-spline coefficients must have 14 joints, got {coefficients.shape}")
    if not np.issubdtype(degree.dtype, np.integer) or np.any(degree < 0):
        raise ValueError("B-spline degrees must be nonnegative integers")
    expected_knot_counts = coefficients.shape[-1] + degree + 1
    if np.any(expected_knot_counts != knots.shape[1]):
        raise ValueError("B-spline knot/control-point/degree counts are inconsistent")
    if not np.isfinite(knots).all() or not np.isfinite(coefficients).all():
        raise ValueError("B-spline parameters contain nonfinite values")
    if np.any(np.diff(knots, axis=1) < 0):
        raise ValueError("B-spline knots must be nondecreasing")

    rows = np.arange(len(knots))
    lower = knots[rows, degree][:, None]
    upper = knots[:, coefficients.shape[-1]][:, None]
    reversed_knots = lower + upper - knots[:, ::-1]
    reversed_coefficients = coefficients[:, :, ::-1]
    return (
        np.ascontiguousarray(reversed_knots),
        np.ascontiguousarray(reversed_coefficients),
        degree.copy(),
    )


def _build_dual_fk_pose_fn():
    from torch_robotics.robots import RobotMarvinBimanual

    tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}
    robot = RobotMarvinBimanual(tensor_args=tensor_args)

    def dual_fk(q):
        q_tensor = torch.as_tensor(q, **tensor_args)
        with torch.no_grad():
            poses = torch.stack((robot.fk_left(q_tensor), robot.fk_right(q_tensor)), dim=1)
        if poses.shape != (len(q_tensor), 2, 3, 4) or not torch.isfinite(poses).all():
            raise ValueError(f"invalid Marvin dual-TCP FK result: {tuple(poses.shape)}")
        return poses.cpu().numpy().astype(np.float32, copy=False)

    return dual_fk


def _validate_source_fields(handle):
    fields = set(handle.keys())
    missing = REQUIRED_FIELDS.difference(fields)
    if missing:
        raise ValueError(f"source dataset lacks fields: {sorted(missing)}")
    unexpected = fields.difference(REQUIRED_FIELDS)
    if unexpected:
        raise ValueError("source contains fields with no defined reversal semantics: " f"{sorted(unexpected)}")
    count = len(handle["task_id"])
    if len(np.unique(handle["task_id"][:])) != count:
        raise ValueError(
            "source task_id values must be unique; run reversal only after all source datasets are combined"
        )


def _create_output_datasets(source, destination, count):
    destination.attrs.update(source.attrs)
    destination.attrs["augmentation_schema"] = "marvin_bimanual_warehouse_reverse/v1"
    destination.attrs["forward_reverse_pairs_share_task_id"] = True
    for key, value in source.items():
        destination.create_dataset(
            key,
            shape=(2 * count,) + value.shape[1:],
            dtype=value.dtype,
            compression="gzip",
        )
    destination.create_dataset("augmentation_pair_id", shape=(2 * count,), dtype=np.int64, compression="gzip")
    destination.create_dataset("augmentation_is_reversed", shape=(2 * count,), dtype=np.bool_, compression="gzip")
    destination.create_dataset("augmentation_source_row", shape=(2 * count,), dtype=np.int64, compression="gzip")


def _write_pair(destination, key, output_start, forward, reverse):
    output_stop = output_start + 2 * len(forward)
    destination[key][output_start:output_stop:2] = forward
    destination[key][output_start + 1 : output_stop : 2] = reverse


def _update_counts(counters, task_modes, directions, source_regions, goal_regions):
    counters["task_modes"].update(map(str, task_modes))
    counters["directions"].update(map(str, directions))
    for arm in ("left", "right"):
        source_key = f"source_region_{arm}"
        goal_key = f"goal_region_{arm}"
        counters[source_key].update(map(str, source_regions[arm]))
        counters[goal_key].update(map(str, goal_regions[arm]))
        for mode, direction, source, goal in zip(task_modes, directions, source_regions[arm], goal_regions[arm]):
            active = mode == "dual_independent" or mode == f"{arm}_only"
            if active:
                counters[f"accepted_pairs_{arm}"].update([f"{mode}/{direction}/{source}->{goal}"])


def _write_augmented_hdf5(source_path, target_path, chunk_size, dual_fk_pose_fn):
    counters = {
        "task_modes": Counter(),
        "directions": Counter(),
        **{f"{side}_region_{arm}": Counter() for side in ("source", "goal") for arm in ("left", "right")},
        "accepted_pairs_left": Counter(),
        "accepted_pairs_right": Counter(),
    }
    with h5py.File(source_path, "r") as source, h5py.File(target_path, "x") as destination:
        _validate_source_fields(source)
        count = len(source["task_id"])
        _create_output_datasets(source, destination, count)

        for input_start in range(0, count, chunk_size):
            input_stop = min(input_start + chunk_size, count)
            size = input_stop - input_start
            output_start = 2 * input_start
            rows = slice(input_start, input_stop)

            sol_path = source["sol_path"][rows]
            q_start = source["q_start"][rows]
            q_goal = source["q_goal"][rows]
            knots = source["bspline_params_tt"][rows]
            coefficients = source["bspline_params_cc"][rows]
            degree = source["bspline_params_k"][rows]
            reverse_knots, reverse_coefficients, reverse_degree = reverse_bspline_parameters(
                knots, coefficients, degree
            )
            if not np.allclose(sol_path[:, 0], q_start, atol=2e-5, rtol=0.0):
                raise ValueError(f"sol_path/q_start mismatch in source rows {input_start}:{input_stop}")
            if not np.allclose(sol_path[:, -1], q_goal, atol=2e-5, rtol=0.0):
                raise ValueError(f"sol_path/q_goal mismatch in source rows {input_start}:{input_stop}")

            _write_pair(destination, "sol_path", output_start, sol_path, sol_path[:, ::-1])
            _write_pair(destination, "q_start", output_start, q_start, q_goal)
            _write_pair(destination, "q_goal", output_start, q_goal, q_start)
            _write_pair(destination, "bspline_params_tt", output_start, knots, reverse_knots)
            _write_pair(destination, "bspline_params_cc", output_start, coefficients, reverse_coefficients)
            _write_pair(destination, "bspline_params_k", output_start, degree, reverse_degree)

            task_ids = source["task_id"][rows]
            _write_pair(destination, "task_id", output_start, task_ids, task_ids)
            for key in ("planning_time", "joint_path_length", "task_mode", "active_joint_mask", "active_ee_mask"):
                values = source[key][rows]
                _write_pair(destination, key, output_start, values, values)

            # Recompute both forward and reversed goal poses from the canonical
            # joint goal. This verifies the current robot/TCP model rather than
            # copying potentially stale endpoint metadata.
            _write_pair(
                destination,
                "ee_goal_pose",
                output_start,
                dual_fk_pose_fn(q_goal),
                dual_fk_pose_fn(q_start),
            )

            task_modes = _decode_strings(source["task_mode"][rows])
            directions = _decode_strings(source["direction"][rows])
            reverse_direction = reverse_directions(directions)
            _write_pair(destination, "direction", output_start, directions, reverse_direction)

            forward_source = {}
            forward_goal = {}
            for arm in ("left", "right"):
                source_key = f"source_region_{arm}"
                goal_key = f"goal_region_{arm}"
                forward_source[arm] = _decode_strings(source[source_key][rows])
                forward_goal[arm] = _decode_strings(source[goal_key][rows])
                _write_pair(destination, source_key, output_start, forward_source[arm], forward_goal[arm])
                _write_pair(destination, goal_key, output_start, forward_goal[arm], forward_source[arm])

            paired_modes = np.repeat(task_modes, 2)
            paired_directions = np.column_stack((directions, reverse_direction)).reshape(-1)
            paired_source = {
                arm: np.column_stack((forward_source[arm], forward_goal[arm])).reshape(-1) for arm in ("left", "right")
            }
            paired_goal = {
                arm: np.column_stack((forward_goal[arm], forward_source[arm])).reshape(-1) for arm in ("left", "right")
            }
            _update_counts(counters, paired_modes, paired_directions, paired_source, paired_goal)

            output_stop = output_start + 2 * size
            source_rows = np.arange(input_start, input_stop, dtype=np.int64)
            pair_ids = np.asarray(task_ids, dtype=np.int64)
            destination["augmentation_pair_id"][output_start:output_stop:2] = pair_ids
            destination["augmentation_pair_id"][output_start + 1 : output_stop : 2] = pair_ids
            destination["augmentation_source_row"][output_start:output_stop:2] = source_rows
            destination["augmentation_source_row"][output_start + 1 : output_stop : 2] = source_rows
            destination["augmentation_is_reversed"][output_start:output_stop:2] = False
            destination["augmentation_is_reversed"][output_start + 1 : output_stop : 2] = True

            print(f"augmented rows {input_stop}/{count}", flush=True)
    return counters


def augment_dataset(source, output, chunk_size=256, dual_fk_pose_fn=None, validated=None):
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    if chunk_size < 1:
        raise ValueError("chunk-size must be positive")

    if validated is None:
        validated = validate_sources([source])
    source_root, source_dataset = validated["resolved"][0]
    source_count = validated["total"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if dual_fk_pose_fn is None:
        dual_fk_pose_fn = _build_dual_fk_pose_fn()

    with tempfile.TemporaryDirectory(prefix=f".{output.name}.incomplete-", dir=output.parent) as temporary:
        staging = Path(temporary)
        target_dataset = staging / "dataset_merged.hdf5"
        counters = _write_augmented_hdf5(source_dataset, target_dataset, chunk_size, dual_fk_pose_fn)
        digest = file_sha256(target_dataset)
        augmentation = {
            "schema": "marvin_bimanual_warehouse_reverse/v1",
            "source_root": str(source_root),
            "source_dataset_sha256": validated["sources"][0]["dataset_sha256"],
            "source_num_trajectories": source_count,
            "output_num_trajectories": 2 * source_count,
            "rows_interleaved_forward_reverse": True,
            "pairs_share_task_id": True,
            "split_grouping": "existing training split groups rows by task_id",
            "ee_goal_pose_recomputed_with_canonical_marvin_fk": True,
        }

        manifest = deepcopy(validated["manifest"])
        manifest.update(
            num_trajectories=2 * source_count,
            task_counts=dict(counters["task_modes"]),
            direction_counts=dict(counters["directions"]),
            accepted_pair_counts={arm: dict(counters[f"accepted_pairs_{arm}"]) for arm in ("left", "right")},
            region_counts={
                key: dict(counters[key])
                for key in (
                    "source_region_left",
                    "source_region_right",
                    "goal_region_left",
                    "goal_region_right",
                )
            },
            dataset_sha256=digest,
            reverse_augmentation=augmentation,
        )
        config = deepcopy(validated["config"])
        config["output_dir"] = str(output)
        config["num_trajectories"] = 2 * source_count
        config["reverse_augmentation"] = augmentation
        args = deepcopy(validated["args"])
        args["num_trajectories"] = 2 * source_count
        args["reverse_augmentation"] = augmentation

        (staging / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        (staging / "generation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (staging / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False))
        (staging / "generation_summary.json").write_text(json.dumps(manifest, indent=2))
        (staging / "augmentation_report.json").write_text(
            json.dumps({**augmentation, "dataset_sha256": digest}, indent=2)
        )
        os.replace(staging, output)
    return {**augmentation, "output": str(output), "dataset_sha256": digest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source dataset root or dataset_merged.hdf5")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    args = parser.parse_args(argv)
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    if args.output_dir.expanduser().resolve().exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir.expanduser().resolve()}")

    validated = validate_sources([args.source])
    preview = {
        "source": str(validated["resolved"][0][0]),
        "output": str(args.output_dir.expanduser().resolve()),
        "source_num_trajectories": validated["total"],
        "output_num_trajectories": 2 * validated["total"],
        "pairs_share_task_id": True,
    }
    print(json.dumps(preview, indent=2))
    if args.dry_run:
        return 0

    report = augment_dataset(args.source, args.output_dir, args.chunk_size, validated=validated)
    print(f"augmented {report['source_num_trajectories']} -> {report['output_num_trajectories']} trajectories")
    print(f"output: {report['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
