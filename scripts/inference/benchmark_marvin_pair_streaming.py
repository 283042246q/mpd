#!/usr/bin/env python3
"""Benchmark exact Marvin self-pair full-tensor and streaming reductions."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from mpd.bimanual.costs import partition_self_collision_pair_indices
from torch_robotics.robots.robot_marvin_bimanual import RobotMarvinBimanual
from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    CollisionSelfField,
)


def _cuda_snapshot(device):
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _cleanup(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _run(field, positions, pair_indices, *, pair_chunk_size, repeats):
    device = positions.device
    _cleanup(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = _cuda_snapshot(device)
    try:
        timings = []
        cost = gradient = None
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            cost, gradient = field.compute_distance_field_cost_and_gradient(
                positions,
                self_pair_indices=pair_indices,
                pair_chunk_size=pair_chunk_size,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
        result = {
            "status": "success",
            "elapsed_seconds": timings,
            "cost_sum": float(cost.sum().detach().cpu()),
            "gradient_norm": float(torch.linalg.norm(gradient).detach().cpu()),
        }
    except torch.OutOfMemoryError as error:
        result = {"status": "cuda_oom", "error": str(error)}
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        result = {"status": "cuda_oom", "error": str(error)}
    result["memory_before"] = before
    if device.type == "cuda":
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--pair-category",
        choices=("all", "left_intraarm", "right_intraarm", "interarm"),
        default="interarm",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("full", "streaming"), default=("full", "streaming")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.horizon, args.chunk_size, args.repeats) < 1:
        raise SystemExit("batch-size, horizon, chunk-size and repeats must be positive")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tensor_args = {"device": device, "dtype": dtype}
    robot = RobotMarvinBimanual(tensor_args=tensor_args)
    field = CollisionSelfField(
        robot=robot,
        link_self_collision_tuples=robot.link_self_collision_tuples,
        tensor_args=tensor_args,
    )
    partitions = partition_self_collision_pair_indices(robot)
    pair_indices = None
    pair_count = len(robot.link_self_collision_tuples)
    if args.pair_category != "all":
        pair_indices = torch.as_tensor(
            partitions[args.pair_category], dtype=torch.long, device=device
        )
        pair_count = int(pair_indices.numel())

    # The benchmark isolates the pair reduction. Local sphere centers plus a
    # small deterministic displacement provide a real Marvin-sized tensor
    # without including TorchKin setup/FK in the measured section.
    generator = torch.Generator(device=device).manual_seed(47)
    centers = robot.collision_sphere_local_positions.to(dtype=dtype, device=device)
    positions = centers[None, None].expand(
        args.batch_size, args.horizon, -1, -1
    ).clone()
    positions.add_(
        0.01
        * torch.randn(
            positions.shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )
    )

    report = {
        "schema": "marvin_pair_streaming_benchmark/v1",
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "sphere_count": len(robot.link_collision_spheres_names),
        "pair_category": args.pair_category,
        "pair_count": pair_count,
        "chunk_size": args.chunk_size,
        "runs": {},
    }
    for mode in args.modes:
        report["runs"][mode] = _run(
            field,
            positions,
            pair_indices,
            pair_chunk_size=args.chunk_size if mode == "streaming" else None,
            repeats=args.repeats,
        )
        _cleanup(device)

    full = report["runs"].get("full")
    streaming = report["runs"].get("streaming")
    if full and streaming and full["status"] == streaming["status"] == "success":
        report["equivalence"] = {
            "cost_sum_abs_difference": abs(full["cost_sum"] - streaming["cost_sum"]),
            "gradient_norm_abs_difference": abs(
                full["gradient_norm"] - streaming["gradient_norm"]
            ),
        }

    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
