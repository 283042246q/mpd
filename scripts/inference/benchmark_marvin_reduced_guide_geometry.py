#!/usr/bin/env python3
"""Compare production and Foam Marvin collision-guide geometry kernels."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from mpd.bimanual.costs import partition_self_collision_pair_indices
from torch_robotics.robots.robot_marvin_bimanual import RobotMarvinBimanual


def _cleanup(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _memory(device):
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _run(profile, args, device, dtype):
    tensor_args = {"device": device, "dtype": dtype}
    robot = RobotMarvinBimanual(
        collision_geometry_profile=profile, tensor_args=tensor_args
    )
    partitions = partition_self_collision_pair_indices(robot)
    pair_indices = None
    if args.pair_category != "all":
        pair_indices = torch.as_tensor(
            partitions[args.pair_category], dtype=torch.long, device=device
        )
    pair_count = (
        len(robot.link_self_collision_tuples)
        if pair_indices is None
        else int(pair_indices.numel())
    )
    centers = robot.collision_sphere_local_positions
    generator = torch.Generator(device=device).manual_seed(47)
    positions = centers[None, None].expand(
        args.batch_size, args.horizon, -1, -1
    ).clone()
    positions.add_(
        0.01
        * torch.randn(
            positions.shape,
            generator=generator,
            dtype=dtype,
            device=device,
        )
    )
    output = {
        "profile": profile,
        "geometry_sha256": robot.collision_geometry_hash,
        "sphere_count": len(robot.link_collision_spheres_names),
        "pair_count": len(robot.link_self_collision_tuples),
        "pair_category_counts": {
            key: len(value) for key, value in partitions.items()
        },
        "measured_pair_count": pair_count,
        "estimated_full_difference_tensor_bytes": (
            args.batch_size
            * args.horizon
            * pair_count
            * 3
            * torch.empty((), dtype=dtype).element_size()
        ),
        "runs": {},
    }
    for mode in args.modes:
        _cleanup(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        before = _memory(device)
        try:
            timings = []
            cost = gradient = None
            for _ in range(args.repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                cost, gradient = robot.df_collision_self.compute_distance_field_cost_and_gradient(
                    positions,
                    self_pair_indices=pair_indices,
                    pair_chunk_size=(
                        args.pair_chunk_size if mode == "streaming" else None
                    ),
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - start)
            run = {
                "status": "success",
                "elapsed_seconds": timings,
                "cost_sum": float(cost.sum().detach().cpu()),
                "gradient_norm": float(torch.linalg.norm(gradient).detach().cpu()),
            }
        except torch.cuda.OutOfMemoryError as error:
            run = {"status": "cuda_oom", "error": str(error)}
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            run = {"status": "cuda_oom", "error": str(error)}
        run["memory_before"] = before
        if device.type == "cuda":
            run["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            run["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        output["runs"][mode] = run
    full = output["runs"].get("full")
    streaming = output["runs"].get("streaming")
    if full and streaming and full["status"] == streaming["status"] == "success":
        output["equivalence"] = {
            "cost_sum_abs_difference": abs(full["cost_sum"] - streaming["cost_sum"]),
            "gradient_norm_abs_difference": abs(
                full["gradient_norm"] - streaming["gradient_norm"]
            ),
        }
    del positions, robot
    _cleanup(device)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--pair-chunk-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--pair-category",
        choices=("all", "left_intraarm", "right_intraarm", "interarm"),
        default="interarm",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("production", "foam_pika_100"),
        default=("production", "foam_pika_100"),
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("full", "streaming"), default=("full", "streaming")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.horizon, args.pair_chunk_size, args.repeats) < 1:
        raise SystemExit("batch, horizon, chunk and repeat values must be positive")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    report = {
        "schema": "marvin_reduced_guide_geometry_benchmark/v1",
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "pair_category": args.pair_category,
        "pair_chunk_size": args.pair_chunk_size,
        "profiles": {},
    }
    for profile in args.profiles:
        report["profiles"][profile] = _run(profile, args, device, dtype)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
