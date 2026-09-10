#!/usr/bin/env python3
"""Benchmark Marvin fine-pair scanning against conservative parent bounds."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from mpd.bimanual.costs import partition_self_collision_pair_indices
from mpd.inference.collision_risk_selector import CollisionRiskSelector
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


def _measure(device, function):
    _cleanup(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = _memory(device)
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            value = function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result = {
            "status": "success",
            "elapsed_seconds": time.perf_counter() - start,
            "value": value,
        }
    except torch.cuda.OutOfMemoryError as error:
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
    parser.add_argument("--self-margin", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.horizon < 1 or args.self_margin < 0:
        raise SystemExit("batch-size/horizon must be positive and self-margin non-negative")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tensor_args = {"device": device, "dtype": dtype}
    robot = RobotMarvinBimanual(tensor_args=tensor_args)
    generator = torch.Generator(device=device).manual_seed(47)
    midpoint = 0.5 * (robot.q_pos_min + robot.q_pos_max)
    span = 0.20 * (robot.q_pos_max - robot.q_pos_min)
    q = midpoint + span * (
        2
        * torch.rand(
            (args.batch_size, args.horizon, robot.q_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        - 1
    )
    selector = CollisionRiskSelector(
        robot,
        config={"coarse_points": args.horizon},
        use_parent_link_kinematics=True,
        link_broad_phase_config={
            "enabled": True,
            "full_scan": True,
            "scan_geometry": "parent_bounds",
            "environment_margin": 0.20,
            "self_margin": args.self_margin,
        },
    )

    fine = _measure(
        device,
        lambda: selector.compute_clearances(
            q, None, robot.df_collision_self, return_details=True
        )[3],
    )
    parent = _measure(
        device,
        lambda: selector.compute_parent_bound_clearances(
            q, None, robot.df_collision_self
        )[3],
    )
    fine_values = fine.pop("value", None)
    parent_values = parent.pop("value", None)
    report = {
        "schema": "marvin_parent_broad_phase_benchmark/v1",
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "sphere_count": len(robot.link_collision_spheres_names),
        "fine_pair_count": len(robot.link_self_collision_tuples),
        "parent_count": len(robot.collision_sphere_unique_parent_links),
        "parent_pair_count": len(robot.collision_parent_self_pairs),
        "self_margin": args.self_margin,
        "runs": {"fine_scan": fine, "parent_scan": parent},
    }
    if parent_values is not None:
        _, _, pair_mask = selector._link_broad_phase_masks_from_parent_bounds(
            torch.full(
                (args.batch_size, args.horizon, len(robot.collision_sphere_unique_parent_links)),
                torch.inf,
                dtype=dtype,
                device=device,
            ),
            parent_values,
        )
        partitions = partition_self_collision_pair_indices(robot)
        active_counts = {}
        for category in ("left_intraarm", "right_intraarm", "interarm"):
            indices = torch.as_tensor(partitions[category], dtype=torch.long, device=device)
            active_counts[category] = pair_mask.index_select(-1, indices).sum(dim=-1).cpu().tolist()
        report["active_fine_pair_counts_by_candidate"] = active_counts
        report["active_fine_pair_ratio"] = float(pair_mask.float().mean().cpu())
        if fine_values is not None:
            risky = fine_values.amin(dim=1) < args.self_margin
            false_negative = risky & ~pair_mask
            report["fine_pair_false_negative_count"] = int(false_negative.sum().cpu())
            report["fine_pair_risky_count"] = int(risky.sum().cpu())

    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
