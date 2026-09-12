#!/usr/bin/env python3
"""Benchmark full and exact chunked Marvin dense self-collision validation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import time

import torch

from mpd.bimanual.trajectory_validator import BimanualDenseTrajectoryValidator
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


def _run(validator, q, zeros):
    device = q.device
    _cleanup(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = _memory(device)
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            result = validator.validate(
                q_position=q,
                q_velocity=zeros,
                q_acceleration=zeros,
                num_points=q.shape[1],
                check_environment=False,
                check_joint_velocity=False,
                check_joint_acceleration=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        output = {
            "status": "success",
            "elapsed_seconds": time.perf_counter() - start,
            "valid": result.trajectory_valid_mask.detach().cpu().tolist(),
            "first_invalid_index": result.first_invalid_index.detach().cpu().tolist(),
            "minimum_self_clearance": result.minimum_self_clearance.detach().cpu().tolist(),
            "minimum_left_self_clearance": result.minimum_left_self_clearance.detach().cpu().tolist(),
            "minimum_right_self_clearance": result.minimum_right_self_clearance.detach().cpu().tolist(),
            "minimum_interarm_clearance": result.minimum_interarm_clearance.detach().cpu().tolist(),
        }
    except torch.cuda.
    
    OutOfMemoryError as error:
        output = {"status": "cuda_oom", "error": str(error)}
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        output = {"status": "cuda_oom", "error": str(error)}
    output["memory_before"] = before
    if device.type == "cuda":
        output["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        output["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--candidate-chunk-size", type=int, default=4)
    parser.add_argument("--time-chunk-size", type=int, default=16)
    parser.add_argument("--pair-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--modes", nargs="+", choices=("full", "chunked"), default=("full", "chunked")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sizes = (
        args.batch_size,
        args.horizon,
        args.candidate_chunk_size,
        args.time_chunk_size,
        args.pair_chunk_size,
    )
    if min(sizes) < 1:
        raise SystemExit("all batch, horizon and chunk sizes must be positive")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tensor_args = {"device": device, "dtype": dtype}
    robot = RobotMarvinBimanual(tensor_args=tensor_args)
    generator = torch.Generator(device=device).manual_seed(47)
    midpoint = 0.5 * (robot.q_pos_min + robot.q_pos_max)
    span = 0.15 * (robot.q_pos_max - robot.q_pos_min)
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
    zeros = torch.zeros_like(q)
    terminal = q[0, -1:].detach()
    task = SimpleNamespace(
        robot=robot,
        parametric_trajectory=None,
        ee_pose_goal=torch.stack(
            (robot.fk_left(terminal)[0], robot.fk_right(terminal)[0])
        ),
        active_ee_mask=torch.ones(2, dtype=dtype, device=device),
        get_collision_objects_field=lambda: None,
        get_collision_ws_boundaries_field=lambda: None,
        get_collision_self_field=lambda: robot.df_collision_self,
    )

    report = {
        "schema": "marvin_validator_chunking_benchmark/v1",
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "sphere_count": len(robot.link_collision_spheres_names),
        "pair_count": len(robot.link_self_collision_tuples),
        "chunk_sizes": {
            "candidate": args.candidate_chunk_size,
            "time": args.time_chunk_size,
            "pair": args.pair_chunk_size,
        },
        "runs": {},
    }
    for mode in args.modes:
        config = {
            "chunking": {
                "enabled": mode == "chunked",
                "candidate_chunk_size": args.candidate_chunk_size,
                "time_chunk_size": args.time_chunk_size,
                "self_pair_chunk_size": args.pair_chunk_size,
            }
        }
        report["runs"][mode] = _run(
            BimanualDenseTrajectoryValidator(task, config=config), q, zeros
        )
        _cleanup(device)

    full = report["runs"].get("full")
    chunked = report["runs"].get("chunked")
    if full and chunked and full["status"] == chunked["status"] == "success":
        report["equivalent"] = all(
            full[key] == chunked[key]
            for key in ("valid", "first_invalid_index")
        ) and all(
            torch.allclose(torch.tensor(full[key]), torch.tensor(chunked[key]))
            for key in (
                "minimum_self_clearance",
                "minimum_left_self_clearance",
                "minimum_right_self_clearance",
                "minimum_interarm_clearance",
            )
        )

    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
