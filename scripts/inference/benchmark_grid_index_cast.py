#!/usr/bin/env python3
"""Benchmark CPU versus device-local GridMapSDF index projection."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def project_cpu(x, limits, map_dim, cmap_dim):
    indices = ((x - limits[0]) / map_dim * cmap_dim).round().type(
        torch.LongTensor
    )
    maximum = torch.tensor(tuple(int(value) for value in cmap_dim.tolist())) - 1
    return indices.clamp(torch.zeros_like(maximum), maximum)


def project_device(x, limits, map_dim, cmap_dim):
    indices = ((x - limits[0]) / map_dim * cmap_dim).round().to(
        dtype=torch.long
    )
    maximum = cmap_dim.to(device=x.device, dtype=torch.long) - 1
    return indices.clamp(torch.zeros_like(maximum), maximum)


def measure(fn, repeats):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - started) * 1000.0

    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    return value, first_ms, statistics.median(values), min(values), max(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--spheres", type=int, default=69)
    parser.add_argument("--batches", default="8,16,32,64,100")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    limits = torch.tensor(
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    map_dim = limits[1] - limits[0]
    cmap_dim = torch.tensor([128, 128, 128], dtype=torch.long, device=device)
    sdf = torch.randn(tuple(cmap_dim.tolist()), device=device)

    rows = []
    for batch in [int(value) for value in args.batches.split(",")]:
        torch.manual_seed(1000 + batch)
        x = -1.2 + 2.4 * torch.rand(
            batch,
            args.horizon,
            args.spheres,
            3,
            device=device,
        )

        cpu_project = lambda: project_cpu(x, limits, map_dim, cmap_dim)
        gpu_project = lambda: project_device(x, limits, map_dim, cmap_dim)
        cpu_idx, cpu_first, cpu_p50, cpu_min, cpu_max = measure(
            cpu_project, args.repeats
        )
        gpu_idx, gpu_first, gpu_p50, gpu_min, gpu_max = measure(
            gpu_project, args.repeats
        )
        if not torch.equal(cpu_idx, gpu_idx.cpu()):
            raise RuntimeError("CPU and GPU projected indices differ.")

        def cpu_lookup():
            index = cpu_project()
            return sdf[index[..., 0], index[..., 1], index[..., 2]]

        def gpu_lookup():
            index = gpu_project()
            return sdf[index[..., 0], index[..., 1], index[..., 2]]

        cpu_values, cpu_lookup_first, cpu_lookup_p50, _, _ = measure(
            cpu_lookup, args.repeats
        )
        gpu_values, gpu_lookup_first, gpu_lookup_p50, _, _ = measure(
            gpu_lookup, args.repeats
        )
        if not torch.equal(cpu_values, gpu_values):
            raise RuntimeError("CPU and GPU SDF lookup values differ.")

        rows.append(
            {
                "batch": batch,
                "points": batch * args.horizon * args.spheres,
                "projection_cpu_first_ms": cpu_first,
                "projection_gpu_first_ms": gpu_first,
                "projection_cpu_p50_ms": cpu_p50,
                "projection_gpu_p50_ms": gpu_p50,
                "projection_speedup": cpu_p50 / gpu_p50,
                "lookup_cpu_first_ms": cpu_lookup_first,
                "lookup_gpu_first_ms": gpu_lookup_first,
                "lookup_cpu_p50_ms": cpu_lookup_p50,
                "lookup_gpu_p50_ms": gpu_lookup_p50,
                "lookup_speedup": cpu_lookup_p50 / gpu_lookup_p50,
                "projection_cpu_min_ms": cpu_min,
                "projection_gpu_min_ms": gpu_min,
                "projection_cpu_max_ms": cpu_max,
                "projection_gpu_max_ms": gpu_max,
            }
        )

    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
