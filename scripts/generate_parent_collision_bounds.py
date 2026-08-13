#!/usr/bin/env python3
"""Generate conservative parent-link bounding spheres from fine collision spheres."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "mpd/torch_robotics/torch_robotics/data/configs/panda/panda_sphere_config.yaml"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "mpd/torch_robotics/torch_robotics/data/configs/panda/"
    "panda_parent_collision_bounds.yaml"
)


def enclosing_sphere(spheres, padding):
    """Return a tight, verified enclosing sphere for a set of input spheres."""

    spheres = np.asarray(spheres, dtype=np.float64)
    centers = spheres[:, :3]
    radii = spheres[:, 3]
    lower = np.min(centers - radii[:, None], axis=0)
    upper = np.max(centers + radii[:, None], axis=0)
    initial_center = (lower + upper) / 2
    initial_radius = np.max(
        np.linalg.norm(centers - initial_center, axis=1) + radii
    )

    def constraints(value):
        return value[3] - np.linalg.norm(centers - value[:3], axis=1) - radii

    candidates = [initial_center]
    for start_center in (initial_center, np.mean(centers, axis=0)):
        result = minimize(
            lambda value: value[3],
            np.r_[start_center, initial_radius],
            jac=lambda value: np.array([0.0, 0.0, 0.0, 1.0]),
            constraints={"type": "ineq", "fun": constraints},
            method="SLSQP",
            options={"ftol": 1e-13, "maxiter": 1000},
        )
        if np.all(np.isfinite(result.x)):
            candidates.append(result.x[:3])

    best_center = min(
        candidates,
        key=lambda center: np.max(
            np.linalg.norm(centers - center, axis=1) + radii
        ),
    )
    exact_cover_radius = np.max(
        np.linalg.norm(centers - best_center, axis=1) + radii
    )
    return best_center, exact_cover_radius + padding


def topology_partitioned_bounds(spheres, count, padding):
    """Split spheres contiguously along their principal spatial axis."""

    spheres = np.asarray(spheres, dtype=np.float64)
    count = min(max(int(count), 1), len(spheres))
    centered = spheres[:, :3] - np.mean(spheres[:, :3], axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis
    order = np.argsort(spheres[:, :3] @ axis, kind="stable")

    best = None
    for cuts in itertools.combinations(range(1, len(spheres)), count - 1):
        groups = np.split(order, cuts)
        bounds = []
        for group in groups:
            center, radius = enclosing_sphere(spheres[group], padding)
            bounds.append((group, center, radius))
        # Avoid one oversized segment first, then minimize total proxy volume.
        # This is deliberately topology-oriented: a long link should be
        # represented by similarly local regions instead of one large ball and
        # a few tiny outlier balls.
        score = (
            max(radius for _, _, radius in bounds),
            sum(radius**3 for _, _, radius in bounds),
            cuts,
        )
        if best is None or score < best[0]:
            best = (score, bounds)
    return best[1]


def default_bound_count(sphere_count):
    if sphere_count <= 1:
        return 1
    if sphere_count <= 5:
        return 2
    return 3


def generate(input_path, padding):
    with input_path.open() as file:
        fine_config = yaml.safe_load(file)

    parent_bounds = {}
    for parent_name, spheres in fine_config.items():
        if parent_name == "self_collision":
            continue
        bounds = topology_partitioned_bounds(
            spheres, default_bound_count(len(spheres)), padding
        )
        parent_bounds[parent_name] = [
            {
                "center": [round(float(value), 9) for value in center],
                "radius": round(float(radius), 9),
                "source_sphere_indices": [int(value) for value in group],
            }
            for group, center, radius in bounds
        ]
    return {
        "metadata": {
            "source": input_path.name,
            "geometry": "conservative_parent_multi_sphere",
            "method": "principal_axis_topology_partition_then_exact_cover",
            "safety_padding": float(padding),
        },
        "parent_bounds": parent_bounds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--padding", type=float, default=1e-6)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the saved output differs instead of rewriting it.",
    )
    args = parser.parse_args()
    generated_data = generate(args.input, args.padding)
    generated = yaml.safe_dump(generated_data, sort_keys=False, width=100)
    if args.check:
        existing = (
            yaml.safe_load(args.output.read_text()) if args.output.exists() else None
        )
        if existing != generated_data:
            raise SystemExit(f"Generated bounds are stale: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)
    print(args.output)


if __name__ == "__main__":
    main()
