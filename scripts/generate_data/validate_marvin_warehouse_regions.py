#!/usr/bin/env python3
"""Audit region corners and orientation extremes with collision-aware IK.

This is finite evidence of reachability, not a proof for every pose in a box.
Every generated endpoint still goes through the exact same runtime filters.
"""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from scripts.generate_data.generate_marvin_warehouse_bimanual import (
    DEFAULT_CONFIG,
    ARM_REGIONS,
    MarvinWarehouseGenerator,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--restarts", type=int, default=100)
    parser.add_argument("--regions", nargs="+")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    config.update(sampler="region_ik", ik_tries=args.restarts)
    generator = MarvinWarehouseGenerator(config, args.seed)
    results = {}
    try:
        for arm, regions in ARM_REGIONS.items():
            for name in regions:
                if args.regions and name not in args.regions:
                    continue
                region = generator.regions[name]
                translation = np.array([region["translation"][a][0] for a in "xyz"])
                rotation = np.array([region["rotation"][a][0] for a in "xyz"])
                base = np.array(region["rotation"]["base"])
                center = translation.mean(axis=1)
                angles_center = rotation.mean(axis=1)
                # Corners inset by 1 mm to avoid numeric boundary ambiguity.
                points = [
                    (
                        np.array(
                            [translation[i, side] + (0.001 if side == 0 else -0.001) for i, side in enumerate(corner)]
                        ),
                        angles_center,
                    )
                    for corner in product((0, 1), repeat=3)
                ]
                points.append((center, angles_center))
                for axis in range(3):
                    for side in (0, 1):
                        angles = angles_center.copy()
                        angles[axis] = rotation[axis, side] + (0.1 if side == 0 else -0.1)
                        points.append((center, angles))
                records = []
                for position, angles in points:
                    target_rotation = base @ Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
                    generator._sample_pose = lambda _name, p=position, r=target_rotation: (p, r)
                    q = generator._target_state(np.zeros(14), arm, name)
                    record = dict(
                        position=position.tolist(), relative_euler_deg=angles.tolist(), reachable=q is not None
                    )
                    if q is not None:
                        pose = generator._pose(q, arm)
                        record.update(
                            q=q.tolist(),
                            position_error=float(np.linalg.norm(pose.translation - position)),
                            orientation_error_deg=float(
                                np.rad2deg(Rotation.from_matrix(target_rotation.T @ pose.rotation).magnitude())
                            ),
                        )
                    records.append(record)
                results[name] = dict(passed=sum(r["reachable"] for r in records), tested=len(records), records=records)
                print(name, results[name]["passed"], "/", len(records), flush=True)
    finally:
        generator.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(seed=args.seed, restarts=args.restarts, regions=results), indent=2))
    return 0 if all(r["passed"] == r["tested"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
