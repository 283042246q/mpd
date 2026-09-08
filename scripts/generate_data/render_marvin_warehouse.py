#!/usr/bin/env python3
"""Render the actual PyBullet warehouse and a validated two-TCP shelf pose."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from scripts.generate_data.generate_marvin_warehouse_bimanual import DEFAULT_CONFIG, MarvinWarehouseGenerator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    config.update(sampler="region_ik", ik_tries=200)
    generator = MarvinWarehouseGenerator(config, 2026)
    try:
        q = None
        for _ in range(10):
            q = generator._sample_endpoint(
                np.zeros(14), "dual_independent", {"left": "left_cabinet", "right": "right_cabinet"}
            )
            if q is not None:
                break
        if q is None:
            raise RuntimeError("could not find a jointly valid shelf pose")
        generator.robot.set_state(q)
        client = generator.worker.pybullet_client
        for name, region in generator.regions.items():
            bounds = np.array([region["translation"][a][0] for a in "xyz"])
            color = [0.1, 0.5, 1.0, 0.3] if name.startswith("left") else [1.0, 0.4, 0.1, 0.3]
            visual = client.createVisualShape(
                client.GEOM_BOX, halfExtents=(bounds[:, 1] - bounds[:, 0]) / 2, rgbaColor=color
            )
            client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=bounds.mean(axis=1))
        view = client.computeViewMatrix(
            cameraEyePosition=[-2.2, 0, 1.7], cameraTargetPosition=[0.25, 0, 0.0], cameraUpVector=[0, 0, 1]
        )
        projection = client.computeProjectionMatrixFOV(fov=45, aspect=1.5, nearVal=0.01, farVal=10)
        pixels = client.getCameraImage(
            1500, 1000, viewMatrix=view, projectionMatrix=projection, renderer=client.ER_TINY_RENDERER, shadow=1
        )[2]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(args.output)
        print("collision-valid shelf q:", q.tolist())
    finally:
        generator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
