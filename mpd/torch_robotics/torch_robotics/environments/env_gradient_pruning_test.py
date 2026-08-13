"""Deterministic manifest-driven 2D environments for pruning tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

import torch_robotics.robots as tr_robots
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import MultiBoxField, MultiSphereField, ObjectField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


def load_gradient_pruning_2d_scenario(scenario_yaml):
    path = Path(scenario_yaml).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        scenario = yaml.safe_load(stream)
    if scenario.get("type") not in {"simple_2d", "narrow_2d"}:
        raise ValueError(f"Unsupported 2D gradient-pruning scenario type: {scenario.get('type')!r}")
    scenario["source_path"] = path.as_posix()
    return scenario


def build_gradient_pruning_2d_objects(scenario, tensor_args=DEFAULT_TENSOR_ARGS):
    fields = []
    sphere_centers, sphere_radii = [], []
    box_centers, box_sizes = [], []
    for obstacle in scenario.get("obstacles", []):
        if obstacle["shape"] == "circle":
            sphere_centers.append(obstacle["center"])
            sphere_radii.append(obstacle["radius"])
        elif obstacle["shape"] == "box":
            box_centers.append(obstacle["center"])
            box_sizes.append(obstacle["size"])
        else:
            raise ValueError(f"Unknown 2D obstacle shape: {obstacle['shape']!r}")

    if scenario["type"] == "narrow_2d":
        gap_width = float(scenario["gap_width"])
        if not 0 < gap_width < 2:
            raise ValueError("narrow_2d gap_width must be between 0 and 2.")
        wall_height = 1.0 - gap_width / 2.0
        wall_center = (1.0 + gap_width / 2.0) / 2.0
        box_centers.extend([[0.0, -wall_center], [0.0, wall_center]])
        box_sizes.extend([[0.20, wall_height], [0.20, wall_height]])

    if sphere_centers:
        fields.append(
            MultiSphereField(
                np.asarray(sphere_centers, dtype=np.float64),
                np.asarray(sphere_radii, dtype=np.float64),
                tensor_args=tensor_args,
            )
        )
    if box_centers:
        fields.append(
            MultiBoxField(
                np.asarray(box_centers, dtype=np.float64),
                np.asarray(box_sizes, dtype=np.float64),
                tensor_args=tensor_args,
            )
        )
    return [ObjectField(fields, scenario["id"])] if fields else []


class EnvGradientPruning2DTest(EnvBase):
    def __init__(
        self,
        gradient_pruning_scenario_yaml,
        tensor_args=DEFAULT_TENSOR_ARGS,
        precompute_sdf_obj_fixed=False,
        **kwargs,
    ):
        scenario = load_gradient_pruning_2d_scenario(gradient_pruning_scenario_yaml)
        self.gradient_pruning_scenario = scenario
        objects = build_gradient_pruning_2d_objects(scenario, tensor_args=tensor_args)
        super().__init__(
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
            obj_fixed_list=objects,
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            tensor_args=tensor_args,
            **kwargs,
        )

    def get_rrt_connect_params(self, robot=None):
        if not isinstance(robot, tr_robots.RobotPointMass2D):
            raise NotImplementedError
        return dict(n_iters=10000, step_size=0.01, n_radius=0.3, n_pre_samples=50000, max_time=50)
