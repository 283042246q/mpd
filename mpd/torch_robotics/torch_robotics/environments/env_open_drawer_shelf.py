import numpy as np
import torch

import torch_robotics.robots as tr_robots
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import MultiBoxField, ObjectField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


# All dimensions are in metres in the Panda base frame. The cabinet and shelf
# open towards -x (towards the robot).
DRAWER_CABINET_BOXES = {
    "centers": [
        [0.72, -0.14, 0.60],  # cabinet side, shelf-facing
        [0.72, 0.58, 0.60],  # cabinet side, outer
        [0.93, 0.22, 0.60],  # cabinet back
        [0.72, 0.22, 1.17],  # cabinet top
        [0.72, 0.22, 0.03],  # cabinet plinth
        [0.72, 0.22, 0.62],  # separator above the large bottom drawer
        [0.465, 0.22, 0.76],  # simplified closed upper drawer front
        [0.465, 0.22, 0.99],  # simplified closed upper drawer front
    ],
    "sizes": [
        [0.48, 0.06, 1.20],
        [0.48, 0.06, 1.20],
        [0.06, 0.78, 1.20],
        [0.48, 0.78, 0.06],
        [0.48, 0.78, 0.06],
        [0.48, 0.78, 0.05],
        [0.05, 0.66, 0.18],
        [0.05, 0.66, 0.18],
    ],
}

OPEN_BOTTOM_DRAWER_BOXES = {
    "centers": [
        [0.525, 0.22, 0.11],  # tray bottom
        [0.525, -0.07, 0.20],  # side wall, shelf-facing
        [0.525, 0.51, 0.20],  # side wall, outer
        [0.25, 0.22, 0.20],  # pulled-out drawer front
        [0.80, 0.22, 0.20],  # drawer back
    ],
    "sizes": [
        [0.55, 0.62, 0.04],
        [0.55, 0.04, 0.20],
        [0.55, 0.04, 0.20],
        [0.04, 0.62, 0.20],
        [0.04, 0.62, 0.20],
    ],
}

ADJACENT_SHELF_BOXES = {
    "centers": [
        [0.70, -0.66, 0.55],  # shelf outer side
        [0.70, -0.20, 0.55],  # shelf cabinet-facing side
        [0.93, -0.43, 0.55],  # shelf back
        [0.70, -0.43, 0.03],  # bottom
        [0.70, -0.43, 0.38],  # lower shelf
        [0.70, -0.43, 0.72],  # middle shelf
        [0.70, -0.43, 1.07],  # top
    ],
    "sizes": [
        [0.48, 0.05, 1.10],
        [0.48, 0.05, 1.10],
        [0.06, 0.51, 1.10],
        [0.48, 0.51, 0.06],
        [0.48, 0.51, 0.05],
        [0.48, 0.51, 0.05],
        [0.48, 0.51, 0.06],
    ],
}


def _box_object(name, box_spec, tensor_args):
    boxes = MultiBoxField(
        np.asarray(box_spec["centers"], dtype=np.float64),
        np.asarray(box_spec["sizes"], dtype=np.float64),
        tensor_args=tensor_args,
    )
    return ObjectField([boxes], name)


def create_open_drawer_cabinet_fields(tensor_args=DEFAULT_TENSOR_ARGS):
    """Build the cabinet carcass and its explicitly opened bottom drawer."""
    return [
        _box_object("drawer_cabinet", DRAWER_CABINET_BOXES, tensor_args),
        _box_object("open_bottom_drawer", OPEN_BOTTOM_DRAWER_BOXES, tensor_args),
    ]


def create_adjacent_shelf_field(tensor_args=DEFAULT_TENSOR_ARGS):
    """Build the open-front shelf immediately beside the drawer cabinet."""
    return _box_object("adjacent_shelf", ADJACENT_SHELF_BOXES, tensor_args)


class EnvOpenDrawerShelf(EnvBase):
    """A Panda scene with one large open bottom drawer and an adjacent shelf."""

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        objects = create_open_drawer_cabinet_fields(tensor_args=tensor_args)
        objects.append(create_adjacent_shelf_field(tensor_args=tensor_args))

        super().__init__(
            limits=torch.tensor([[-0.45, -0.90, -0.10], [1.20, 0.90, 1.40]], **tensor_args),
            obj_fixed_list=objects,
            tensor_args=tensor_args,
            **kwargs,
        )

    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=250,
            num_samples=64,
            sigma_start=1e-3,
            sigma_gp=1e-1,
            sigma_goal_prior=1e-3,
            sigma_coll=1e-4,
            step_size=5e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.1,
            sigma_start_sample=1e-3,
            sigma_goal_sample=1e-3,
            solver_params={"delta": 1e-2, "trust_region": True, "method": "cholesky"},
        )
        if isinstance(robot, tr_robots.RobotPanda):
            return params
        raise NotImplementedError

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=15000,
            step_size=torch.pi / 100,
            n_radius=torch.pi / 4,
            n_pre_samples=50000,
            max_time=30,
        )
        if isinstance(robot, tr_robots.RobotPanda):
            return params
        raise NotImplementedError
