import numpy as np
import torch

import torch_robotics.robots as tr_robots
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import MultiBoxField, ObjectField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


# Two front pillars form the entrance. The offset rear pillar blocks a straight
# continuation, leaving two arm-width diagonal routes through the cluster.
THREE_PILLAR_BOXES = {
    "centers": [
        [0.32, -0.46, 0.65],
        [0.32, 0.46, 0.65],
        [0.55, 0.00, 0.65],
    ],
    "sizes": [
        [0.16, 0.16, 1.30],
        [0.16, 0.16, 1.30],
        [0.16, 0.16, 1.30],
    ],
}


def create_three_pillars_field(tensor_args=DEFAULT_TENSOR_ARGS):
    """Create three arm-width, floor-standing vertical pillars."""
    pillars = MultiBoxField(
        np.asarray(THREE_PILLAR_BOXES["centers"], dtype=np.float64),
        np.asarray(THREE_PILLAR_BOXES["sizes"], dtype=np.float64),
        tensor_args=tensor_args,
    )
    return ObjectField([pillars], "three_pillars")


class EnvThreePillarsPassage(EnvBase):
    """A Panda scene whose end effector must pass through three tall pillars."""

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        super().__init__(
            limits=torch.tensor([[-0.45, -0.80, -0.10], [1.05, 0.80, 1.40]], **tensor_args),
            obj_fixed_list=[create_three_pillars_field(tensor_args=tensor_args)],
            tensor_args=tensor_args,
            **kwargs,
        )

    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=300,
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
            n_iters=20000,
            step_size=torch.pi / 120,
            n_radius=torch.pi / 4,
            n_pre_samples=50000,
            max_time=45,
        )
        if isinstance(robot, tr_robots.RobotPanda):
            return params
        raise NotImplementedError
