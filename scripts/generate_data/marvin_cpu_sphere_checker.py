"""CPU-only Marvin sphere predicate for collision-guided IK proposals."""
from __future__ import annotations

import torch

from torch_robotics.environments.env_warehouse_marvin_bimanual import (
    EnvWarehouseMarvinBimanual,
)
from torch_robotics.robots.robot_marvin_bimanual import RobotMarvinBimanual
from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    CollisionObjectDistanceField,
)


class MarvinCpuSphereChecker:
    def __init__(self, config):
        self.config = dict(config)
        tensor_args = {"device": "cpu", "dtype": torch.float32}
        self.robot = RobotMarvinBimanual(tensor_args=tensor_args)
        environment = EnvWarehouseMarvinBimanual(
            precompute_sdf_obj_fixed=False,
            precompute_sdf_obj_extra=False,
            tensor_args=tensor_args,
        )
        self.object_field = CollisionObjectDistanceField(
            self.robot,
            df_obj_list_fn=environment.get_df_obj_list,
            link_margins_for_object_collision_checking_tensor=(
                self.robot.link_collision_spheres_radii
            ),
            cutoff_margin=float(self.config.get("min_distance_robot_env", 0.02)),
            tensor_args=tensor_args,
        )

    def _positions(self, q):
        parent_poses = torch.stack(
            self.robot.fk_collision_sphere_parent_links(q), dim=1
        )
        selected = parent_poses[:, self.robot.collision_sphere_parent_indices]
        return (
            torch.einsum(
                "bsij,sj->bsi",
                selected[..., :3, :3],
                self.robot.collision_sphere_local_positions,
            )
            + selected[..., :3, 3]
        )

    @torch.no_grad()
    def valid(self, state):
        q = torch.as_tensor(state, **self.robot.tensor_args)
        if q.ndim == 1:
            q = q[None]
        if bool(((q < self.robot.q_pos_min) | (q > self.robot.q_pos_max)).any().item()):
            return False
        positions = self._positions(q)
        self_collision = self.robot.df_collision_self.compute_cost(
            q, positions, field_type="occupancy"
        )
        object_collision = self.object_field.compute_cost(
            q, positions, field_type="occupancy"
        )
        return not bool((self_collision | object_collision).any().item())
