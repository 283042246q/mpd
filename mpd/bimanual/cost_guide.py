"""Static 14D MPD guide for Marvin's fixed-slot dual end effectors."""

import torch
from dotmap import DotMap

from mpd.inference import cost_guides as common_costs
from mpd.inference.cost_guides import (
    CostGuideManagerParametricTrajectory,
    CostTaskSpace,
    NoCostException,
)
from mpd.inference.guidance_config import resolve_collision_optimization_config
from torch_robotics.robots.robot_marvin_bimanual import RobotMarvinBimanual
from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    CollisionObjectDistanceField,
    CollisionWorkspaceBoundariesDistanceField,
)

from .costs import (
    SELF_COLLISION_PAIR_CATEGORIES,
    dual_ee_goal_cost_gradient,
    partition_self_collision_pair_indices,
)


class _CollisionGuideTaskProxy:
    """Expose production scene SDFs with an alternate collision robot.

    The proxy is intentionally private and guidance-only. Dense validation and
    every non-collision cost continue to receive the original planning task.
    """

    def __init__(self, planning_task, robot):
        self._planning_task = planning_task
        self.robot = robot
        self.env = planning_task.env
        self.tensor_args = planning_task.tensor_args
        self.parametric_trajectory = planning_task.parametric_trajectory
        self.df_collision_self = (
            robot.df_collision_self
            if planning_task.get_collision_self_field() is not None
            else None
        )
        self.df_collision_objects = self._object_field(
            planning_task.get_collision_objects_field()
        )
        self.df_collision_extra_objects = self._object_field(
            planning_task.get_collision_extra_objects_field()
        )
        source_workspace = planning_task.get_collision_ws_boundaries_field()
        self.df_collision_ws_boundaries = None
        if source_workspace is not None:
            self.df_collision_ws_boundaries = CollisionWorkspaceBoundariesDistanceField(
                robot,
                link_margins_for_object_collision_checking_tensor=robot.link_collision_spheres_radii,
                cutoff_margin=source_workspace.cutoff_margin,
                clamp_sdf=source_workspace.clamp_sdf,
                ws_bounds_min=source_workspace.ws_min,
                ws_bounds_max=source_workspace.ws_max,
                tensor_args=self.tensor_args,
            )

    def _object_field(self, source):
        if source is None:
            return None
        return CollisionObjectDistanceField(
            self.robot,
            df_obj_list_fn=source.df_obj_list_fn,
            link_margins_for_object_collision_checking_tensor=self.robot.link_collision_spheres_radii,
            cutoff_margin=source.cutoff_margin,
            clamp_sdf=source.clamp_sdf,
            tensor_args=self.tensor_args,
        )

    def get_collision_self_field(self):
        return self.df_collision_self

    def get_collision_objects_field(self):
        return self.df_collision_objects

    def get_collision_extra_objects_field(self):
        return self.df_collision_extra_objects

    def get_collision_ws_boundaries_field(self):
        return self.df_collision_ws_boundaries

    def __getattr__(self, name):
        return getattr(self._planning_task, name)


class _CostTaskSpaceCollisionSelfSubset(common_costs.CostTaskSpaceCollisionSelf):
    """Apply the common analytic self-collision gradient to one pair class."""

    pair_category = None

    def __init__(self, planning_task, **kwargs):
        super().__init__(planning_task, **kwargs)
        partitions = partition_self_collision_pair_indices(self.robot)
        self.pair_indices = partitions[self.pair_category]
        self.pair_category_id = SELF_COLLISION_PAIR_CATEGORIES.index(
            self.pair_category
        )
        if not self.pair_indices:
            raise NoCostException

    def compute_cost_grad_wrt_q(
        self,
        control_points,
        q_traj_pos_in_phase,
        q_traj_vel_in_phase,
        q_traj_acc_in_phase,
        x_poses,
        jacobians_spatial,
        *args,
        **kwargs,
    ):
        requested = kwargs.get("self_pair_indices")
        device = x_poses.device
        allowed = torch.as_tensor(self.pair_indices, dtype=torch.long, device=device)
        if requested is not None:
            requested = torch.as_tensor(requested, dtype=torch.long, device=device)
            category_ids = self.robot._bimanual_self_collision_pair_category_ids.to(
                device
            )
            allowed = requested[
                category_ids.index_select(0, requested) == self.pair_category_id
            ]
        link_indices = kwargs.get("link_indices")
        if link_indices is None:
            link_indices = torch.arange(
                len(self.robot.link_collision_spheres_names),
                dtype=torch.long,
                device=device,
            )
        return super().compute_cost_grad_wrt_q(
            control_points,
            q_traj_pos_in_phase,
            q_traj_vel_in_phase,
            q_traj_acc_in_phase,
            x_poses,
            jacobians_spatial,
            *args,
            **{
                **kwargs,
                "link_indices": link_indices,
                "self_pair_indices": allowed,
            },
        )


class CostTaskSpaceCollisionSelfLeftArm(_CostTaskSpaceCollisionSelfSubset):
    pair_category = "left_intraarm"


class CostTaskSpaceCollisionSelfRightArm(_CostTaskSpaceCollisionSelfSubset):
    pair_category = "right_intraarm"


class CostTaskSpaceCollisionInterArm(_CostTaskSpaceCollisionSelfSubset):
    pair_category = "interarm"


class CostTaskSpaceBimanualEEGoalComponent(CostTaskSpace):
    component = "pose"

    def __init__(self, planning_task, error_scale=1.0, **kwargs):
        super().__init__(planning_task, **kwargs)
        self.error_scale = float(error_scale)

    def compute_cost_grad_wrt_q(
        self,
        control_points,
        q_traj_pos_in_phase,
        q_traj_vel_in_phase,
        q_traj_acc_in_phase,
        link_poses_th,
        jacs_spatial_th,
        link_poses_th_ee,
        jacs_spatial_th_ee,
        *args,
        **kwargs,
    ):
        if link_poses_th_ee is None or jacs_spatial_th_ee is None:
            raise ValueError("dual-EE cost requires poses and Jacobians")
        cost, gradient, _ = dual_ee_goal_cost_gradient(
            link_poses_th_ee,
            jacs_spatial_th_ee,
            self.planning_task.ee_pose_goal,
            self.planning_task.active_ee_mask,
            component=self.component,
            error_scale=self.error_scale,
        )
        # Match the common EE contract: only the terminal trajectory point has
        # a non-zero cost; the manager also retains only the last CP gradient.
        if cost.shape[-1] > 1:
            cost = cost.clone()
            cost[..., :-1] = 0.0
        return cost, {"pos": gradient}

    def compute_cost_grad_wrt_cp(self, control_points, *args, **kwargs):
        cost, gradient = super().compute_cost_grad_wrt_cp(
            control_points, *args, **kwargs
        )
        gradient[..., :-1, :] = 0.0
        return cost, gradient


class CostTaskSpaceEEGoalPosition(CostTaskSpaceBimanualEEGoalComponent):
    component = "position"


class CostTaskSpaceEEGoalOrientation(CostTaskSpaceBimanualEEGoalComponent):
    component = "orientation"


class CostTaskSpaceEEGoalPose(CostTaskSpaceBimanualEEGoalComponent):
    component = "pose"


class BimanualCostGuideManagerParametricTrajectory(
    CostGuideManagerParametricTrajectory
):
    """Reuse common B-spline/B3 machinery with Marvin dual-EE costs."""

    DUAL_EE_COSTS = {
        "CostTaskSpaceCollisionSelfLeftArm": CostTaskSpaceCollisionSelfLeftArm,
        "CostTaskSpaceCollisionSelfRightArm": CostTaskSpaceCollisionSelfRightArm,
        "CostTaskSpaceCollisionInterArm": CostTaskSpaceCollisionInterArm,
        "CostTaskSpaceEEGoalPosition": CostTaskSpaceEEGoalPosition,
        "CostTaskSpaceEEGoalOrientation": CostTaskSpaceEEGoalOrientation,
        "CostTaskSpaceEEGoalPose": CostTaskSpaceEEGoalPose,
    }

    COLLISION_COST_KEYS = {
        "CostTaskSpaceCollisionObjects",
        "CostTaskSpaceCollisionSelf",
        "CostTaskSpaceCollisionSelfLeftArm",
        "CostTaskSpaceCollisionSelfRightArm",
        "CostTaskSpaceCollisionInterArm",
    }

    def __init__(
        self,
        planning_task,
        dataset,
        args_inference,
        tensor_args=None,
        debug=False,
        **kwargs,
    ):
        collision_config = resolve_collision_optimization_config(args_inference)
        reduced = collision_config["reduced_guide_geometry"]
        collision_guide_task = None
        self.guide_collision_robot = None
        if reduced["enabled"]:
            if not isinstance(planning_task.robot, RobotMarvinBimanual):
                raise TypeError(
                    "reduced Marvin guide geometry requires RobotMarvinBimanual"
                )
            self.guide_collision_robot = RobotMarvinBimanual(
                with_pika=planning_task.robot.with_pika,
                collision_geometry_profile=reduced["profile"],
                grasped_object=planning_task.robot.grasped_object,
                tensor_args=planning_task.robot.tensor_args,
            )
            collision_guide_task = _CollisionGuideTaskProxy(
                planning_task, self.guide_collision_robot
            )
        manager_kwargs = {
            **kwargs,
            "debug": debug,
            "collision_guide_task": collision_guide_task,
        }
        if tensor_args is not None:
            manager_kwargs["tensor_args"] = tensor_args
        super().__init__(planning_task, dataset, args_inference, **manager_kwargs)

    def setup_costs(self):
        if not getattr(self.dataset, "context_ee_goal_pose_bimanual", False):
            raise ValueError("bimanual cost guide requires dual-EE dataset context")
        for cost_key in self.args_inference.costs:
            options = self.args_inference.costs[cost_key]
            cost_class = self.DUAL_EE_COSTS.get(cost_key)
            if cost_class is None:
                cost_class = getattr(common_costs, cost_key, None)
            if cost_class is None:
                raise ValueError(f"Unknown bimanual inference cost: {cost_key}")
            try:
                cost_task = (
                    self.collision_guide_task
                    if cost_key in self.COLLISION_COST_KEYS
                    and self.collision_guide_task is not None
                    else self.planning_task
                )
                cost = cost_class(cost_task, **options)
            except NoCostException:
                continue
            self.costs[cost_key] = DotMap(cost=cost, weight=options.weight)

        split_keys = {
            "CostTaskSpaceCollisionSelfLeftArm",
            "CostTaskSpaceCollisionSelfRightArm",
            "CostTaskSpaceCollisionInterArm",
        }
        configured_split = split_keys.intersection(self.args_inference.costs)
        if configured_split and "CostTaskSpaceCollisionSelf" in self.args_inference.costs:
            raise ValueError(
                "Do not combine the unified self-collision cost with split bimanual costs"
            )
        if configured_split and configured_split != split_keys:
            missing = sorted(split_keys - configured_split)
            raise ValueError(f"Split bimanual collision costs are incomplete; missing {missing}")
        partitions = partition_self_collision_pair_indices(self.robot)
        if configured_split and partitions["shared_base"]:
            raise ValueError(
                "Shared-base-only collision pairs require an explicit cost category"
            )
        parent_links = getattr(
            self.robot,
            "collision_sphere_parent_links",
            self.planning_task.robot.link_collision_spheres_names,
        )
        tuples = self.robot.link_self_collision_tuples
        link_pairs = {
            key: {
                (parent_links[tuples[index][0]], parent_links[tuples[index][1]])
                for index in indices
            }
            for key, indices in partitions.items()
        }
        self.self_collision_pair_counts = {
            "parent_link": {key: len(value) for key, value in link_pairs.items()},
            "fine_sphere": {key: len(value) for key, value in partitions.items()},
        }

    def project(self, q):
        return self.planning_task.project(q)

    def compute_task_cost(self, q):
        return self.planning_task.closure_cost(q)
