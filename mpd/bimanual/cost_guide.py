"""Static 14D MPD guide for Marvin's fixed-slot dual end effectors."""

import torch
from dotmap import DotMap

from mpd.inference import cost_guides as common_costs
from mpd.inference.cost_guides import (
    CostGuideManagerParametricTrajectory,
    CostTaskSpace,
    NoCostException,
    map_jacobian_from_world_to_local_world_aligned,
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
from torchlie.functional import SE3 as SE3_Func


def _homogeneous(pose):
    bottom = torch.zeros((*pose.shape[:-2], 1, 4), dtype=pose.dtype, device=pose.device)
    bottom[..., 0, 3] = 1.0
    return torch.cat((pose, bottom), dim=-2)


def _pose34(transform):
    return transform[..., :3, :]


class CostTaskSpaceCooperativeClosure(CostTaskSpace):
    """Path-wide rigid-grasp cost using both inferred object frames.

    Each arm is pulled toward the object frame inferred from the other arm.
    This symmetric local objective is paired with the optional hard projector;
    it is not used as a substitute for final closure validation.
    """

    def __init__(
        self,
        planning_task,
        translation_scale=0.01,
        rotation_scale=0.0523598776,
        **kwargs,
    ):
        super().__init__(planning_task, **kwargs)
        if planning_task.object_to_left_grasp is None or planning_task.object_to_right_grasp is None:
            raise ValueError("cooperative closure cost requires both grasp transforms")
        self.translation_scale = float(translation_scale)
        self.rotation_scale = float(rotation_scale)

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
            raise ValueError("cooperative closure cost requires dual-EE kinematics")
        dtype, device = link_poses_th_ee.dtype, link_poses_th_ee.device
        grasp_left = torch.as_tensor(
            self.planning_task.object_to_left_grasp, dtype=dtype, device=device
        )
        grasp_right = torch.as_tensor(
            self.planning_task.object_to_right_grasp, dtype=dtype, device=device
        )
        current_left = link_poses_th_ee[..., 0, :, :]
        current_right = link_poses_th_ee[..., 1, :, :]
        object_left = _homogeneous(current_left) @ torch.linalg.inv(grasp_left)
        object_right = _homogeneous(current_right) @ torch.linalg.inv(grasp_right)
        target_left = _pose34(object_right @ grasp_left)
        target_right = _pose34(object_left @ grasp_right)
        targets = torch.stack((target_left, target_right), dim=-3)
        current_inverse = SE3_Func.inv(link_poses_th_ee)
        error = SE3_Func.log(SE3_Func.compose(targets, current_inverse))
        scale = torch.as_tensor(
            [self.translation_scale] * 3 + [self.rotation_scale] * 3,
            dtype=dtype,
            device=device,
        )
        scaled = error / scale
        cost = 0.25 * scaled.square().sum(dim=(-1, -2))
        gradient_task = -0.5 * error / scale.square()
        gradient_joint = torch.einsum(
            "...adj,...ad->...j", jacs_spatial_th_ee, gradient_task
        )
        return cost, {"pos": gradient_joint}


class CostTaskSpacePayloadCollision(CostTaskSpace):
    """Differentiable box-payload/environment proxy attached to the left TCP."""

    def __init__(
        self,
        planning_task,
        size_xyz=(0.30, 0.24, 0.20),
        proxy_radius=0.025,
        margin=0.01,
        **kwargs,
    ):
        super().__init__(planning_task, **kwargs)
        if planning_task.object_to_left_grasp is None:
            raise ValueError("payload collision cost requires the left grasp transform")
        self.environment_field = planning_task.get_collision_objects_field()
        if self.environment_field is None:
            raise NoCostException
        half = torch.as_tensor(size_xyz, **planning_task.tensor_args) * 0.5
        signs = torch.tensor(
            [
                [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
            ],
            **planning_task.tensor_args,
        )
        faces = torch.tensor(
            [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
            **planning_task.tensor_args,
        )
        self.local_points = torch.cat((signs * half, faces * half), dim=0)
        self.clearance = float(proxy_radius) + float(margin)

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
            raise ValueError("payload collision cost requires dual-EE kinematics")
        left_pose = link_poses_th_ee[..., 0, :, :]
        left_jacobian = jacs_spatial_th_ee[..., 0, :, :]
        grasp = torch.as_tensor(
            self.planning_task.object_to_left_grasp,
            dtype=left_pose.dtype,
            device=left_pose.device,
        )
        world_object = _homogeneous(left_pose) @ torch.linalg.inv(grasp)
        rotation = world_object[..., :3, :3]
        translation = world_object[..., :3, 3]
        local = self.local_points.to(dtype=left_pose.dtype, device=left_pose.device)
        points = torch.einsum("...ij,pj->...pi", rotation, local) + translation.unsqueeze(-2)

        sdf, sdf_gradient = self.environment_field.object_signed_distances(
            points, get_gradient=True
        )
        penetration = torch.relu(self.clearance - sdf)
        if penetration.shape[-2] == 1:
            active_penetration = penetration.squeeze(-2)
            active_sdf_gradient = sdf_gradient.squeeze(-3)
        else:
            active_penetration, obstacle_index = penetration.max(dim=-2)
            gather_index = obstacle_index.unsqueeze(-2).unsqueeze(-1).expand(
                *obstacle_index.shape[:-1], 1, obstacle_index.shape[-1], 3
            )
            active_sdf_gradient = sdf_gradient.gather(-3, gather_index).squeeze(-3)
        active = active_penetration > 0
        gradient_points = torch.where(
            active.unsqueeze(-1), -active_sdf_gradient, torch.zeros_like(active_sdf_gradient)
        )

        point_poses = world_object[..., None, :3, :].expand(
            *world_object.shape[:-2], local.shape[0], 3, 4
        ).clone()
        point_poses[..., :3, 3] = points
        point_jacobians = left_jacobian.unsqueeze(-3).expand(
            *left_jacobian.shape[:-2], local.shape[0], 6, left_jacobian.shape[-1]
        )
        point_jacobians = map_jacobian_from_world_to_local_world_aligned(
            point_poses, point_jacobians
        )[..., :3, :]
        gradient_joint = torch.einsum(
            "...pdj,...pd->...j", point_jacobians, gradient_points
        ) / float(local.shape[0])
        cost = active_penetration.mean(dim=-1)
        return cost, {"pos": gradient_joint}


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


class CostTaskSpaceObjectGoalComponent(CostTaskSpaceBimanualEEGoalComponent):
    """Terminal object goal expressed through both fixed grasp transforms."""

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
        object_goal = self.planning_task.object_goal_pose
        if object_goal is None:
            raise ValueError("object goal cost requires planning_task.object_goal_pose")
        dtype, device = link_poses_th_ee.dtype, link_poses_th_ee.device
        object_goal = torch.as_tensor(object_goal, dtype=dtype, device=device)
        object_goal_h = _homogeneous(object_goal)
        targets = torch.stack(
            (
                _pose34(
                    object_goal_h
                    @ torch.as_tensor(
                        self.planning_task.object_to_left_grasp,
                        dtype=dtype,
                        device=device,
                    )
                ),
                _pose34(
                    object_goal_h
                    @ torch.as_tensor(
                        self.planning_task.object_to_right_grasp,
                        dtype=dtype,
                        device=device,
                    )
                ),
            ),
            dim=-3,
        )
        cost, gradient, _ = dual_ee_goal_cost_gradient(
            link_poses_th_ee,
            jacs_spatial_th_ee,
            targets,
            self.planning_task.active_ee_mask,
            component=self.component,
            error_scale=self.error_scale,
        )
        if cost.shape[-1] > 1:
            cost = cost.clone()
            cost[..., :-1] = 0.0
        return cost, {"pos": gradient}


class CostTaskSpaceObjectGoalPosition(CostTaskSpaceObjectGoalComponent):
    component = "position"


class CostTaskSpaceObjectGoalOrientation(CostTaskSpaceObjectGoalComponent):
    component = "orientation"


class CostTaskSpaceObjectGoalPose(CostTaskSpaceObjectGoalComponent):
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
        "CostTaskSpaceCooperativeClosure": CostTaskSpaceCooperativeClosure,
        "CostTaskSpacePayloadCollision": CostTaskSpacePayloadCollision,
        "CostTaskSpaceObjectGoalPosition": CostTaskSpaceObjectGoalPosition,
        "CostTaskSpaceObjectGoalOrientation": CostTaskSpaceObjectGoalOrientation,
        "CostTaskSpaceObjectGoalPose": CostTaskSpaceObjectGoalPose,
    }

    EE_GOAL_COST_KEYS = CostGuideManagerParametricTrajectory.EE_GOAL_COST_KEYS | {
        "CostTaskSpaceObjectGoalPosition",
        "CostTaskSpaceObjectGoalOrientation",
        "CostTaskSpaceObjectGoalPose",
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
        raw_args = (
            args_inference.toDict()
            if hasattr(args_inference, "toDict")
            else args_inference
        )
        raw_collision = (
            raw_args.get("collision_optimization", {})
            if isinstance(raw_args, dict)
            else {}
        )
        raw_reduced = (
            raw_collision.get("reduced_guide_geometry", {})
            if isinstance(raw_collision, dict)
            else {}
        )
        has_explicit_switch = (
            isinstance(raw_reduced, dict) and "enabled" in raw_reduced
        )
        if (
            not has_explicit_switch
            and isinstance(planning_task.robot, RobotMarvinBimanual)
            and planning_task.robot.with_pika
        ):
            # Reduced geometry is the Marvin/Pika collision-guidance default,
            # not a global default for Panda, mocks, or non-Pika robots.
            reduced["enabled"] = True
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

        path_wide_ee_costs = {
            "CostTaskSpaceCooperativeClosure",
            "CostTaskSpacePayloadCollision",
        }.intersection(self.costs)
        endpoint_config = self.args_inference.get("gradient_pruning", {}).get(
            "endpoint", {}
        )
        if path_wide_ee_costs and endpoint_config.get("ee_only_last_point", False):
            raise ValueError(
                "cooperative closure/payload guidance requires "
                "gradient_pruning.endpoint.ee_only_last_point=false"
            )

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
