"""Static 14D MPD guide for Marvin's fixed-slot dual end effectors."""

from dotmap import DotMap

from mpd.inference import cost_guides as common_costs
from mpd.inference.cost_guides import (
    CostGuideManagerParametricTrajectory,
    CostTaskSpace,
    NoCostException,
)

from .costs import dual_ee_goal_cost_gradient


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
        "CostTaskSpaceEEGoalPosition": CostTaskSpaceEEGoalPosition,
        "CostTaskSpaceEEGoalOrientation": CostTaskSpaceEEGoalOrientation,
        "CostTaskSpaceEEGoalPose": CostTaskSpaceEEGoalPose,
    }

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
                cost = cost_class(self.planning_task, **options)
            except NoCostException:
                continue
            self.costs[cost_key] = DotMap(cost=cost, weight=options.weight)

    def project(self, q):
        return self.planning_task.project(q)

    def compute_task_cost(self, q):
        return self.planning_task.closure_cost(q)
