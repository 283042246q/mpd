import time
from functools import partial
from numbers import Real

import einops
import torch
from dotmap import DotMap

from deps.theseus.torchlie.torchlie.functional.se3_impl import _adjoint_impl
from mpd.parametric_trajectory.trajectory_waypoints import ParametricTrajectoryWaypoints
from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline
from mpd.inference.active_jacobian import ActiveJacobianComputer
from mpd.inference.collision_risk_selector import CollisionRiskSelector, TemporalSelection
from mpd.inference.guidance_config import resolve_gradient_pruning_config
from mpd.inference.guidance_profiler import GuidanceProfiler
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor
from torch_robotics.torch_utils.torch_timer import TimerCUDA

from torchlie.functional import SE3 as SE3_Func

from torch.func import vmap, jacrev, functional_call

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_numpy
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


def project_hierarchical_gradients_fast(grads):
    """
    Faster implementation of hierarchical gradient projection that properly
    preserves higher priority constraints.

    Args:
        grads: List of gradients, highest priority first
    Returns:
        final_grad: Sum of projected gradients
        grads_projected_l: List of individual projected gradients
    """
    if len(grads) == 1:
        return grads[0], grads

    # Pre-compute normalized high-priority gradients
    grads_stack_flatten = einops.rearrange(torch.stack(grads), "... h d -> ... (h d)")
    norms = torch.norm(grads_stack_flatten, dim=-1, keepdim=True)
    normalized_grads = grads_stack_flatten / (norms + 1e-8)

    # Initialize with the highest priority gradient
    grads_projected_l = [
        einops.rearrange(grads_stack_flatten[0], "... (h d) -> ... h d", h=grads[0].shape[-2], d=grads[0].shape[-1])
    ]

    # For each lower priority gradient
    for i in range(1, len(grads_stack_flatten)):
        curr_grad = grads_stack_flatten[i]

        # Project sequentially through all higher priority gradients
        for j in range(i):
            # Create projector for this constraint
            n = normalized_grads[j]
            projection = torch.einsum("...i,...j->...ij", n, n)
            # Apply projection
            curr_grad = curr_grad - torch.einsum("...ij,...j->...i", projection, curr_grad)

        grads_projected_l.append(
            einops.rearrange(curr_grad, "... (h d) -> ... h d", h=grads[0].shape[-2], d=grads[0].shape[-1])
        )

    # Sum all projected gradients
    final_grad = torch.stack(grads_projected_l).sum(dim=0)

    return final_grad, grads_projected_l


class NoCostException(Exception):
    pass


def _is_static_zero_weight(weight):
    """Return true for configuration/schedule scalars without synchronizing CUDA."""

    return isinstance(weight, Real) and float(weight) == 0.0


class CostGuideManagerParametricTrajectory:

    EE_GOAL_COST_KEYS = {
        "CostTaskSpaceEEGoalPose",
        "CostTaskSpaceEEGoalPosition",
        "CostTaskSpaceEEGoalOrientation",
    }

    def __init__(self, planning_task, dataset, args_inference, tensor_args=DEFAULT_TENSOR_ARGS, debug=False, **kwargs):
        self.args_inference = args_inference

        self.tensor_args = tensor_args

        self.planning_task = planning_task
        self.env = planning_task.env
        self.robot = planning_task.robot
        self.parametric_trajectory = planning_task.parametric_trajectory

        self.dataset = dataset

        # control points normalization function gradients
        self.q_cps_unnormalize_fn = lambda x: self.dataset.unnormalize_control_points(x).sum()
        self.grad_cps_wrt_cps_normalized = self.dataset.grad_unnormalized_wrt_control_points_normalized

        # Setup costs
        self.costs = DotMap()
        self.setup_costs()
        if not self.costs:
            raise NoCostException

        self.step_guide_call = 0
        self._t = 0
        self.profile_context_index = None

        self.gradient_pruning_config = resolve_gradient_pruning_config(args_inference)
        self.gradient_pruning_enabled = self.gradient_pruning_config["enabled"]
        self.guidance_profiler = GuidanceProfiler(
            enabled=self.gradient_pruning_config["profile"],
            record_active_statistics=self.gradient_pruning_config["record_active_statistics"],
        )
        self.collision_risk_selector = None
        self.active_jacobian_computer = None
        self._temporal_selection_cache = None
        self._bspline_support_cache = {}
        self.use_parent_link_kinematics = False
        self.use_dense_parent_fast_path = False
        self.use_fused_bspline_integration = False
        self.use_sparse_bspline_support = False
        self.use_active_link_pruning = False
        self.use_link_broad_phase = False
        self.use_span_certificate = False
        if self.gradient_pruning_enabled:
            self.use_parent_link_kinematics = bool(
                self.gradient_pruning_config["spatial"]["parent_link_kinematics"]
            )
            self.use_dense_parent_fast_path = bool(
                self.gradient_pruning_config["spatial"].get("dense_parent_fast_path", True)
            ) and self.use_parent_link_kinematics
            self.use_fused_bspline_integration = bool(
                self.gradient_pruning_config["mapping"].get("fused_bspline_integration", True)
            ) and isinstance(self.parametric_trajectory, ParametricTrajectoryBspline)
            self.use_sparse_bspline_support = bool(
                self.gradient_pruning_config["mapping"].get("sparse_bspline_support", False)
            ) and isinstance(self.parametric_trajectory, ParametricTrajectoryBspline)
            self.use_active_link_pruning = bool(
                self.gradient_pruning_config["spatial"].get("active_link_pruning", False)
            )
            self.use_link_broad_phase = bool(
                self.gradient_pruning_config["spatial"]
                .get("link_broad_phase", {})
                .get("enabled", False)
            )
            self.use_span_certificate = bool(
                self.gradient_pruning_config.get("span_certificate", {}).get(
                    "enabled", False
                )
            )
            self.collision_risk_selector = CollisionRiskSelector(
                self.robot,
                self.gradient_pruning_config["temporal"],
                use_parent_link_kinematics=self.use_parent_link_kinematics,
                candidate_pruning_enabled=self.gradient_pruning_config["candidate"]["enabled"],
                link_broad_phase_config=self.gradient_pruning_config["spatial"].get(
                    "link_broad_phase", {}
                ),
                use_parent_bounds_scan=self.gradient_pruning_config["preselection"].get(
                    "parent_bounds_scan", False
                ),
                span_certificate_config=self.gradient_pruning_config.get(
                    "span_certificate", {}
                ),
                guidance_profiler=self.guidance_profiler,
            )
            self.active_jacobian_computer = ActiveJacobianComputer(
                self.robot,
                use_parent_link_kinematics=self.use_parent_link_kinematics,
            )

        self.debug = debug

    def setup_costs(self):
        for cost_key in self.args_inference.costs:
            if cost_key in self.EE_GOAL_COST_KEYS and not self.dataset.context_ee_goal_pose:
                # Skip the cost if the EE goal context is not set. This means the last joint position is fixed, so the
                # EE pose is determined via FK.
                continue
            try:
                cost_options = self.args_inference.costs[cost_key]
                cost = eval(cost_key)(self.planning_task, **cost_options)
                self.costs[cost_key] = DotMap()
                self.costs[cost_key].cost = cost
                self.costs[cost_key].weight = cost_options.weight
            except NoCostException:
                continue

    def use_all_collision_objects(self):
        """
        Set the cost to use all collision objects.
        """
        cost_key = "CostTaskSpaceCollisionObjects"
        if cost_key in self.costs:
            self.costs[cost_key].cost.use_only_on_extra_objects = False

    def __call__(
        self,
        control_points_normalized,
        return_cost=False,
        warmup=False,
        plot_gradients=False,
        cost_weight_overrides=None,
        **kwargs,
    ):
        # Check the strict top-level switch before constructing any selector,
        # gather/scatter operation, or endpoint-only kinematics.
        if not self.gradient_pruning_enabled:
            return self._legacy_call(
                control_points_normalized,
                return_cost=return_cost,
                warmup=warmup,
                plot_gradients=plot_gradients,
                cost_weight_overrides=cost_weight_overrides,
                **kwargs,
            )
        return self._pruned_call(
            control_points_normalized,
            return_cost=return_cost,
            warmup=warmup,
            plot_gradients=plot_gradients,
            cost_weight_overrides=cost_weight_overrides,
            **kwargs,
        )

    @torch.no_grad()
    def _legacy_call(
        self,
        control_points_normalized,
        return_cost=False,
        warmup=False,
        plot_gradients=False,
        cost_weight_overrides=None,
        **kwargs,
    ):
        """
        Args:
            control_points_normalized: (batch_size, n_control_points, q_dim)
        """
        batch_size = control_points_normalized.shape[0]
        diffusion_timestep = kwargs.get("diffusion_timestep")
        if torch.is_tensor(diffusion_timestep):
            diffusion_timestep = int(diffusion_timestep.flatten()[0].detach().cpu().item())
        self.guidance_profiler.begin_call(
            guide_call_index=self.step_guide_call,
            context_index=self.profile_context_index,
            diffusion_timestep=diffusion_timestep,
            n_candidates=batch_size,
            legacy_path=True,
        )
        if self.debug:
            print()
            print(f"Guide step {self.step_guide_call}")

        # Unnormalize the control points.
        # The generative model outputs normalized control points, but the costs are defined on the unnormalized
        # trajectory space.
        with self.guidance_profiler.section("bspline_expansion"):
            control_points = self.dataset.unnormalize_control_points(control_points_normalized)

            # Get the trajectory (position, velocity, acceleration) from the control points, in phase.
            q_traj_in_phase_d = self.parametric_trajectory.get_q_trajectory(
                control_points, None, None, get_type=("pos", "vel", "acc"), get_time_representation=False
            )
            q_traj_pos_in_phase = q_traj_in_phase_d["pos"]
            q_traj_vel_in_phase = q_traj_in_phase_d["vel"]
            q_traj_acc_in_phase = q_traj_in_phase_d["acc"]

        # Compute forward kinematics and spatial (world) jacobians
        assert q_traj_pos_in_phase.ndim == 3
        q_traj_pos_in_phase_original_shape = q_traj_pos_in_phase.shape
        q_traj_pos_aux = einops.rearrange(q_traj_pos_in_phase, "... d -> (...) d")

        with TimerCUDA() as t_fk_jac:
            # collision links and jacobians
            with self.guidance_profiler.section("collision_fk_jacobian"):
                jacs_spatial, link_poses = self.robot.jfk_s_collision_spheres(q_traj_pos_aux)
            jacs_spatial_th = torch.stack(jacs_spatial).transpose(
                0, 1
            )  # ((batch_size, traejectory_length), n_links, 6, d)
            jacs_spatial_th = einops.rearrange(
                jacs_spatial_th, "(b h) ... -> b h ...", b=q_traj_pos_in_phase_original_shape[0]
            )
            link_poses_th = torch.stack(link_poses).transpose(0, 1)  # ((batch_size, traejectory_length), n_links, 3, 4)
            link_poses_th = einops.rearrange(
                link_poses_th, "(b h) ... -> b h ...", b=q_traj_pos_in_phase_original_shape[0]
            )

            # end effector links and jacobians
            with self.guidance_profiler.section("ee_fk_jacobian"):
                jacs_spatial_ee, link_poses_ee = self.robot.jfk_s_ee(q_traj_pos_aux)
            jacs_spatial_th_ee = torch.stack(jacs_spatial_ee).transpose(
                0, 1
            )  # ((batch_size, traejectory_length), n_links, 6, d)
            jacs_spatial_th_ee = einops.rearrange(
                jacs_spatial_th_ee, "(b h) ... -> b h ...", b=q_traj_pos_in_phase_original_shape[0]
            )
            link_poses_th_ee = torch.stack(link_poses_ee).transpose(
                0, 1
            )  # ((batch_size, traejectory_length), n_links, 3, 4)
            link_poses_th_ee = einops.rearrange(
                link_poses_th_ee, "(b h) ... -> b h ...", b=q_traj_pos_in_phase_original_shape[0]
            )

        if self.debug:
            print(f"FK and Jacobians (time): {t_fk_jac.elapsed:.4f} s")
            print("-" * 50)

        # Compute cost and gradients wrt to the control points normalized
        with TimerCUDA() as t_cost_grad_all:
            cost_all = torch.zeros(batch_size, dtype=control_points.dtype, device=control_points.device)
            grad_costs_wrt_cp_normalized_l = []
            rs_inv = self.parametric_trajectory.phase_time.rs_inv
            s = self.parametric_trajectory.phase_time.s
            for k, cost_key in enumerate(self.costs):
                s_time = time.perf_counter()

                cost_fn = self.costs[cost_key].cost
                weight = self.costs[cost_key].weight
                if cost_weight_overrides and cost_key in cost_weight_overrides:
                    weight = cost_weight_overrides[cost_key]
                if _is_static_zero_weight(weight):
                    continue

                if cost_key == "CostTaskSpaceCollisionObjects":
                    profile_name = "environment_sdf_query"
                elif cost_key == "CostTaskSpaceCollisionSelf":
                    profile_name = "self_collision"
                elif cost_key in self.EE_GOAL_COST_KEYS:
                    profile_name = "ee_cost_mapping"
                else:
                    profile_name = "joint_space_costs"
                with self.guidance_profiler.section(profile_name):
                    cost_single_in_phase, grad_cost_single_wrt_cp_normalized_in_phase = self.compute_cost_grad_cp_normalized(
                        cost_fn,
                        control_points_normalized,
                        control_points,
                        q_traj_pos_in_phase,
                        q_traj_vel_in_phase,
                        q_traj_acc_in_phase,
                        link_poses_th,
                        jacs_spatial_th,
                        link_poses_th_ee,
                        jacs_spatial_th_ee,
                    )

                if self.dataset.context_ee_goal_pose and cost_key not in self.EE_GOAL_COST_KEYS:
                    # If the EE pose goal context is set, the generative model determines the last joint position.
                    # Hence, we zero the gradient of the last control point for all costs except the EE pose goal cost,
                    # to avoid changing the last joint position.
                    # Note that the penultimate control point of the B-spline still affects the last portion of
                    # the trajectory, which means it can remove it from collisions, even though the last control point
                    # is fixed.
                    grad_cost_single_wrt_cp_normalized_in_phase[..., -1, :] = 0.0

                if self.dataset.context_ee_goal_pose and cost_key in self.EE_GOAL_COST_KEYS:
                    # EE goal costs are defined only on the last point of the trajectory,
                    # so we use directly the cost and gradient at the last point, without integration.
                    cost_single = cost_single_in_phase[..., -1]
                    grad_cost_single_wrt_cp_normalized = grad_cost_single_wrt_cp_normalized_in_phase[:, -1, ...]
                else:
                    # Approximate integral in eq. 28 -- https://arxiv.org/pdf/2412.19948
                    cost_single = torch.trapezoid(
                        cost_single_in_phase * rs_inv,
                        s,
                        dim=-1,
                    )

                    grad_cost_single_wrt_cp_normalized = torch.trapezoid(
                        grad_cost_single_wrt_cp_normalized_in_phase * rs_inv[None, :, None, None],
                        s,
                        dim=-3,
                    )

                cost_single_weighted = weight * cost_single
                cost_all += cost_single_weighted
                grad_cost_single_wrt_cp_normalized_weighted = weight * grad_cost_single_wrt_cp_normalized
                grad_costs_wrt_cp_normalized_l.append(grad_cost_single_wrt_cp_normalized_weighted)

                if self.debug:
                    print(f"{cost_key} (cost): {cost_single_weighted.mean():.4f} +- {cost_single_weighted.std():.4f}")
                    grad_cost_all_wrt_cp_normalized_norm_weighted = torch.linalg.norm(
                        grad_cost_single_wrt_cp_normalized_weighted, dim=-1
                    )
                    print(
                        f"{cost_key} (grad norm):"
                        f" {grad_cost_all_wrt_cp_normalized_norm_weighted.mean():.4f}"
                        f" +- {grad_cost_all_wrt_cp_normalized_norm_weighted.std():.4f}"
                    )
                    print(f"{cost_key} (time): {time.perf_counter() - s_time:.4f} s")
                    print(f"--------------------------------")

        if self.debug:
            print(f"Costs and gradients (time): {t_cost_grad_all.elapsed:.4f} s")

        # Project gradients respecting hierarchy
        if self.args_inference.project_gradient_hierarchy:
            with TimerCUDA() as t_project_gradients:
                if grad_costs_wrt_cp_normalized_l:
                    grad_costs_all_wrt_cp_normalized, grad_costs_all_wrt_cp_normalized_projected_l = (
                        project_hierarchical_gradients_fast(grad_costs_wrt_cp_normalized_l)
                    )
                else:
                    grad_costs_all_wrt_cp_normalized = torch.zeros_like(control_points_normalized)
                    grad_costs_all_wrt_cp_normalized_projected_l = []
            if self.debug:
                print(f"Project gradients (time): {t_project_gradients.elapsed:.4f} s")
        else:
            with self.guidance_profiler.section("gradient_aggregation_projection"):
                grad_costs_all_wrt_cp_normalized = (
                    torch.stack(grad_costs_wrt_cp_normalized_l).sum(dim=0)
                    if grad_costs_wrt_cp_normalized_l
                    else torch.zeros_like(control_points_normalized)
                )

        # -1 because the denoising gradient methods expect an objective function to maximize, but we want to minimize
        # the cost
        grad_costs_all_wrt_cp_normalized = -1.0 * grad_costs_all_wrt_cp_normalized

        # scatter plot of gradients in 2D
        if plot_gradients and self.debug and control_points_normalized.shape[-1] == 2:
            import matplotlib.pyplot as plt

            fig, ax = create_fig_and_axes(self.planning_task.env.dim)
            ax.scatter(to_numpy(control_points)[..., 0], to_numpy(control_points[..., 1]))
            self.planning_task.env.render(ax)
            self.planning_task.robot.render_trajectories(
                ax, q_traj_pos_in_phase, plot_points_scatter=False, control_points=control_points
            )

            grad_costs_wrt_cp_normalized_l += [-1.0 * grad_costs_all_wrt_cp_normalized]
            colors = ["g", "y", "c", "m", "k"]
            for k, grad in enumerate(grad_costs_wrt_cp_normalized_l):
                grad *= -1  # flip the gradient direction
                ax.quiver(
                    to_numpy(control_points[..., 0]),
                    to_numpy(control_points[..., 1]),
                    to_numpy(grad[..., 0]),
                    to_numpy(grad[..., 1]),
                    color=colors[k % len(colors)] if k < len(grad_costs_wrt_cp_normalized_l) - 1 else "blue",
                    label=f"{list(self.costs.keys())[k]}" if k < len(grad_costs_wrt_cp_normalized_l) - 1 else "Total",
                    scale=25,
                    width=0.005,
                )
            ax.legend()
            plt.show()

        # Increment step counter
        if not warmup:
            self.step_guide_call += 1
        horizon = q_traj_pos_in_phase.shape[1]
        n_spheres = len(self.robot.link_collision_spheres_names)
        n_self_pairs = len(self.robot.link_self_collision_tuples)
        self.guidance_profiler.update(
            n_time_points_total=batch_size * horizon,
            n_time_points_active=batch_size * horizon,
            n_spheres_total=batch_size * horizon * n_spheres,
            n_spheres_active=batch_size * horizon * n_spheres,
            n_self_pairs_total=batch_size * horizon * n_self_pairs,
            n_self_pairs_active=batch_size * horizon * n_self_pairs,
            fused_bspline_integration=False,
        )
        self.guidance_profiler.end_call(warmup=warmup)
        if return_cost:
            return cost_all, grad_costs_all_wrt_cp_normalized
        return grad_costs_all_wrt_cp_normalized

    @torch.no_grad()
    def _pruned_call(
        self,
        control_points_normalized,
        return_cost=False,
        warmup=False,
        plot_gradients=False,
        cost_weight_overrides=None,
        **kwargs,
    ):
        if self.debug:
            print()
            print(f"Pruned guide step {self.step_guide_call}")

        batch_size = control_points_normalized.shape[0]
        diffusion_timestep = kwargs.get("diffusion_timestep")
        if torch.is_tensor(diffusion_timestep):
            diffusion_timestep = int(diffusion_timestep.flatten()[0].detach().cpu().item())
        guide_iteration = kwargs.get("guide_iteration")
        if torch.is_tensor(guide_iteration):
            guide_iteration = int(guide_iteration.flatten()[0].detach().cpu().item())
        elif guide_iteration is not None:
            guide_iteration = int(guide_iteration)
        self.guidance_profiler.begin_call(
            guide_call_index=self.step_guide_call,
            context_index=self.profile_context_index,
            diffusion_timestep=diffusion_timestep,
            n_candidates=batch_size,
        )

        with self.guidance_profiler.section("bspline_expansion"):
            control_points = self.dataset.unnormalize_control_points(control_points_normalized)
            trajectory = self.parametric_trajectory.get_q_trajectory(
                control_points,
                None,
                None,
                get_type=("pos", "vel", "acc"),
                get_time_representation=False,
            )
            q_dense = trajectory["pos"]
            q_velocity_dense = trajectory["vel"]
            q_acceleration_dense = trajectory["acc"]

        horizon = q_dense.shape[1]
        collision_object_cost = self.costs.get("CostTaskSpaceCollisionObjects")
        environment_field = (
            collision_object_cost.cost.collision_objects_field if collision_object_cost is not None else None
        )
        self_collision_cost = self.costs.get("CostTaskSpaceCollisionSelf")
        self_field = self_collision_cost.cost.collision_self_field if self_collision_cost is not None else None

        force_all_active = bool(self.gradient_pruning_config["force_all_active"])
        temporal_enabled = bool(self.gradient_pruning_config["temporal"]["enabled"])
        conditional_temporal_enabled = bool(
            self.gradient_pruning_config["temporal"].get("conditional_enabled", False)
        )
        candidate_enabled = bool(self.gradient_pruning_config["candidate"]["enabled"])
        use_sparse_selection = (
            temporal_enabled
            or conditional_temporal_enabled
            or candidate_enabled
            or self.use_link_broad_phase
            or self.use_span_certificate
        ) and not force_all_active
        use_dense_full_parent_fast = self.use_dense_parent_fast_path and not use_sparse_selection
        temporal_selection_cache_hit = False
        if use_dense_full_parent_fast:
            # A2P-fast deliberately bypasses TemporalSelection. Besides avoiding
            # the collision scan, this keeps the full [B, H, D] layout intact
            # all the way through parent-link J-FK and collision integration.
            selection = None
        elif not use_sparse_selection:
            all_indices = torch.arange(horizon, device=q_dense.device)
            selection = TemporalSelection(
                active_indices=[all_indices for _ in range(batch_size)],
                bucket_sizes=torch.full((batch_size,), horizon, dtype=torch.long, device=q_dense.device),
                risk_mask=torch.ones((batch_size, horizon), dtype=torch.bool, device=q_dense.device),
                environment_clearance=torch.full(
                    (batch_size, horizon), torch.nan, dtype=q_dense.dtype, device=q_dense.device
                ),
                self_clearance=torch.full(
                    (batch_size, horizon), torch.nan, dtype=q_dense.dtype, device=q_dense.device
                ),
            )
        else:
            with self.guidance_profiler.section("collision_fk_and_risk_selection"):
                cache_enabled = bool(
                    self.gradient_pruning_config["temporal"].get(
                        "reuse_selection_within_ddim_step", False
                    )
                ) and not self.use_span_certificate
                cache_key = (
                    self.profile_context_index,
                    diffusion_timestep,
                    batch_size,
                    horizon,
                    str(q_dense.device),
                    str(q_dense.dtype),
                )
                can_reuse = (
                    cache_enabled
                    and diffusion_timestep is not None
                    and guide_iteration is not None
                    and guide_iteration > 0
                    and self._temporal_selection_cache is not None
                    and self._temporal_selection_cache["key"] == cache_key
                )
                if can_reuse:
                    selection = self._temporal_selection_cache["selection"]
                    temporal_selection_cache_hit = True
                else:
                    if self.use_span_certificate:
                        selection = self.collision_risk_selector.select_span_certificate(
                            q_dense.detach(),
                            control_points.detach(),
                            self.parametric_trajectory,
                            environment_field=environment_field,
                            self_field=self_field,
                        )
                    else:
                        selection = self.collision_risk_selector.select(
                            q_dense.detach(),
                            environment_field=environment_field,
                            self_field=self_field,
                        )
                    if cache_enabled and diffusion_timestep is not None:
                        self._temporal_selection_cache = {
                            "key": cache_key,
                            "selection": selection,
                        }

        dense_collision_batch = None
        link_broad_phase_scan_cache_reused = False
        with self.guidance_profiler.section("collision_jacobian"):
            if self.use_link_broad_phase and selection is not None:
                reuse_scan_cache = (
                    not temporal_selection_cache_hit
                    and selection.fine_sphere_scan_cache is not None
                )
                active_buckets = self.active_jacobian_computer.compute_selection_link_broad_phase(
                    q_dense,
                    selection,
                    # A selection cached from an earlier guide iteration has
                    # valid masks but stale geometry.  Reuse scan poses/SDF
                    # values only in the call that produced them.
                    reuse_scan_cache=reuse_scan_cache,
                )
                link_broad_phase_scan_cache_reused = reuse_scan_cache
                if reuse_scan_cache:
                    # Buckets now own only their selected poses/SDF values and
                    # Jacobians. Drop the dense cache immediately; later guide
                    # iterations keep the masks but must not see stale geometry.
                    selection.fine_sphere_scan_cache = None
            elif use_dense_full_parent_fast:
                dense_collision_batch = self.active_jacobian_computer.compute_dense(q_dense)
                active_buckets = []
            else:
                if self.use_dense_parent_fast_path:
                    # Temporal is layered on top of the same parent-link
                    # infrastructure. Full-horizon candidates use the dense
                    # fast path; only K=0/32/64 candidates remain sparse.
                    conditional_dense_fallback = (
                        conditional_temporal_enabled
                        and not selection.temporal_sparse_applied
                        and not candidate_enabled
                    )
                    if conditional_dense_fallback:
                        dense_collision_batch = self.active_jacobian_computer.compute_dense(q_dense)
                    else:
                        dense_candidate_indices = torch.nonzero(
                            selection.bucket_sizes == horizon,
                            as_tuple=False,
                        ).flatten()
                        dense_collision_batch = self.active_jacobian_computer.compute_dense(
                            q_dense,
                            dense_candidate_indices,
                        )
                active_buckets = self.active_jacobian_computer.compute_selection(
                    q_dense,
                    selection,
                    exclude_full_horizon=self.use_dense_parent_fast_path,
                )

        endpoint_only = bool(self.gradient_pruning_config["endpoint"]["ee_only_last_point"])
        with self.guidance_profiler.section("ee_fk_jacobian"):
            q_ee = q_dense[:, -1] if endpoint_only else q_dense.reshape(-1, q_dense.shape[-1])
            jacobians_ee, poses_ee = self.robot.jfk_s_ee(q_ee)
            jacobians_ee = torch.stack(jacobians_ee).transpose(0, 1)
            poses_ee = torch.stack(poses_ee).transpose(0, 1)
            if endpoint_only:
                jacobians_ee = jacobians_ee[:, None]
                poses_ee = poses_ee[:, None]
            else:
                jacobians_ee = jacobians_ee.reshape(batch_size, horizon, *jacobians_ee.shape[1:])
                poses_ee = poses_ee.reshape(batch_size, horizon, *poses_ee.shape[1:])

        grad_cp_wrt_cp_normalized = self.grad_cps_wrt_cps_normalized(control_points_normalized)
        phase = self.parametric_trajectory.phase_time.s
        rs_inv = self.parametric_trajectory.phase_time.rs_inv
        integrated_mapping_fn = None
        if self.use_sparse_bspline_support:
            integrated_mapping_fn = self.compute_cost_grad_cp_normalized_sparse_support
        elif self.use_fused_bspline_integration:
            integrated_mapping_fn = self.compute_cost_grad_cp_normalized_fused
        collision_keys = {"CostTaskSpaceCollisionObjects", "CostTaskSpaceCollisionSelf"}
        grad_costs = []
        cost_all = torch.zeros(batch_size, dtype=q_dense.dtype, device=q_dense.device)

        for cost_key in self.costs:
            cost_fn = self.costs[cost_key].cost
            weight = self.costs[cost_key].weight
            if cost_weight_overrides and cost_key in cost_weight_overrides:
                weight = cost_weight_overrides[cost_key]
            if _is_static_zero_weight(weight):
                continue

            if cost_key in collision_keys:
                cost_single = torch.zeros(batch_size, dtype=q_dense.dtype, device=q_dense.device)
                grad_single = torch.zeros_like(control_points_normalized)
                profile_name = "environment_sdf_query" if cost_key == "CostTaskSpaceCollisionObjects" else "self_collision"
                with self.guidance_profiler.section(profile_name):
                    if dense_collision_batch is not None:
                        dense_candidates = dense_collision_batch.candidate_indices
                        if dense_collision_batch.covers_full_batch:
                            dense_control_points_normalized = control_points_normalized
                            dense_control_points = control_points
                            dense_velocity = q_velocity_dense
                            dense_acceleration = q_acceleration_dense
                            dense_cp_scale = grad_cp_wrt_cp_normalized
                        else:
                            dense_control_points_normalized = control_points_normalized.index_select(
                                0, dense_candidates
                            )
                            dense_control_points = control_points.index_select(0, dense_candidates)
                            dense_velocity = q_velocity_dense.index_select(0, dense_candidates)
                            dense_acceleration = q_acceleration_dense.index_select(0, dense_candidates)
                            dense_cp_scale = grad_cp_wrt_cp_normalized.index_select(0, dense_candidates)

                        if integrated_mapping_fn is not None:
                            dense_cost_phase, dense_grad = integrated_mapping_fn(
                                cost_fn,
                                dense_control_points_normalized,
                                dense_control_points,
                                dense_collision_batch.q_dense,
                                dense_velocity,
                                dense_acceleration,
                                dense_collision_batch.poses,
                                dense_collision_batch.jacobians,
                                None,
                                None,
                                phase,
                                rs_inv,
                                grad_cp_wrt_cp_normalized=dense_cp_scale,
                                active_link_pruning=self.use_active_link_pruning,
                            )
                        else:
                            dense_cost_phase, dense_grad_phase = self.compute_cost_grad_cp_normalized(
                                cost_fn,
                                dense_control_points_normalized,
                                dense_control_points,
                                dense_collision_batch.q_dense,
                                dense_velocity,
                                dense_acceleration,
                                dense_collision_batch.poses,
                                dense_collision_batch.jacobians,
                                None,
                                None,
                                grad_cp_wrt_cp_normalized=dense_cp_scale,
                                active_link_pruning=self.use_active_link_pruning,
                            )
                            dense_grad = torch.trapezoid(
                                dense_grad_phase * rs_inv[None, :, None, None],
                                phase,
                                dim=-3,
                            )
                        dense_cost = torch.trapezoid(
                            dense_cost_phase * rs_inv,
                            phase,
                            dim=-1,
                        )
                        cost_single[dense_candidates] = dense_cost
                        grad_single[dense_candidates] = dense_grad

                    for bucket in active_buckets:
                        candidate_indices = bucket.candidate_indices
                        bucket_collision_kwargs = {
                            "phase_indices": bucket.phase_indices,
                            "grad_cp_wrt_cp_normalized": grad_cp_wrt_cp_normalized[
                                candidate_indices
                            ],
                            "active_link_pruning": self.use_active_link_pruning,
                            "link_indices": bucket.sphere_indices,
                            "self_pair_indices": bucket.self_pair_indices,
                        }
                        if (
                            cost_key == "CostTaskSpaceCollisionObjects"
                            and bucket.environment_sdf_values is not None
                        ):
                            bucket_collision_kwargs["precomputed_sdf_values"] = (
                                bucket.environment_sdf_values
                            )
                        if integrated_mapping_fn is not None:
                            cost_phase, bucket_grad = integrated_mapping_fn(
                                cost_fn,
                                control_points_normalized[candidate_indices],
                                control_points[candidate_indices],
                                bucket.q_active,
                                q_velocity_dense[candidate_indices[:, None], bucket.phase_indices],
                                q_acceleration_dense[candidate_indices[:, None], bucket.phase_indices],
                                bucket.poses,
                                bucket.jacobians,
                                None,
                                None,
                                phase,
                                rs_inv,
                                **bucket_collision_kwargs,
                            )
                        else:
                            cost_phase, grad_phase = self.compute_cost_grad_cp_normalized(
                                cost_fn,
                                control_points_normalized[candidate_indices],
                                control_points[candidate_indices],
                                bucket.q_active,
                                q_velocity_dense[candidate_indices[:, None], bucket.phase_indices],
                                q_acceleration_dense[candidate_indices[:, None], bucket.phase_indices],
                                bucket.poses,
                                bucket.jacobians,
                                None,
                                None,
                                **bucket_collision_kwargs,
                            )
                        selected_phase = phase[bucket.phase_indices]
                        selected_rs_inv = rs_inv[bucket.phase_indices]
                        bucket_cost = torch.trapezoid(cost_phase * selected_rs_inv, selected_phase, dim=-1)
                        if integrated_mapping_fn is None:
                            bucket_grad = torch.trapezoid(
                                grad_phase * selected_rs_inv[..., None, None],
                                selected_phase[..., None, None],
                                dim=-3,
                            )
                        cost_single[candidate_indices] = bucket_cost
                        grad_single[candidate_indices] = bucket_grad
            elif cost_key in self.EE_GOAL_COST_KEYS and self.dataset.context_ee_goal_pose:
                endpoint_phase_indices = torch.tensor([horizon - 1], device=q_dense.device)
                endpoint_poses = poses_ee if endpoint_only else poses_ee[:, -1:]
                endpoint_jacobians = jacobians_ee if endpoint_only else jacobians_ee[:, -1:]
                compute_endpoint = integrated_mapping_fn or self.compute_cost_grad_cp_normalized
                endpoint_kwargs = {
                    "phase_indices": endpoint_phase_indices,
                    "grad_cp_wrt_cp_normalized": grad_cp_wrt_cp_normalized,
                }
                if integrated_mapping_fn is not None:
                    endpoint_kwargs.update(
                        {"phase": phase, "rs_inv": rs_inv, "integrate": False}
                    )
                cost_phase, endpoint_grad = compute_endpoint(
                    cost_fn,
                    control_points_normalized,
                    control_points,
                    q_dense[:, -1:],
                    q_velocity_dense[:, -1:],
                    q_acceleration_dense[:, -1:],
                    None,
                    None,
                    endpoint_poses,
                    endpoint_jacobians,
                    **endpoint_kwargs,
                )
                cost_single = cost_phase[..., -1]
                grad_single = (
                    endpoint_grad
                    if integrated_mapping_fn is not None
                    else endpoint_grad[:, -1]
                )
            else:
                if integrated_mapping_fn is not None:
                    cost_phase, grad_single = integrated_mapping_fn(
                        cost_fn,
                        control_points_normalized,
                        control_points,
                        q_dense,
                        q_velocity_dense,
                        q_acceleration_dense,
                        None,
                        None,
                        poses_ee,
                        jacobians_ee,
                        phase,
                        rs_inv,
                        grad_cp_wrt_cp_normalized=grad_cp_wrt_cp_normalized,
                    )
                    cost_single = torch.trapezoid(cost_phase * rs_inv, phase, dim=-1)
                else:
                    cost_phase, grad_phase = self.compute_cost_grad_cp_normalized(
                        cost_fn,
                        control_points_normalized,
                        control_points,
                        q_dense,
                        q_velocity_dense,
                        q_acceleration_dense,
                        None,
                        None,
                        poses_ee,
                        jacobians_ee,
                        grad_cp_wrt_cp_normalized=grad_cp_wrt_cp_normalized,
                    )
                    cost_single = torch.trapezoid(cost_phase * rs_inv, phase, dim=-1)
                    grad_single = torch.trapezoid(
                        grad_phase * rs_inv[None, :, None, None],
                        phase,
                        dim=-3,
                    )

            if self.dataset.context_ee_goal_pose and cost_key not in self.EE_GOAL_COST_KEYS:
                grad_single[..., -1, :] = 0.0
            elif cost_key in self.EE_GOAL_COST_KEYS:
                grad_single[..., :-1, :] = 0.0

            cost_all = cost_all + weight * cost_single
            grad_costs.append(weight * grad_single)

        with self.guidance_profiler.section("gradient_aggregation_projection"):
            if not grad_costs:
                grad_all = torch.zeros_like(control_points_normalized)
            elif self.args_inference.project_gradient_hierarchy:
                grad_all, _ = project_hierarchical_gradients_fast(grad_costs)
            else:
                grad_all = torch.stack(grad_costs).sum(dim=0)
            grad_all = -grad_all

        n_spheres = len(self.robot.link_collision_spheres_names)
        n_self_pairs = len(self.robot.link_self_collision_tuples)
        broad_sphere_points = None
        broad_self_pair_points = None
        if self.use_link_broad_phase:
            broad_sphere_points = sum(
                int(bucket.candidate_indices.numel())
                * int(bucket.phase_indices.shape[1])
                * int(bucket.sphere_indices.numel())
                for bucket in active_buckets
            )
            broad_self_pair_points = sum(
                int(bucket.candidate_indices.numel())
                * int(bucket.phase_indices.shape[1])
                * int(bucket.self_pair_indices.numel())
                for bucket in active_buckets
            )
        if selection is None:
            # Dense ParentLinkFast has no selector result by design.
            n_active = batch_size * horizon
            min_environment_clearance = float("inf")
            min_self_clearance = float("inf")
            bucket_0 = bucket_32 = bucket_64 = 0
            bucket_full = batch_size
        else:
            environment_statistics = torch.where(
                torch.isnan(selection.environment_clearance),
                torch.full_like(selection.environment_clearance, torch.inf),
                selection.environment_clearance,
            )
            self_statistics = torch.where(
                torch.isnan(selection.self_clearance),
                torch.full_like(selection.self_clearance, torch.inf),
                selection.self_clearance,
            )
            n_active = int(selection.active_counts.sum().item())
            min_environment_clearance = float(environment_statistics.amin().detach().cpu().item())
            min_self_clearance = float(self_statistics.amin().detach().cpu().item())
            bucket_0 = int((selection.bucket_sizes == 0).sum().item())
            bucket_32 = int((selection.bucket_sizes == 32).sum().item())
            bucket_64 = int((selection.bucket_sizes == 64).sum().item())
            bucket_full = int((selection.bucket_sizes == horizon).sum().item())
        self.guidance_profiler.update(
            n_time_points_total=batch_size * horizon,
            n_time_points_active=n_active,
            n_spheres_total=batch_size * horizon * n_spheres,
            n_spheres_active=(
                broad_sphere_points if broad_sphere_points is not None else n_active * n_spheres
            ),
            n_self_pairs_total=batch_size * horizon * n_self_pairs,
            n_self_pairs_active=(
                broad_self_pair_points
                if broad_self_pair_points is not None
                else n_active * n_self_pairs
            ),
            min_environment_clearance=min_environment_clearance,
            min_self_clearance=min_self_clearance,
            bucket_0=bucket_0,
            bucket_32=bucket_32,
            bucket_64=bucket_64,
            bucket_128=bucket_full,
            parent_link_kinematics=self.use_parent_link_kinematics,
            dense_parent_fast_path=bool(dense_collision_batch is not None),
            fused_bspline_integration=self.use_fused_bspline_integration,
            sparse_bspline_support=self.use_sparse_bspline_support,
            active_link_pruning=self.use_active_link_pruning,
            link_broad_phase=self.use_link_broad_phase,
            link_broad_phase_scan_cache_reused=link_broad_phase_scan_cache_reused,
            link_broad_phase_scan_geometry=(
                self.gradient_pruning_config["spatial"]
                .get("link_broad_phase", {})
                .get("scan_geometry", "parent_bounds")
            ),
            parent_bounds_preselection=self.gradient_pruning_config["preselection"].get(
                "parent_bounds_scan", False
            ),
            conditional_temporal=conditional_temporal_enabled,
            conditional_temporal_applied=(
                bool(selection.temporal_sparse_applied) if selection is not None else False
            ),
            predicted_active_ratio=(
                float(selection.predicted_active_ratio) if selection is not None else 1.0
            ),
            temporal_selection_cache_hit=temporal_selection_cache_hit,
            guide_iteration=guide_iteration,
            span_certificate=self.use_span_certificate,
            **(
                selection.span_certificate_statistics
                if selection is not None and selection.span_certificate_statistics is not None
                else {}
            ),
        )
        self.guidance_profiler.end_call(warmup=warmup)

        if plot_gradients and self.debug:
            print("plot_gradients is not rendered by the pruned path.")
        if not warmup:
            self.step_guide_call += 1
        if return_cost:
            return cost_all, grad_all
        return grad_all

    def compute_cost_grad_cp_normalized(
        self,
        cost_fn,
        control_points_normalized,
        control_points,
        q_traj_pos_in_phase,
        q_traj_vel_in_phase,
        q_traj_acc_in_phase,
        link_poses_th,
        jacs_spatial_th,
        link_poses_th_ee,
        jacs_spatial_th_ee,
        grad_cp_wrt_cp_normalized=None,
        **kwargs,
    ):
        # compute cost gradients wrt to the control points normalized
        # The cost C can be a function of the trajectory q(s), the control points cp, or the task space x.
        # dC/dcp_norm = dC/dx * dx/dq * dq/dcp * dcp/dcp_normalized

        # We compute the gradient of the cost wrt to the joint space q
        # For TaskSpace costs, we compute dC/dq = dC/dx * dx/dq * dq/dcp
        # For JointSpace costs, we compute dC/dq = dC/dq * dq/dcp
        cost_value_in_phase, grad_cost_wrt_cp_in_phase = cost_fn.compute_cost_grad_wrt_cp(
            control_points,
            q_traj_pos_in_phase,
            q_traj_vel_in_phase,
            q_traj_acc_in_phase,
            link_poses_th,
            jacs_spatial_th,
            link_poses_th_ee,
            jacs_spatial_th_ee,
            **kwargs,
        )

        # Gradient of the control points wrt to the control points normalized
        # dcp/dcp_norm
        if grad_cp_wrt_cp_normalized is None:
            grad_cp_wrt_cp_normalized = self.grad_cps_wrt_cps_normalized(control_points_normalized)

        # Gradient of the cost wrt to the control points normalized
        # dC/dcp_norm = dC/dcp * dcp/dcp_norm
        # In matrix form -- Hadamard product (the normalization is done element-wise)
        grad_cost_wrt_cp_normalized_per_shape_step = torch.einsum(
            "...jkn,...kn->...jkn", grad_cost_wrt_cp_in_phase, grad_cp_wrt_cp_normalized
        )

        return cost_value_in_phase, grad_cost_wrt_cp_normalized_per_shape_step

    @staticmethod
    def _trapezoid_weights(phase):
        """Return weights equivalent to ``torch.trapezoid(values, phase)``."""

        weights = torch.zeros_like(phase)
        if phase.shape[-1] < 2:
            return weights
        half_delta = 0.5 * (phase[..., 1:] - phase[..., :-1])
        weights[..., :-1] += half_delta
        weights[..., 1:] += half_delta
        return weights

    def _bspline_basis_for_phases(self, derivative_type, phase_indices=None):
        basis = self.parametric_trajectory.get_grad_q_traj_in_phase_wrt_control_points(
            None,
            get_type=(derivative_type,),
            remove_control_points=True,
        )[derivative_type]
        while basis.ndim > 2 and basis.shape[0] == 1:
            basis = basis.squeeze(0)
        if basis.ndim != 2:
            raise ValueError(
                "Fused B-spline integration expects a two-dimensional [phase, control] basis."
            )
        if phase_indices is not None:
            basis = basis[phase_indices]
        return basis

    def _bspline_sparse_support_for_phases(self, derivative_type, phase_indices=None):
        """Return compact non-zero B-spline columns for each phase.

        A degree-p B-spline has at most p+1 active basis functions at any
        phase. Fixed boundary control points may remove columns, but can never
        increase that width. Keeping the fixed p+1 layout avoids ragged GPU
        metadata while replacing the full K-column contraction by a compact
        scatter-add.
        """

        cached = self._bspline_support_cache.get(derivative_type)
        if cached is None:
            basis = self._bspline_basis_for_phases(derivative_type)
            support_width = min(int(self.parametric_trajectory.bspline.d) + 1, basis.shape[-1])
            _, support_indices = torch.topk(
                basis.abs(),
                k=support_width,
                dim=-1,
                largest=True,
                sorted=False,
            )
            support_values = torch.gather(basis, -1, support_indices)
            cached = (support_indices, support_values)
            self._bspline_support_cache[derivative_type] = cached
        support_indices, support_values = cached
        if phase_indices is not None:
            support_indices = support_indices[phase_indices]
            support_values = support_values[phase_indices]
        return support_indices, support_values

    def compute_cost_grad_cp_normalized_fused(
        self,
        cost_fn,
        control_points_normalized,
        control_points,
        q_traj_pos_in_phase,
        q_traj_vel_in_phase,
        q_traj_acc_in_phase,
        link_poses_th,
        jacs_spatial_th,
        link_poses_th_ee,
        jacs_spatial_th_ee,
        phase,
        rs_inv,
        integrate=True,
        grad_cp_wrt_cp_normalized=None,
        **kwargs,
    ):
        """Map ``dC/dq`` directly to integrated control-point gradients.

        This is algebraically equivalent to materializing [B, H, K, D] and
        then applying ``torch.trapezoid``, but keeps only the [B, K, D]
        result. The original implementation remains available through the
        configuration switch.
        """

        cost_value_in_phase, gradients_wrt_q = cost_fn.compute_cost_grad_wrt_q(
            control_points,
            q_traj_pos_in_phase,
            q_traj_vel_in_phase,
            q_traj_acc_in_phase,
            link_poses_th,
            jacs_spatial_th,
            link_poses_th_ee,
            jacs_spatial_th_ee,
            **kwargs,
        )
        if grad_cp_wrt_cp_normalized is None:
            grad_cp_wrt_cp_normalized = self.grad_cps_wrt_cps_normalized(
                control_points_normalized
            )

        phase_indices = kwargs.get("phase_indices")
        selected_phase = phase if phase_indices is None else phase[phase_indices]
        selected_rs_inv = rs_inv if phase_indices is None else rs_inv[phase_indices]
        integration_weights = (
            self._trapezoid_weights(selected_phase) * selected_rs_inv
            if integrate
            else None
        )
        grad_cost_wrt_cp = torch.zeros_like(control_points_normalized)
        for derivative_type, grad_cost_wrt_q in gradients_wrt_q.items():
            basis = self._bspline_basis_for_phases(derivative_type, phase_indices)
            if basis.ndim == 2:
                if integrate:
                    mapped = torch.einsum(
                        "hk,bhd,h->bkd",
                        basis,
                        grad_cost_wrt_q,
                        integration_weights,
                    )
                else:
                    mapped = torch.einsum("hk,bhd->bkd", basis, grad_cost_wrt_q)
            elif basis.ndim == 3:
                if integrate:
                    mapped = torch.einsum(
                        "bhk,bhd,bh->bkd",
                        basis,
                        grad_cost_wrt_q,
                        integration_weights,
                    )
                else:
                    mapped = torch.einsum("bhk,bhd->bkd", basis, grad_cost_wrt_q)
            else:
                raise ValueError("Selected B-spline basis must be [H,K] or [B,H,K].")
            grad_cost_wrt_cp = grad_cost_wrt_cp + mapped

        return cost_value_in_phase, grad_cost_wrt_cp * grad_cp_wrt_cp_normalized

    def compute_cost_grad_cp_normalized_sparse_support(
        self,
        cost_fn,
        control_points_normalized,
        control_points,
        q_traj_pos_in_phase,
        q_traj_vel_in_phase,
        q_traj_acc_in_phase,
        link_poses_th,
        jacs_spatial_th,
        link_poses_th_ee,
        jacs_spatial_th_ee,
        phase,
        rs_inv,
        integrate=True,
        grad_cp_wrt_cp_normalized=None,
        **kwargs,
    ):
        """Map dC/dq through only the local B-spline support and scatter to K."""

        cost_value_in_phase, gradients_wrt_q = cost_fn.compute_cost_grad_wrt_q(
            control_points,
            q_traj_pos_in_phase,
            q_traj_vel_in_phase,
            q_traj_acc_in_phase,
            link_poses_th,
            jacs_spatial_th,
            link_poses_th_ee,
            jacs_spatial_th_ee,
            **kwargs,
        )
        if grad_cp_wrt_cp_normalized is None:
            grad_cp_wrt_cp_normalized = self.grad_cps_wrt_cps_normalized(
                control_points_normalized
            )

        phase_indices = kwargs.get("phase_indices")
        selected_phase = phase if phase_indices is None else phase[phase_indices]
        selected_rs_inv = rs_inv if phase_indices is None else rs_inv[phase_indices]
        integration_weights = (
            self._trapezoid_weights(selected_phase) * selected_rs_inv
            if integrate
            else None
        )
        batch_size, n_control_points, dof = control_points_normalized.shape
        grad_cost_wrt_cp = torch.zeros_like(control_points_normalized)
        for derivative_type, grad_cost_wrt_q in gradients_wrt_q.items():
            support_indices, support_values = self._bspline_sparse_support_for_phases(
                derivative_type,
                phase_indices,
            )
            if support_indices.ndim == 2:
                support_indices = support_indices.unsqueeze(0).expand(batch_size, -1, -1)
                support_values = support_values.unsqueeze(0).expand(batch_size, -1, -1)
            elif support_indices.ndim != 3:
                raise ValueError("Sparse B-spline support must be [H,S] or [B,H,S].")

            weighted_grad = grad_cost_wrt_q
            if integrate:
                weights = integration_weights
                if weights.ndim == 1:
                    weights = weights.unsqueeze(0)
                weighted_grad = weighted_grad * weights[..., None]
            contributions = support_values[..., None] * weighted_grad[..., None, :]
            scatter_indices = support_indices.reshape(batch_size, -1, 1).expand(-1, -1, dof)
            grad_cost_wrt_cp.scatter_add_(
                1,
                scatter_indices,
                contributions.reshape(batch_size, -1, dof),
            )

        return cost_value_in_phase, grad_cost_wrt_cp * grad_cp_wrt_cp_normalized

    def warmup(self, shape_x, **kwargs):
        x = torch.randn(shape_x, **self.tensor_args)
        self.__call__(x, warmup=True)


class CostSpace:

    def __init__(self, planning_task, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        self.planning_task = planning_task
        self.parametric_trajectory = planning_task.parametric_trajectory
        self.robot = planning_task.robot
        self.env = planning_task.env
        self.tensor_args = tensor_args

    def compute_cost_grad_wrt_q(self, *args, **kwargs):
        raise NotImplementedError

    def compute_cost_grad_wrt_cp(self, control_points, *args, **kwargs):
        """Compatibility mapping used by Legacy and by the fused-path fallback."""

        cost, gradients_wrt_q = self.compute_cost_grad_wrt_q(
            control_points,
            *args,
            **kwargs,
        )
        grad_cost_wrt_cp = 0.0
        for derivative_type, grad_cost_wrt_q in gradients_wrt_q.items():
            grad_cost_wrt_cp = grad_cost_wrt_cp + self.compute_grad_cost_wrt_cp(
                control_points,
                grad_cost_wrt_q,
                get_type_single=derivative_type,
                phase_indices=kwargs.get("phase_indices"),
            )
        return cost, grad_cost_wrt_cp

    def compute_grad_cost_wrt_cp(
        self,
        control_points,
        grad_cost_wrt_q,
        get_type_single="pos",
        phase_indices=None,
    ):
        # Gradient of the joint space trajectory position wrt to the control points
        # dq/dcp
        grad_q_pos_wrt_cp = self.parametric_trajectory.get_grad_q_traj_in_phase_wrt_control_points(
            control_points,
            get_type=(get_type_single,),
            remove_control_points=True,
        )[get_type_single]
        if phase_indices is not None:
            phase_indices = torch.as_tensor(
                phase_indices,
                dtype=torch.long,
                device=grad_q_pos_wrt_cp.device,
            )
            if phase_indices.ndim == 1:
                grad_q_pos_wrt_cp = grad_q_pos_wrt_cp[..., phase_indices, :]
            elif phase_indices.ndim == 2:
                grad_q_pos_wrt_cp = grad_q_pos_wrt_cp.unsqueeze(0).expand(
                    phase_indices.shape[0], -1, -1, -1
                )
                gather_indices = phase_indices[:, None, :, None].expand(
                    -1,
                    grad_q_pos_wrt_cp.shape[1],
                    -1,
                    grad_q_pos_wrt_cp.shape[-1],
                )
                grad_q_pos_wrt_cp = torch.gather(
                    grad_q_pos_wrt_cp,
                    dim=-2,
                    index=gather_indices,
                )
            else:
                raise ValueError("phase_indices must be one- or two-dimensional.")
        # Gradient of cost wrt to the control points per phase step
        # dC/dcp = dC/dq * dq/dcp
        # In matrix form dC/dcp = (dq/dcp)^T @ dC/dq
        try:
            grad_cost_wrt_cp = torch.einsum("...ihk,...hn->...hkn", grad_q_pos_wrt_cp, grad_cost_wrt_q)
        except RuntimeError:
            # for waypoints, we sum over the state dimension
            grad_cost_wrt_cp = torch.einsum("...hdkn,...hn->...hkn", grad_q_pos_wrt_cp, grad_cost_wrt_q)

        return grad_cost_wrt_cp


class CostJointSpace(CostSpace):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CostJointSpaceJointLimits(CostJointSpace):

    def __init__(self, planning_task, eps=0.03, **kwargs):
        super().__init__(planning_task, **kwargs)
        self.eps = eps
        self.q_min = self.robot.q_pos_min
        self.q_max = self.robot.q_pos_max
        if self.robot.dq_max is not None:
            self.dq_min = -self.robot.dq_max
            self.dq_max = self.robot.dq_max
        else:
            self.dq_min = None
            self.dq_max = None
        if self.robot.ddq_max is not None:
            self.ddq_min = -self.robot.ddq_max
            self.ddq_max = self.robot.ddq_max
        else:
            self.ddq_min = None
            self.ddq_max = None

    def compute_cost_grad_wrt_q(
        self, control_points, q_traj_pos_in_phase, q_traj_vel_in_phase, q_traj_acc_in_phase, *args, **kwargs
    ):
        # positions
        # transform the joint position limits from time to phase
        q_min_in_phase = self.q_min + self.eps
        q_max_in_phase = self.q_max - self.eps
        mask_low = torch.less_equal(q_traj_pos_in_phase, q_min_in_phase)
        mask_high = torch.greater_equal(q_traj_pos_in_phase, q_max_in_phase)

        cost_pos_limit_low = 0.5 * torch.sum((q_min_in_phase - q_traj_pos_in_phase).square() * mask_low, dim=-1)
        cost_pos_limit_high = 0.5 * torch.sum((q_max_in_phase - q_traj_pos_in_phase).square() * mask_high, dim=-1)
        cost_pos_limit = cost_pos_limit_low + cost_pos_limit_high
        grad_cost_q_pos_low = mask_low * (q_min_in_phase - q_traj_pos_in_phase) * -1
        grad_cost_q_pos_high = mask_high * (q_max_in_phase - q_traj_pos_in_phase) * -1
        grad_cost_wrt_q_pos = grad_cost_q_pos_low + grad_cost_q_pos_high

        gradients_wrt_q = {"pos": grad_cost_wrt_q_pos}

        # velocities
        rs_inv = self.parametric_trajectory.phase_time.rs_inv
        rs = self.parametric_trajectory.phase_time.rs
        dr_ds = self.parametric_trajectory.phase_time.dr_ds

        cost_vel_limit = 0.0
        if self.dq_max is not None:
            # transform the joint velocity limits from time to phase
            dq_min_in_phase = (self.dq_min + self.eps) * rs_inv[..., None]
            dq_max_in_phase = (self.dq_max - self.eps) * rs_inv[..., None]
            mask_low = torch.less_equal(q_traj_vel_in_phase, dq_min_in_phase)
            mask_high = torch.greater_equal(q_traj_vel_in_phase, dq_max_in_phase)

            cost_vel_limit_low = 0.5 * torch.sum((dq_min_in_phase - q_traj_vel_in_phase).square() * mask_low, dim=-1)
            cost_vel_limit_high = 0.5 * torch.sum((dq_max_in_phase - q_traj_vel_in_phase).square() * mask_high, dim=-1)
            cost_vel_limit = cost_vel_limit_low + cost_vel_limit_high
            grad_cost_q_vel_low = mask_low * (dq_min_in_phase - q_traj_vel_in_phase) * -1
            grad_cost_q_vel_high = mask_high * (dq_max_in_phase - q_traj_vel_in_phase) * -1
            grad_cost_wrt_q_vel = grad_cost_q_vel_low + grad_cost_q_vel_high

            gradients_wrt_q["vel"] = grad_cost_wrt_q_vel

        # accelerations
        cost_acc_limit = 0.0
        if self.ddq_max is not None:
            # transform the joint acceleration limits from time to phase
            ddq_min_in_phase = (
                self.ddq_min + self.eps - q_traj_vel_in_phase * dr_ds[..., None] * rs[..., None]
            ) * rs_inv[..., None] ** 2
            ddq_max_in_phase = (
                self.ddq_max - self.eps - q_traj_vel_in_phase * dr_ds[..., None] * rs[..., None]
            ) * rs_inv[..., None] ** 2
            mask_low = torch.less_equal(q_traj_acc_in_phase, ddq_min_in_phase)
            mask_high = torch.greater_equal(q_traj_acc_in_phase, ddq_max_in_phase)

            cost_acc_limit_low = 0.5 * torch.sum((ddq_min_in_phase - q_traj_acc_in_phase).square() * mask_low, dim=-1)
            cost_acc_limit_high = 0.5 * torch.sum((ddq_max_in_phase - q_traj_acc_in_phase).square() * mask_high, dim=-1)
            cost_acc_limit = cost_acc_limit_low + cost_acc_limit_high
            grad_cost_q_acc_low = mask_low * (ddq_min_in_phase - q_traj_acc_in_phase) * -1
            grad_cost_q_acc_high = mask_high * (ddq_max_in_phase - q_traj_acc_in_phase) * -1
            grad_cost_wrt_q_acc = grad_cost_q_acc_low + grad_cost_q_acc_high

            gradients_wrt_q["acc"] = grad_cost_wrt_q_acc

        return (
            cost_pos_limit + cost_vel_limit + cost_acc_limit,
            gradients_wrt_q,
        )


class CostJointSpacePathLength(CostJointSpace):

    def __init__(self, planning_task, **kwargs):
        super().__init__(planning_task, **kwargs)

    def compute_cost_grad_wrt_q(
        self, control_points, q_traj_pos_in_phase, q_traj_vel_in_phase, q_traj_acc_in_phase, *args, **kwargs
    ):
        q_traj_pos_diff = torch.zeros_like(q_traj_pos_in_phase)
        q_traj_pos_diff[..., 1:, :] = torch.diff(q_traj_pos_in_phase, dim=-2)
        segment_norm = torch.linalg.norm(q_traj_pos_diff, dim=-1)
        cost_pos = 0.5 * segment_norm
        segment_gradient = 0.5 * torch.where(
            segment_norm[..., None] > 0,
            q_traj_pos_diff / segment_norm.clamp_min(torch.finfo(q_traj_pos_diff.dtype).eps)[..., None],
            torch.zeros_like(q_traj_pos_diff),
        )
        grad_cost_wrt_q_pos = torch.zeros_like(q_traj_pos_in_phase)
        grad_cost_wrt_q_pos[..., 1:, :] += segment_gradient[..., 1:, :]
        grad_cost_wrt_q_pos[..., :-1, :] -= segment_gradient[..., 1:, :]

        return cost_pos, {"pos": grad_cost_wrt_q_pos}


class CostJointSpaceVelocity(CostJointSpace):

    def __init__(self, planning_task, **kwargs):
        super().__init__(planning_task, **kwargs)

    def compute_cost_grad_wrt_q(
        self, control_points, q_traj_pos_in_phase, q_traj_vel_in_phase, q_traj_acc_in_phase, *args, **kwargs
    ):
        cost_vel = 0.5 * torch.sum(
            q_traj_vel_in_phase.square(), dim=-1
        )  # 0.5 * torch.linalg.norm(q_traj_vel_in_phase, dim=-1)
        return cost_vel, {"vel": q_traj_vel_in_phase}


class CostJointSpaceAcceleration(CostJointSpace):

    def __init__(self, planning_task, **kwargs):
        super().__init__(planning_task, **kwargs)

    def compute_cost_grad_wrt_q(
        self, control_points, q_traj_pos_in_phase, q_traj_vel_in_phase, q_traj_acc_in_phase, *args, **kwargs
    ):
        cost_acc = 0.5 * torch.sum(q_traj_acc_in_phase.square(), dim=-1)
        return cost_acc, {"acc": q_traj_acc_in_phase}


class CostTaskSpace(CostSpace):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_jacobians_position(self, jac_spatial):
        return jac_spatial[..., : self.robot.task_space_dim, :]

    def get_jacobians_orientation(self, jac_spatial):
        # the starting index is 3, because the first 3 elements are the position
        return jac_spatial[..., 3:, :]


def map_jacobian_from_world_to_local_world_aligned(link_pose_w, jacobian_spatial):
    # map jacobian from world (spatial) to local world aligned using the Adjoint matrix
    H_lwa_w = link_pose_w.clone()
    H_lwa_w[..., :3, :3] = torch.eye(3, dtype=link_pose_w.dtype, device=link_pose_w.device)
    H_lwa_w[..., :3, 3] = -1 * H_lwa_w[..., :3, 3]
    Ad_lwa_w = _adjoint_impl(H_lwa_w)
    jac_lwa = Ad_lwa_w @ jacobian_spatial
    return jac_lwa


def project_collision_gradient_to_joints(
    link_poses,
    jacobians_spatial,
    gradient_cost_wrt_x,
    task_space_dim,
    active_link_pruning=False,
):
    """Apply J^T g densely or as a packed exact active-link projection.

    Collision fields already return an exactly zero task-space gradient for
    links outside their support. The sparse path packs only non-zero entries,
    evaluates the adjoint/Jacobian projection for those entries, and sums them
    back into the original batch/phase rows. FK and SDF semantics are unchanged.
    """

    if not active_link_pruning:
        jacs_lwa = map_jacobian_from_world_to_local_world_aligned(
            link_poses,
            jacobians_spatial,
        )
        return torch.einsum(
            "...dj,...d->...j",
            jacs_lwa[..., :task_space_dim, :],
            gradient_cost_wrt_x,
        ).sum(dim=-2)

    prefix_shape = gradient_cost_wrt_x.shape[:-2]
    n_links = gradient_cost_wrt_x.shape[-2]
    dof = jacobians_spatial.shape[-1]
    gradient_flat = gradient_cost_wrt_x.reshape(-1, n_links, task_space_dim)
    poses_flat = link_poses.reshape(-1, n_links, *link_poses.shape[-2:])
    jacobians_flat = jacobians_spatial.reshape(
        -1,
        n_links,
        *jacobians_spatial.shape[-2:],
    )
    active_pairs = torch.nonzero(gradient_flat.ne(0).any(dim=-1), as_tuple=False)
    active_poses = poses_flat[active_pairs[:, 0], active_pairs[:, 1]]
    active_jacobians = jacobians_flat[active_pairs[:, 0], active_pairs[:, 1]]
    active_gradients = gradient_flat[active_pairs[:, 0], active_pairs[:, 1]]
    active_jacs_lwa = map_jacobian_from_world_to_local_world_aligned(
        active_poses,
        active_jacobians,
    )[..., :task_space_dim, :]
    active_joint_gradients = torch.einsum(
        "...dj,...d->...j",
        active_jacs_lwa,
        active_gradients,
    )
    output_flat = torch.zeros(
        (gradient_flat.shape[0], dof),
        dtype=gradient_cost_wrt_x.dtype,
        device=gradient_cost_wrt_x.device,
    )
    output_flat.index_add_(0, active_pairs[:, 0], active_joint_gradients)
    return output_flat.reshape(*prefix_shape, dof)


class CostTaskSpaceCollisionObjects(CostTaskSpace):
    def __init__(self, planning_task, use_only_on_extra_objects=False, **kwargs):
        super().__init__(planning_task, **kwargs)
        self._use_only_on_extra_objects = use_only_on_extra_objects
        self.collision_objects_field = None
        self.update_collision_objects_field()

    @property
    def use_only_on_extra_objects(self):
        return self._use_only_on_extra_objects

    @use_only_on_extra_objects.setter
    def use_only_on_extra_objects(self, val):
        """
        If True, the cost will be computed only on the extra collision objects.
        If False, the cost will be computed on all collision objects.
        """
        self._use_only_on_extra_objects = val
        self.update_collision_objects_field()

    def update_collision_objects_field(self):
        if self._use_only_on_extra_objects:
            self.collision_objects_field = self.planning_task.get_collision_extra_objects_field()
        else:
            self.collision_objects_field = self.planning_task.get_collision_objects_field()
        if self.collision_objects_field is None:
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
        if self.collision_objects_field is None:
            return 0.0, 0.0

        # get link positions
        x_positions = link_pos_from_link_tensor(x_poses)[..., : self.robot.task_space_dim]

        # C, dC/dx
        # cost (and gradient) of q trajectory in phase space
        # derivative of eq. 28 wrt control points -- https://arxiv.org/pdf/2412.19948
        field_kwargs = {}
        if kwargs.get("link_indices") is not None:
            field_kwargs["link_indices"] = kwargs["link_indices"]
        if kwargs.get("precomputed_sdf_values") is not None:
            field_kwargs["precomputed_sdf_values"] = kwargs[
                "precomputed_sdf_values"
            ]
        cost, grad_cost_wrt_x = self.collision_objects_field.compute_distance_field_cost_and_gradient(
            x_positions, **field_kwargs
        )

        # (dx/dq)^T @ dC/dx, optionally packing only non-zero link entries.
        grad_cost_wrt_q_pos = project_collision_gradient_to_joints(
            x_poses,
            jacobians_spatial,
            grad_cost_wrt_x,
            self.robot.task_space_dim,
            active_link_pruning=bool(kwargs.get("active_link_pruning", False)),
        )

        # sum the cost over the task space links
        cost = cost.sum(dim=-1)

        return cost, {"pos": grad_cost_wrt_q_pos}


class CostTaskSpaceCollisionSelf(CostTaskSpace):
    def __init__(self, planning_task, **kwargs):
        super().__init__(planning_task, **kwargs)
        self.collision_self_field = self.planning_task.get_collision_self_field()
        if self.collision_self_field is None:
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
        # get link positions
        x_positions = link_pos_from_link_tensor(x_poses)[..., : self.robot.task_space_dim]

        # C, dC/dx. The field uses an explicit, stable sphere-pair gradient;
        # this avoids building a second autograd graph over all self pairs.
        field_kwargs = {}
        if kwargs.get("link_indices") is not None:
            field_kwargs["link_indices"] = kwargs["link_indices"]
            field_kwargs["self_pair_indices"] = kwargs.get("self_pair_indices")
        cost, grad_cost_wrt_x = self.collision_self_field.compute_distance_field_cost_and_gradient(
            x_positions, **field_kwargs
        )

        # (dx/dq)^T @ dC/dx, optionally packing the two active self-pair links.
        grad_cost_wrt_q_pos = project_collision_gradient_to_joints(
            x_poses,
            jacobians_spatial,
            grad_cost_wrt_x,
            self.robot.task_space_dim,
            active_link_pruning=bool(kwargs.get("active_link_pruning", False)),
        )

        # sum the cost and gradient over the task space links
        cost = cost  # the cost is the max absolute self collision distance

        return cost, {"pos": grad_cost_wrt_q_pos}


class CostTaskSpaceEEGoalComponent(CostTaskSpace):
    error_slice = slice(None)

    def __init__(self, planning_task, **kwargs):
        super().__init__(planning_task, **kwargs)

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
        ee_pose_goal = self.planning_task.ee_pose_goal
        assert ee_pose_goal is not None, "The end effector goal pose is not set in planning_task."

        # last n points of the trajectory
        ee_pose = link_poses_th_ee.squeeze(-3)

        # pose error in tangent space se(3)
        # error = log(W_EE_goal * W_EE_current^-1)
        ee_pose_inv = SE3_Func.inv(ee_pose)
        error = SE3_Func.log(SE3_Func.compose(ee_pose_goal, ee_pose_inv))
        # torch.set_printoptions(precision=2, sci_mode=False)
        # print(error[..., -1, :])

        error_component = error[..., self.error_slice]
        jacobian_component = jacs_spatial_th_ee.squeeze(-3)[..., self.error_slice, :]

        # C, dC/dx -- Task space error
        cost = 0.5 * torch.sum(error_component.square(), dim=-1)  # torch.linalg.norm(error_component, dim=-1)
        gradient_cost_wrt_x = -1.0 * error_component

        # (dx/dq)^T @ dC/dx (jacobian transpose x task space error)
        # sum the gradient and cost over the task space links
        grad_cost_wrt_q_pos = torch.einsum("...dj,...d->...j", jacobian_component, gradient_cost_wrt_x)

        # The cost is the pose error at the last trajectory point, so we also set the cost of all other points to zero
        cost[..., :-1] = 0.0

        return cost, {"pos": grad_cost_wrt_q_pos}

    def compute_cost_grad_wrt_cp(self, control_points, *args, **kwargs):
        cost, grad_cost_wrt_cp = super().compute_cost_grad_wrt_cp(
            control_points,
            *args,
            **kwargs,
        )
        # Preserve the original EE policy: only the last learnable control
        # point is adjusted, even though the endpoint basis has wider support.
        grad_cost_wrt_cp[..., :-1, :] = 0.0
        return cost, grad_cost_wrt_cp


class CostTaskSpaceEEGoalPosition(CostTaskSpaceEEGoalComponent):
    error_slice = slice(0, 3)


class CostTaskSpaceEEGoalOrientation(CostTaskSpaceEEGoalComponent):
    error_slice = slice(3, 6)


class CostTaskSpaceEEGoalPose(CostTaskSpaceEEGoalComponent):
    pass
