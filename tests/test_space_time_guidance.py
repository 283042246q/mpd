import pytest
import torch

from mpd.inference.dynamic_collision import FixedCapacityDynamicWorld
from mpd.inference.space_time_guidance import (
    InferenceOnlySpaceTimeGuide,
    SpaceTimeCostEvaluator,
    SpaceTimeGuidanceSettings,
)
from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline
from mpd.parametric_trajectory.timing_spline import TimingSpline


TENSOR_ARGS = {"device": torch.device("cpu"), "dtype": torch.float64}


def _moving_gate_world():
    world = FixedCapacityDynamicWorld(
        1,
        trajectory_duration_s=2.0,
        tensor_args=TENSOR_ARGS,
        shape_grouping_enabled=False,
        fused_reduction_enabled=False,
    )
    world.update(
        {
            "world_version": 1,
            "frame_id": "world",
            "stamp_unix_ns": 1_000_000_000,
            "valid_until_unix_ns": 6_000_000_000,
            "objects": [
                {
                    "id": "moving-gate",
                    "local_sdf": {"type": "sphere", "radius": 0.22},
                    "pose": {"position": [0.0, -1.0, 0.0]},
                    "linear_velocity": [0.0, 1.0, 0.0],
                    "inflation": {"mode": "linear"},
                }
            ],
        }
    )
    world.set_plan_start(1_000_000_000, world_version=1)
    return world


def _problem():
    timing = TimingSpline(
        num_control_points=8,
        degree=3,
        num_phase_points=65,
        u_min=0.05,
        duration_min=1.0,
        duration_max=4.0,
        tensor_args=TENSOR_ARGS,
    )
    settings = SpaceTimeGuidanceSettings(
        mode="phase5_joint",
        u_min=0.05,
        duration_min=1.0,
        duration_max=4.0,
        nominal_duration=2.0,
        dynamic_collision_weight=10.0,
        velocity_weight=0.0,
        acceleration_weight=0.0,
        duration_weight=0.01,
        timing_smoothness_weight=0.01,
    )
    evaluator = SpaceTimeCostEvaluator(
        timing,
        _moving_gate_world(),
        collision_margins=torch.tensor([0.08], **TENSOR_ARGS),
        cutoff_margin=0.0,
        velocity_limits=None,
        acceleration_limits=None,
        settings=settings,
    )
    phase = timing.phase
    q = (2.0 * phase - 1.0)[None, :, None]
    q_s = torch.full_like(q, 2.0)
    q_ss = torch.zeros_like(q)
    sphere_positions = torch.zeros((1, len(phase), 1, 3), **TENSOR_ARGS)
    sphere_positions[..., 0, 0] = q[..., 0]
    controls = timing.linear_control_points(2.0, batch_shape=(1,))
    return timing, evaluator, controls, q, q_s, q_ss, sphere_positions


def _cost(evaluator, controls, q, q_s, q_ss, sphere_positions):
    return evaluator(
        controls,
        q=q,
        q_s=q_s,
        q_ss=q_ss,
        collision_sphere_positions=sphere_positions,
    )[0].sum()


def test_duration_cost_is_single_normalized_makespan_term():
    timing, evaluator, controls, q, q_s, q_ss, sphere_positions = _problem()
    _, breakdown, evaluation = evaluator(
        controls,
        q=q,
        q_s=q_s,
        q_ss=q_ss,
        collision_sphere_positions=sphere_positions,
    )
    shorter_controls = timing.linear_control_points(1.5, batch_shape=(1,))
    _, shorter_breakdown, shorter_evaluation = evaluator(
        shorter_controls,
        q=q,
        q_s=q_s,
        q_ss=q_ss,
        collision_sphere_positions=sphere_positions,
    )

    torch.testing.assert_close(
        breakdown["duration"], evaluation.duration / 10.0
    )
    torch.testing.assert_close(
        shorter_breakdown["duration"], shorter_evaluation.duration / 10.0
    )
    assert shorter_breakdown["duration"].item() < breakdown["duration"].item()
    assert "duration_bounds" not in breakdown


def test_time_integrated_costs_are_normalized_by_candidate_duration():
    timing, evaluator, _, q, _, _, sphere_positions = _problem()

    class _ConstantDistanceWorld:
        @staticmethod
        def minimum_signed_distance(points, **_kwargs):
            return torch.zeros(points.shape[:-1], **TENSOR_ARGS)

    evaluator.dynamic_world = _ConstantDistanceWorld()
    evaluator.velocity_limits = torch.tensor([1.0], **TENSOR_ARGS)
    evaluator.acceleration_limits = torch.tensor([1.0], **TENSOR_ARGS)
    controls = torch.cat(
        (
            timing.linear_control_points(1.5, batch_shape=(1,)),
            timing.linear_control_points(3.0, batch_shape=(1,)),
        ),
        dim=0,
    )
    timing_evaluation = timing.evaluate(controls)
    q = q.expand(2, -1, -1).clone()
    q_s = 2.0 * timing_evaluation.u[..., None]
    q_ss = 2.0 * timing_evaluation.u[..., None].square()
    sphere_positions = sphere_positions.expand(2, -1, -1, -1).clone()

    _, breakdown, _ = evaluator(
        controls,
        q=q,
        q_s=q_s,
        q_ss=q_ss,
        collision_sphere_positions=sphere_positions,
    )

    for name in ("dynamic_collision", "velocity", "acceleration"):
        assert breakdown[name][0].item() == pytest.approx(
            breakdown[name][1].item(), rel=1e-10, abs=1e-10
        )
    assert breakdown["duration"][0].item() < breakdown["duration"][1].item()


def test_default_duration_weight_is_one():
    assert SpaceTimeGuidanceSettings().duration_weight == 1.0


def test_joint_dynamic_cost_gradients_match_central_difference():
    _, evaluator, controls, q, q_s, q_ss, sphere_positions = _problem()
    controls.requires_grad_(True)
    spatial_offset = torch.tensor(0.0, **TENSOR_ARGS, requires_grad=True)
    shifted_q = q + spatial_offset
    shifted_spheres = sphere_positions.clone()
    shifted_spheres[..., 0, 0] += spatial_offset
    total = _cost(evaluator, controls, shifted_q, q_s, q_ss, shifted_spheres)
    timing_gradient, spatial_gradient = torch.autograd.grad(
        total, (controls, spatial_offset)
    )

    epsilon = 1e-5
    plus = controls.detach().clone()
    minus = controls.detach().clone()
    plus[0, 3] += epsilon
    minus[0, 3] -= epsilon
    timing_fd = (
        _cost(evaluator, plus, q, q_s, q_ss, sphere_positions)
        - _cost(evaluator, minus, q, q_s, q_ss, sphere_positions)
    ) / (2.0 * epsilon)

    shifted_plus = sphere_positions.clone()
    shifted_minus = sphere_positions.clone()
    shifted_plus[..., 0, 0] += epsilon
    shifted_minus[..., 0, 0] -= epsilon
    spatial_fd = (
        _cost(evaluator, controls.detach(), q + epsilon, q_s, q_ss, shifted_plus)
        - _cost(evaluator, controls.detach(), q - epsilon, q_s, q_ss, shifted_minus)
    ) / (2.0 * epsilon)

    assert timing_gradient[0, 3].item() == pytest.approx(
        timing_fd.item(), rel=2e-5, abs=2e-6
    )
    assert spatial_gradient.item() == pytest.approx(
        spatial_fd.item(), rel=2e-5, abs=2e-6
    )


def test_moving_gate_timing_descent_reduces_cost_without_extending_duration():
    timing, evaluator, controls, q, q_s, q_ss, sphere_positions = _problem()
    controls.requires_grad_(True)
    initial = _cost(evaluator, controls, q, q_s, q_ss, sphere_positions)
    gradient = torch.autograd.grad(initial, controls)[0]
    mask = timing.optimizable_mask.to(dtype=gradient.dtype)
    updated = timing.enforce_endpoint_derivatives(
        controls.detach() - 0.1 * gradient * mask
    )
    final = _cost(evaluator, updated, q, q_s, q_ss, sphere_positions)
    duration = timing.evaluate(updated).duration.item()

    assert final.item() < initial.item()
    assert 1.0 <= duration <= 4.0
    assert abs(duration - 2.0) < 0.1


class _Dataset:
    @staticmethod
    def unnormalize_control_points(value):
        return value


class _Robot:
    dq_max = torch.tensor([20.0], **TENSOR_ARGS)
    ddq_max = torch.tensor([100.0], **TENSOR_ARGS)

    @staticmethod
    def fk_collision_spheres(q):
        transform = torch.zeros((q.shape[0], 3, 4), dtype=q.dtype, device=q.device)
        transform[:, 0, 0] = 1.0
        transform[:, 1, 1] = 1.0
        transform[:, 2, 2] = 1.0
        transform[:, 0, 3] = q[:, 0]
        return [transform]


class _DynamicField:
    def __init__(self, world):
        self.dynamic_world = world
        self.collision_margins = torch.tensor([0.08], **TENSOR_ARGS)
        self.cutoff_margin = 0.0
        self.static_field = object()


class _Profiler:
    @staticmethod
    def snapshot():
        return []


class _SpatialGuide:
    costs = {}
    guidance_profiler = _Profiler()

    @staticmethod
    def warmup(_shape):
        return None

    @staticmethod
    def use_all_collision_objects():
        return None

    @staticmethod
    def __call__(control_points, **_kwargs):
        return torch.ones_like(control_points)


class _RecordingSpatialGuide(_SpatialGuide):
    def __init__(self):
        self.call_kwargs = None

    def __call__(self, control_points, **kwargs):
        self.call_kwargs = kwargs
        return torch.ones_like(control_points)


class _PlanningTask:
    def __init__(self):
        self.robot = _Robot()
        self.parametric_trajectory = ParametricTrajectoryBspline(
            n_control_points=7,
            degree=3,
            zero_vel_at_start_and_goal=True,
            zero_acc_at_start_and_goal=True,
            remove_outer_control_points=False,
            num_T_pts=65,
            trajectory_duration=2.0,
            tensor_args=TENSOR_ARGS,
        )
        self.parametric_trajectory.set_boundary_conditions(
            q_pos_start=torch.tensor([-1.0], **TENSOR_ARGS),
            q_pos_goal=torch.tensor([1.0], **TENSOR_ARGS),
        )


def test_phase5_spatial_descent_disables_fixed_time_kinematic_costs():
    settings = SpaceTimeGuidanceSettings.from_mapping(
        {
            "mode": "phase5_joint",
            "duration_min": 1.0,
            "duration_max": 4.0,
            "nominal_duration": 2.0,
        }
    )
    spatial_guide = _RecordingSpatialGuide()
    guide = InferenceOnlySpaceTimeGuide(
        spatial_guide,
        _PlanningTask(),
        _Dataset(),
        _DynamicField(_moving_gate_world()),
        settings,
        TENSOR_ARGS,
    )
    control_points = torch.zeros((2, 7, 1), **TENSOR_ARGS)

    guide._spatial_descent(
        control_points,
        cost_weight_overrides={
            "CostJointSpaceVelocity": 99.0,
            "CostTaskSpaceEEGoalPosition": 3.0,
        },
    )

    assert spatial_guide.call_kwargs["cost_weight_overrides"] == {
        "CostJointSpaceVelocity": 0.0,
        "CostTaskSpaceEEGoalPosition": 3.0,
        "CostJointSpaceAcceleration": 0.0,
    }


def test_shared_trajectory_state_evaluates_timing_once():
    settings = SpaceTimeGuidanceSettings.from_mapping(
        {
            "mode": "phase5_joint",
            "duration_min": 1.0,
            "duration_max": 4.0,
            "nominal_duration": 2.0,
        }
    )
    guide = InferenceOnlySpaceTimeGuide(
        _SpatialGuide(),
        _PlanningTask(),
        _Dataset(),
        _DynamicField(_moving_gate_world()),
        settings,
        TENSOR_ARGS,
    )
    guide.reset(2)
    original_evaluate = guide.timing_spline.evaluate
    evaluate_calls = []

    def counted_evaluate(*args, **kwargs):
        evaluate_calls.append((args, kwargs))
        return original_evaluate(*args, **kwargs)

    guide.timing_spline.evaluate = counted_evaluate
    control_points = torch.zeros((2, 7, 1), **TENSOR_ARGS)

    _, _, evaluation = guide.evaluate_control_points(control_points)

    assert len(evaluate_calls) == 1
    assert evaluation.q is not None
    assert evaluation.dq is not None
    assert evaluation.ddq is not None


@pytest.mark.parametrize(
    "mode, expect_spatial_update",
    [
        ("phase5_scalar_duration", True),
        ("phase5_timing_only", False),
        ("phase5_joint", True),
    ],
)
def test_wrapper_modes_keep_population_fixed_and_update_only_owned_variables(
    mode, expect_spatial_update
):
    settings = SpaceTimeGuidanceSettings.from_mapping(
        {
            "mode": mode,
            "duration_min": 1.0,
            "duration_max": 4.0,
            "nominal_duration": 2.0,
            "timing_learning_rate": 0.03,
            "velocity_weight": 0.0,
            "acceleration_weight": 0.0,
            "duration_weight": 0.01,
            "timing_smoothness_weight": 0.01,
        }
    )
    task = _PlanningTask()
    guide = InferenceOnlySpaceTimeGuide(
        _SpatialGuide(),
        task,
        _Dataset(),
        _DynamicField(_moving_gate_world()),
        settings,
        TENSOR_ARGS,
    )
    guide.reset(4)
    before = guide.timing_control_points.clone()
    spatial_control_points = torch.linspace(-1.0, 1.0, 7, **TENSOR_ARGS)[None, :, None].expand(4, -1, -1).clone()

    descent = guide(spatial_control_points)

    assert descent.shape == spatial_control_points.shape
    assert guide.timing_control_points.shape == (4, 8)
    assert not torch.equal(guide.timing_control_points, before)
    if expect_spatial_update:
        assert torch.count_nonzero(descent).item() > 0
    else:
        assert torch.count_nonzero(descent).item() == 0
    if mode == "phase5_scalar_duration":
        torch.testing.assert_close(
            guide.timing_control_points,
            guide.timing_control_points[:, :1].expand(-1, 8),
        )
    else:
        torch.testing.assert_close(
            guide.timing_control_points[:, :2], before[:, :2]
        )
        torch.testing.assert_close(
            guide.timing_control_points[:, -2:], before[:, -2:]
        )
    assert guide.statistics[-1]["mode"] == mode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_joint_dynamic_cost_has_finite_cuda_space_time_gradients():
    tensor_args = {"device": torch.device("cuda"), "dtype": torch.float64}
    timing = TimingSpline(
        num_control_points=8,
        degree=3,
        num_phase_points=33,
        u_min=0.05,
        duration_min=1.0,
        duration_max=4.0,
        tensor_args=tensor_args,
    )
    world = FixedCapacityDynamicWorld(
        1, trajectory_duration_s=2.0, tensor_args=tensor_args
    )
    world.update(
        {
            "world_version": 1,
            "frame_id": "world",
            "stamp_unix_ns": 1_000_000_000,
            "valid_until_unix_ns": 6_000_000_000,
            "objects": [
                {
                    "id": "gate",
                    "local_sdf": {"type": "sphere", "radius": 0.2},
                    "pose": {"position": [0.0, -1.0, 0.0]},
                    "linear_velocity": [0.0, 1.0, 0.0],
                }
            ],
        }
    )
    world.set_plan_start(1_000_000_000, world_version=1)
    settings = SpaceTimeGuidanceSettings(
        duration_min=1.0,
        duration_max=4.0,
        nominal_duration=2.0,
        velocity_weight=0.0,
        acceleration_weight=0.0,
    )
    evaluator = SpaceTimeCostEvaluator(
        timing,
        world,
        collision_margins=torch.tensor([0.08], **tensor_args),
        cutoff_margin=0.0,
        velocity_limits=None,
        acceleration_limits=None,
        settings=settings,
    )
    spatial = torch.linspace(-1.0, 1.0, 33, **tensor_args).reshape(1, 33, 1)
    spatial.requires_grad_(True)
    q_s = torch.full_like(spatial, 2.0)
    q_ss = torch.zeros_like(spatial)
    spheres = torch.zeros((1, 33, 1, 3), **tensor_args)
    spheres[..., 0] = spatial
    controls = timing.linear_control_points(2.0, batch_shape=(1,)).requires_grad_(True)

    total = evaluator(
        controls,
        q=spatial,
        q_s=q_s,
        q_ss=q_ss,
        collision_sphere_positions=spheres,
    )[0].sum()
    spatial_gradient, timing_gradient = torch.autograd.grad(total, (spatial, controls))

    assert torch.isfinite(spatial_gradient).all().item()
    assert torch.isfinite(timing_gradient).all().item()
    assert spatial_gradient.norm().item() > 0.0
    assert timing_gradient.norm().item() > 0.0
