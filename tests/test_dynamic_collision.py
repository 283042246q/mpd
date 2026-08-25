import pytest
import torch

from mpd.inference.dynamic_collision import (
    DynamicWorldError,
    FixedCapacityDynamicWorld,
    StaticDynamicCollisionField,
)


TENSOR_ARGS = {"device": torch.device("cpu"), "dtype": torch.float64}


def _world(objects, version=1, stamp=1_000_000_000, valid_until=20_000_000_000):
    return {
        "world_version": version,
        "frame_id": "fr3_link0",
        "stamp_unix_ns": stamp,
        "valid_until_unix_ns": valid_until,
        "objects": objects,
    }


def _sphere(**overrides):
    item = {
        "id": "moving-sphere",
        "local_sdf": {"type": "sphere", "radius": 0.2},
        "pose": {"position": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "linear_velocity": [1.0, 0.0, 0.0],
        "inflation": {"mode": "linear", "base_m": 0.01, "horizon_rate_m_s": 0.02},
    }
    item.update(overrides)
    return item


def _mixed_objects():
    covariance = [0.0] * 36
    for axis in range(3):
        covariance[axis * 6 + axis] = 0.01
    return [
        _sphere(id="sphere"),
        {
            "id": "box",
            "local_sdf": {"type": "box", "size_xyz": [0.4, 0.6, 0.8]},
            "pose": {
                "position": [0.5, -0.2, 0.1],
                "orientation_xyzw": [0.0, 0.0, 2**-0.5, 2**-0.5],
            },
            "linear_velocity": [0.0, 0.1, 0.0],
            "inflation": {"mode": "covariance", "base_m": 0.02},
            "covariance_6x6": covariance,
        },
        {
            "id": "capsule",
            "local_sdf": {"type": "capsule", "radius": 0.1, "length": 0.6},
            "pose": {
                "position": [-0.2, 0.3, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "linear_velocity": [0.0, 0.0, 0.1],
            "inflation": {"mode": "linear", "horizon_rate_m_s": 0.01},
        },
    ]


def test_constant_velocity_and_linear_horizon_inflation_use_fixed_timing():
    world = FixedCapacityDynamicWorld(4, trajectory_duration_s=2.0, tensor_args=TENSOR_ARGS)
    world.update(_world([_sphere()]))
    world.set_plan_start(2_000_000_000, world_version=1)
    # At trajectory t=0 the center is x=1 and inflation is .03.  At t=2 it is
    # x=3 and inflation is .07.  Query .5 m to the right in both cases.
    points = torch.tensor([[[[1.5, 0.0, 0.0]], [[3.5, 0.0, 0.0]]]], **TENSOR_ARGS)
    distances, gradients = world.signed_distances_and_gradients(points)
    assert distances[0, :, 0, 0].tolist() == pytest.approx([0.27, 0.23])
    assert gradients[0, :, 0, 0, 0].tolist() == pytest.approx([1.0, 1.0])
    assert torch.isinf(distances[..., 1:, :]).all()


def test_orientation_is_held_and_applied_to_box_local_sdf():
    box = {
        "id": "rotated-box",
        "local_sdf": {"type": "box", "size_xyz": [2.0, 0.2, 0.2]},
        "pose": {
            "position": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 2**-0.5, 2**-0.5],
        },
        "linear_velocity": [0.0, 0.0, 0.0],
        "inflation": {"mode": "linear"},
    }
    world = FixedCapacityDynamicWorld(1, trajectory_duration_s=1.0, tensor_args=TENSOR_ARGS)
    world.update(_world([box]))
    world.set_plan_start(1_000_000_000, world_version=1)
    points = torch.tensor([[[[0.0, 0.8, 0.0], [0.8, 0.0, 0.0]]]], **TENSOR_ARGS)
    distances, _ = world.signed_distances_and_gradients(points)
    assert distances[0, 0, 0].tolist() == pytest.approx([-0.1, 0.7], abs=1e-7)


def test_covariance_inflation_grows_under_cv_propagation():
    covariance = [0.0] * 36
    for axis in range(3):
        covariance[axis * 6 + axis] = 0.01
        covariance[(axis + 3) * 6 + axis + 3] = 0.04
    sphere = _sphere(
        linear_velocity=[0.0, 0.0, 0.0],
        covariance_6x6=covariance,
        inflation={"mode": "covariance", "base_m": 0.0},
    )
    world = FixedCapacityDynamicWorld(
        1,
        trajectory_duration_s=2.0,
        tensor_args=TENSOR_ARGS,
        covariance_sigma=2.0,
        process_acceleration_std_m_s2=0.0,
    )
    world.update(_world([sphere]))
    world.set_plan_start(1_000_000_000, world_version=1)
    points = torch.tensor([[[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]], **TENSOR_ARGS)
    distances, _ = world.signed_distances_and_gradients(points)
    # sqrt(.01)=.1 initially; sqrt(.01 + 4*.04)=sqrt(.17) at two seconds.
    assert distances[0, 0, 0, 0].item() == pytest.approx(0.6)
    assert distances[0, 1, 0, 0].item() == pytest.approx(0.8 - 2.0 * (0.17**0.5))


def test_world_version_and_prediction_validity_fail_closed():
    world = FixedCapacityDynamicWorld(1, trajectory_duration_s=2.0, tensor_args=TENSOR_ARGS)
    world.update(_world([_sphere()], valid_until=3_000_000_000))
    with pytest.raises(DynamicWorldError, match="world_version"):
        world.set_plan_start(1_000_000_000, world_version=2)
    with pytest.raises(DynamicWorldError, match="exceeds"):
        world.set_plan_start(2_000_000_000, world_version=1)
    with pytest.raises(DynamicWorldError, match="increase"):
        world.update(_world([], version=1))


def test_combined_field_selects_deepest_static_or_dynamic_cost_and_gradient():
    class StaticField:
        robot = object()
        collision_margins = torch.tensor([0.1], **TENSOR_ARGS)
        cutoff_margin = 0.02
        clamp_sdf = True

        def object_signed_distances(self, points, get_gradient=False, **_kwargs):
            distances = torch.full((*points.shape[:-2], 1, points.shape[-2]), 0.5, **TENSOR_ARGS)
            if get_gradient:
                return distances, torch.zeros((*distances.shape, 3), **TENSOR_ARGS)
            return distances

        def object_signed_distance_gradients(self, points, **_kwargs):
            return torch.zeros((*points.shape[:-2], 1, points.shape[-2], 3), **TENSOR_ARGS)

    world = FixedCapacityDynamicWorld(1, trajectory_duration_s=1.0, tensor_args=TENSOR_ARGS)
    world.update(_world([_sphere(linear_velocity=[0.0, 0.0, 0.0], inflation={"mode": "linear"})]))
    world.set_plan_start(1_000_000_000, world_version=1)
    field = StaticDynamicCollisionField(StaticField(), world)
    points = torch.tensor([[[[0.25, 0.0, 0.0]]]], **TENSOR_ARGS)
    cost, gradient = field.compute_distance_field_cost_and_gradient(points)
    # Dynamic distance=.05, link+cutoff margin=.12, penetration=.07.
    assert cost.item() == pytest.approx(0.07)
    assert gradient.flatten().tolist() == pytest.approx([-1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "option",
    [
        "capacity_buckets_enabled",
        "shape_grouping_enabled",
        "time_table_cache_enabled",
    ],
)
def test_each_dynamic_query_optimization_matches_baseline(option):
    baseline = FixedCapacityDynamicWorld(
        8, trajectory_duration_s=2.0, tensor_args=TENSOR_ARGS
    )
    optimized = FixedCapacityDynamicWorld(
        8, trajectory_duration_s=2.0, tensor_args=TENSOR_ARGS, **{option: True}
    )
    for world in (baseline, optimized):
        world.update(_world(_mixed_objects()))
        world.set_plan_start(1_000_000_000, world_version=1)
    points = torch.tensor(
        [
            [
                [[0.2, 0.1, 0.0], [0.7, -0.2, 0.1]],
                [[1.2, 0.0, 0.0], [-0.2, 0.3, 0.4]],
            ]
        ],
        **TENSOR_ARGS,
    )
    expected_distance, expected_gradient = baseline.signed_distances_and_gradients(points)
    actual_distance, actual_gradient = optimized.signed_distances_and_gradients(points)
    active = len(_mixed_objects())
    assert torch.allclose(
        actual_distance[..., :active, :], expected_distance[..., :active, :]
    )
    assert torch.allclose(
        actual_gradient[..., :active, :, :], expected_gradient[..., :active, :, :]
    )
    if option == "capacity_buckets_enabled":
        assert actual_distance.shape[-2] == 4


def test_time_table_cache_is_reused_then_invalidated_by_plan_start():
    world = FixedCapacityDynamicWorld(
        4,
        trajectory_duration_s=2.0,
        tensor_args=TENSOR_ARGS,
        time_table_cache_enabled=True,
    )
    world.update(_world([_sphere()]))
    world.set_plan_start(1_000_000_000, world_version=1)
    first = world._time_table(128, torch.float64, torch.device("cpu"))
    second = world._time_table(128, torch.float64, torch.device("cpu"))
    assert first is second
    world.set_plan_start(2_000_000_000, world_version=1)
    third = world._time_table(128, torch.float64, torch.device("cpu"))
    assert third is not first


def test_grouped_fused_minimum_matches_materialized_object_reduction():
    baseline = FixedCapacityDynamicWorld(
        8, trajectory_duration_s=2.0, tensor_args=TENSOR_ARGS
    )
    optimized = FixedCapacityDynamicWorld(
        8,
        trajectory_duration_s=2.0,
        tensor_args=TENSOR_ARGS,
        capacity_buckets_enabled=True,
        shape_grouping_enabled=True,
        time_table_cache_enabled=True,
        fused_reduction_enabled=True,
    )
    for world in (baseline, optimized):
        world.update(_world(_mixed_objects()))
        world.set_plan_start(1_000_000_000, world_version=1)
    torch.manual_seed(7)
    points = torch.randn((2, 5, 4, 3), **TENSOR_ARGS)
    distances, gradients = baseline.signed_distances_and_gradients(points)
    expected_distance, active_object = distances.min(dim=-2)
    gather_index = active_object.unsqueeze(-2).unsqueeze(-1).expand(
        *active_object.shape[:-1], 1, active_object.shape[-1], 3
    )
    expected_gradient = gradients.gather(-3, gather_index).squeeze(-3)
    actual_distance, actual_gradient = optimized.minimum_signed_distance_and_gradient(points)
    assert torch.allclose(actual_distance, expected_distance)
    assert torch.allclose(actual_gradient, expected_gradient)


def test_candidate_specific_grouped_fused_distance_and_gradients_match_baseline():
    baseline = FixedCapacityDynamicWorld(
        8, trajectory_duration_s=2.0, tensor_args=TENSOR_ARGS
    )
    optimized = FixedCapacityDynamicWorld(
        8,
        trajectory_duration_s=2.0,
        tensor_args=TENSOR_ARGS,
        capacity_buckets_enabled=True,
        shape_grouping_enabled=True,
        fused_reduction_enabled=True,
    )
    for world in (baseline, optimized):
        world.update(_world(_mixed_objects()))
        world.set_plan_start(1_000_000_000, world_version=1)

    torch.manual_seed(17)
    points_value = torch.randn((2, 5, 4, 3), **TENSOR_ARGS)
    times_value = torch.tensor(
        [[0.0, 0.25, 0.7, 1.2, 1.8], [0.0, 0.4, 0.9, 1.4, 2.0]],
        **TENSOR_ARGS,
    )
    weights = torch.linspace(0.2, 1.0, 2 * 5 * 4, **TENSOR_ARGS).reshape(2, 5, 4)

    def evaluate(world):
        points = points_value.clone().requires_grad_(True)
        times = times_value.clone().requires_grad_(True)
        distance, analytic_gradient = world.minimum_signed_distance_and_gradient(
            points, trajectory_times=times
        )
        point_gradient, time_gradient = torch.autograd.grad(
            (distance * weights).sum(), (points, times)
        )
        return distance, analytic_gradient, point_gradient, time_gradient

    expected = evaluate(baseline)
    actual = evaluate(optimized)
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(actual_value, expected_value, atol=1e-9, rtol=1e-8)


def test_candidate_specific_times_change_collision_for_same_spatial_path():
    world = FixedCapacityDynamicWorld(
        1, trajectory_duration_s=2.0, tensor_args=TENSOR_ARGS
    )
    world.update(
        _world(
            [
                _sphere(
                    linear_velocity=[1.0, 0.0, 0.0],
                    inflation={"mode": "linear"},
                )
            ]
        )
    )
    world.set_plan_start(1_000_000_000, world_version=1)
    points = torch.tensor(
        [
            [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]],
        ],
        **TENSOR_ARGS,
    )
    trajectory_times = torch.tensor(
        [[0.0, 1.0, 2.0], [0.0, 0.5, 1.0]], **TENSOR_ARGS
    )

    distances, _ = world.signed_distances_and_gradients(
        points, trajectory_times=trajectory_times
    )

    assert distances[0, :, 0, 0].tolist() == pytest.approx([-0.2, -0.2, -0.2])
    assert distances[1, :, 0, 0].tolist() == pytest.approx([-0.2, 0.3, 0.8])


def test_candidate_time_gradient_matches_central_difference():
    world = FixedCapacityDynamicWorld(
        1, trajectory_duration_s=2.0, tensor_args=TENSOR_ARGS
    )
    world.update(
        _world(
            [
                _sphere(
                    linear_velocity=[0.5, 0.0, 0.0],
                    inflation={"mode": "linear", "horizon_rate_m_s": 0.1},
                )
            ]
        )
    )
    world.set_plan_start(1_000_000_000, world_version=1)
    points = torch.tensor(
        [[[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]],
        **TENSOR_ARGS,
    )
    trajectory_times = torch.tensor(
        [[0.0, 0.6, 1.2]], **TENSOR_ARGS, requires_grad=True
    )
    distance, _ = world.signed_distances_and_gradients(
        points, trajectory_times=trajectory_times
    )
    gradient = torch.autograd.grad(distance[0, 1, 0, 0], trajectory_times)[0][0, 1]

    epsilon = 1e-5
    plus = trajectory_times.detach().clone()
    minus = trajectory_times.detach().clone()
    plus[0, 1] += epsilon
    minus[0, 1] -= epsilon
    plus_distance = world.signed_distances_and_gradients(
        points, trajectory_times=plus
    )[0][0, 1, 0, 0]
    minus_distance = world.signed_distances_and_gradients(
        points, trajectory_times=minus
    )[0][0, 1, 0, 0]
    finite_difference = (plus_distance - minus_distance) / (2.0 * epsilon)
    assert gradient.item() == pytest.approx(finite_difference.item(), rel=1e-8, abs=1e-8)
    assert gradient.item() == pytest.approx(-0.6)


def test_candidate_specific_times_fail_closed_on_invalid_contract():
    world = FixedCapacityDynamicWorld(
        1, trajectory_duration_s=2.0, tensor_args=TENSOR_ARGS
    )
    world.update(_world([_sphere()], valid_until=3_000_000_000))
    world.set_plan_start(1_000_000_000, world_version=1)
    points = torch.zeros((1, 3, 1, 3), **TENSOR_ARGS)

    with pytest.raises(ValueError, match="shape"):
        world.signed_distances_and_gradients(
            points, trajectory_times=torch.zeros((2, 3), **TENSOR_ARGS)
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        world.signed_distances_and_gradients(
            points,
            trajectory_times=torch.tensor([[0.0, 1.0, 1.0]], **TENSOR_ARGS),
        )
    with pytest.raises(DynamicWorldError, match="prediction validity"):
        world.signed_distances_and_gradients(
            points,
            trajectory_times=torch.tensor([[0.0, 1.0, 2.1]], **TENSOR_ARGS),
        )
