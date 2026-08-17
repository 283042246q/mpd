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
