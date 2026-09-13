import pytest
import torch

from mpd.inference.factorized_guidance import FactorizedCostGuide
from mpd.inference.factorized_sampler import FactorizedSettings
from mpd.inference.learned_timing import LearnedTimingCodec
from mpd.inference.space_time_guidance import SpaceTimeGuidanceSettings
from test_space_time_guidance import (
    TENSOR_ARGS, _Dataset, _PlanningTask, _SpatialGuide, _DynamicField, _moving_gate_world,
)


def problem(representation):
    task = _PlanningTask()
    world = _moving_gate_world()
    settings = SpaceTimeGuidanceSettings(duration_min=1., duration_max=4., nominal_duration=2.)
    codec = LearnedTimingCodec(dict(representation=representation, path_mean=[0.], path_std=[1.],
        target_mean=[0.] * 6, target_std=[1.] * 6), num_phase_points=65,
        duration_min=1., duration_max=4., velocity_limits=[20.], acceleration_limits=[100.]).to(**TENSOR_ARGS)
    guide = FactorizedCostGuide(_SpatialGuide(), task, _Dataset(), _DynamicField(world), settings,
        TENSOR_ARGS, codec=codec, factorized_settings=FactorizedSettings())
    p = torch.zeros(2, 7, 1, **TENSOR_ARGS)
    z = torch.full((2, 6), 1.8 if representation == "c" else 0., **TENSOR_ARGS)
    return guide, p, z


@pytest.mark.parametrize("representation", ["c", "tau_r"])
def test_gradient_ownership_and_no_implicit_timing_optimizer(representation):
    guide, p, z = problem(representation)
    saved_p, saved_z = p.clone(), z.clone()
    with torch.no_grad():
        gp, gz = guide.gradients(p, z, active="timing")
        assert torch.count_nonzero(gp) == 0 and gz.abs().sum() > 0
        gp, gz = guide.gradients(p, z, active="space")
        assert torch.count_nonzero(gz) == 0 and gp.abs().sum() > 0
        pp, zz = guide.refine(p, z, active="joint")
    assert torch.isfinite(pp).all() and torch.isfinite(zz).all()
    torch.testing.assert_close(p, saved_p)
    torch.testing.assert_close(z, saved_z)
    assert guide.timing_control_points is None


def test_empty_world_weak_guide_has_zero_dynamic_gradient_but_static_descent():
    guide, p, z = problem("c")
    world = guide.dynamic_field.dynamic_world
    world.update(dict(world_version=2, frame_id="world", stamp_unix_ns=1_000_000_000,
                      valid_until_unix_ns=6_000_000_000, objects=[]))
    world.set_plan_start(1_000_000_000, world_version=2)
    gp, gz = guide.gradients(p, z, active="space", weak=True)
    torch.testing.assert_close(gp, -torch.ones_like(p))
    torch.testing.assert_close(gz, torch.zeros_like(z))


def test_weak_space_stage_uses_exact_nominal_time_and_excludes_kinematic_costs():
    guide, p, z = problem("c")
    total, breakdown, timing = guide.evaluate(p, z, weak=True)
    torch.testing.assert_close(timing.time_from_start,
        timing.phase[None].expand(2, -1) * 2., atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(total, .2 * 10. * breakdown["dynamic_collision"])


def test_nonzero_boundary_rejected_instead_of_silently_changing_request():
    guide, p, z = problem("tau_r")
    guide.planning_task.parametric_trajectory.q_vel_start = p.new_tensor([.1])
    with pytest.raises(ValueError, match="zero endpoint"):
        guide.condition(p)
