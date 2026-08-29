from types import SimpleNamespace

import torch

from mpd.inference.dynamic_risk import mean_cvar_dynamic_risk
from mpd.inference.fixed_time_aligned_guidance import (
    FixedTimeAlignedGuide,
    FixedTimeAlignedGuidanceSettings,
)


def test_dynamic_risk_is_equal_mean_cvar_blend_with_finite_gradient():
    times = torch.linspace(0.0, 10.0, 101, dtype=torch.float64)[None]
    penetration = torch.linspace(0.01, 0.2, 101, dtype=torch.float64)[
        None, :, None
    ].requires_grad_(True)

    risk, breakdown = mean_cvar_dynamic_risk(
        penetration,
        times,
        alpha=0.5,
        cvar_fraction=0.10,
    )

    torch.testing.assert_close(
        risk,
        0.5 * breakdown["mean"] + 0.5 * breakdown["cvar"],
    )
    assert breakdown["cvar"].item() > breakdown["mean"].item()
    gradient = torch.autograd.grad(risk.sum(), penetration)[0]
    assert torch.isfinite(gradient).all()


def test_fixed_time_aligned_guide_replaces_only_dynamic_collision_direction():
    static_field = object()
    dynamic_field = SimpleNamespace(static_field=static_field)
    collision_cost = SimpleNamespace(collision_objects_field=dynamic_field)

    class _SpatialGuide:
        costs = {
            "CostTaskSpaceCollisionObjects": SimpleNamespace(cost=collision_cost)
        }

        def __init__(self):
            self.fields_seen = []
            self.guidance_profiler = object()
            self.used_all_collision_objects = False

        def __call__(self, controls, return_cost=False, **_kwargs):
            self.fields_seen.append(collision_cost.collision_objects_field)
            descent = torch.ones_like(controls)
            if return_cost:
                return torch.full(
                    (controls.shape[0],),
                    3.0,
                    dtype=controls.dtype,
                    device=controls.device,
                ), descent
            return descent

        def use_all_collision_objects(self):
            self.used_all_collision_objects = True

        @staticmethod
        def warmup(shape_x):
            return shape_x

    class _RiskEvaluator:
        @staticmethod
        def evaluate_control_points(controls):
            risk = controls.square().flatten(start_dim=1).sum(dim=-1)
            return risk, {"mean": risk, "cvar": risk}

    spatial = _SpatialGuide()
    guide = FixedTimeAlignedGuide(
        spatial,
        _RiskEvaluator(),
        dynamic_field,
        dynamic_collision_weight=2.0,
        settings=FixedTimeAlignedGuidanceSettings(dynamic_max_grad_norm=100.0),
    )
    controls = torch.full((2, 3, 1), 0.1, dtype=torch.float64)

    total, descent = guide(controls, return_cost=True)

    assert spatial.fields_seen == [static_field]
    assert collision_cost.collision_objects_field is dynamic_field
    assert guide.guidance_profiler is spatial.guidance_profiler
    guide.use_all_collision_objects()
    assert spatial.used_all_collision_objects
    assert guide.warmup((2, 3, 1)) == (2, 3, 1)
    torch.testing.assert_close(descent, torch.full_like(controls, 0.6))
    torch.testing.assert_close(total, torch.full((2,), 3.06, dtype=torch.float64))
