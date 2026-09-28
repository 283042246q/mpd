import numpy as np
import pytest
import torch

from mpd.inference.space_time_guidance import InferenceOnlySpaceTimeGuide, SpaceTimeGuidanceSettings
from mpd.inference.time_corridor import _reachable_branch, _corridor_cost, refine_time_corridor
from test_space_time_guidance import (
    TENSOR_ARGS, _Dataset, _DynamicField, _PlanningTask, _SpatialGuide, _moving_gate_world,
)
from test_factorized_guidance import problem as factorized_problem
from scripts.runtime.infer_space_time_server import _build_parser as phase5_server_parser
from scripts.runtime.infer_factorized_server import _build_parser as f1_server_parser
from scripts.inference.infer_space_time import _build_parser as phase5_direct_parser
from scripts.inference.infer_factorized import _build_parser as f1_direct_parser


def test_branch_search_preserves_disconnected_early_and_late_windows():
    grid = np.arange(0.0, 5.1, 0.1)
    safe = np.ones((3, len(grid)), dtype=bool)
    safe[1, (grid > 1.0) & (grid < 3.0)] = False
    branch = _reachable_branch(
        safe, np.array([0.0, 2.0, 4.0]), grid, np.array([1, 1]),
        duration_min=3.0, margin=0.05, preference_s=-1.0,
    )
    assert branch is not None
    assert branch[1][1] <= 1.0
    arrivals = torch.tensor([0.0, 2.0, 4.0])
    assert _corridor_cost(arrivals, torch.tensor(branch)) > 0


def test_phase5_refinement_keeps_path_and_endpoint_controls_fixed():
    settings = SpaceTimeGuidanceSettings.from_mapping(dict(
        mode="phase5_joint", duration_min=1.0, duration_max=4.0,
        nominal_duration=2.0, corridor_a_enabled=True,
        corridor_a_phase_points=16, corridor_a_time_step_s=0.1,
        corridor_a_steps=5,
    ))
    guide = InferenceOnlySpaceTimeGuide(
        _SpatialGuide(), _PlanningTask(), _Dataset(),
        _DynamicField(_moving_gate_world()), settings, TENSOR_ARGS,
    )
    path = torch.zeros((1, 7, 1), **TENSOR_ARGS)
    guide.reset(1)
    controls = guide.timing_spline.linear_control_points(2.0, batch_shape=(1,))
    before = path.clone()
    original_risk = guide.evaluate_control_points(path, controls)[1]["dynamic_collision"][0]
    refined, stats = refine_time_corridor(guide, path, controls, factorized=False)
    refined_risk = guide.evaluate_control_points(path, refined)[1]["dynamic_collision"][0]
    assert stats["enabled"] and stats["branches_tested"] >= 1
    for field in ("grid_build_s", "branch_selection_s",
                  "refinement_forward_s", "refinement_backward_s"):
        assert stats[field] >= 0.0
    assert refined_risk < original_risk
    assert refined.shape == controls.shape
    torch.testing.assert_close(path, before)
    torch.testing.assert_close(refined[:, :2], controls[:, :2])
    torch.testing.assert_close(refined[:, -2:], controls[:, -2:])


@pytest.mark.parametrize("representation", ["c", "tau_r"])
def test_f1_refinement_preserves_six_dimensional_checkpoint_latent(representation):
    guide, path, latent = factorized_problem(representation)
    guide.settings = SpaceTimeGuidanceSettings.from_mapping(dict(
        duration_min=1.0, duration_max=4.0, nominal_duration=2.0,
        corridor_a_enabled=True, corridor_a_phase_points=16,
        corridor_a_time_step_s=0.1, corridor_a_steps=3,
    ))
    guide.cost_evaluator.settings = guide.settings
    original = latent.clone()
    refined, stats = refine_time_corridor(guide, path[:1], latent[:1], factorized=True)
    assert refined.shape == (1, 6)
    assert torch.isfinite(refined).all()
    assert stats["enabled"]
    torch.testing.assert_close(latent, original)


def test_corridor_rejects_non_joint_phase5_mode():
    with pytest.raises(ValueError, match="phase5_joint"):
        SpaceTimeGuidanceSettings.from_mapping(dict(
            mode="phase5_timing_only", corridor_a_enabled=True,
        ))


def test_phase5_and_f1_entries_have_independent_default_off_switches():
    server_base = ["--socket", "/tmp/corridor.sock", "--output-root", "/tmp/corridor"]
    phase5_base = ["--request", "/tmp/request.json", "--output-dir", "/tmp/output"]
    factorized_base = ["--timing-checkpoint", "/tmp/timing.pt"]
    for parser, base in (
        (phase5_server_parser(), server_base),
        (f1_server_parser(), server_base + factorized_base),
        (phase5_direct_parser(), phase5_base),
        (f1_direct_parser(), phase5_base + factorized_base),
    ):
        assert not parser.parse_args(base).corridor_a
        enabled = parser.parse_args(base + ["--corridor-a", "--corridor-a-weight", "0.2"])
        assert enabled.corridor_a and enabled.corridor_a_weight == 0.2
