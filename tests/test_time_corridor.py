import numpy as np
import math
import pytest
import torch

from mpd.inference.space_time_guidance import InferenceOnlySpaceTimeGuide, SpaceTimeGuidanceSettings
from mpd.inference.time_corridor import (
    _reachable_branch, _safe_intervals, _corridor_cost, refine_time_corridor,
    refine_time_corridor_batch, _time_distance_table,
    _interpolate_time_distance_table, _constant_sphere_interval_grid,
    _reachable_branches_kbest,
    _window_signature,
    validate_alternative_branches,
)


def test_optimized_window_signature_tracks_continuous_arrivals():
    rows = [([(0.0, 1.0)], None), ([(1.0, 2.0), (3.0, 4.0)], None)]
    assert _window_signature([0.0, 3.5], rows) == (0, 1)
    assert _window_signature([0.0, 2.5], rows) == (0, -1)


def test_alternative_branches_use_lowest_cost_first_and_obey_dense_budget():
    selected = torch.tensor([[0.0], [0.0]])
    catalog = {
        0: [dict(parameters=torch.tensor([0.0]), objective=0.0, window_signature=(0,)),
            dict(parameters=torch.tensor([1.0]), objective=1.0, window_signature=(1,))],
        1: [dict(parameters=torch.tensor([0.0]), objective=0.0, window_signature=(0,)),
            dict(parameters=torch.tensor([2.0]), objective=2.0, window_signature=(1,))],
    }
    calls = []
    def validate(indices, parameters):
        calls.append((indices.tolist(), parameters[:, 0].tolist()))
        return parameters[:, 0] == 1.0
    output, stats = validate_alternative_branches(
        catalog, selected, torch.tensor([False, False]), validate,
        budget=1,
    )
    assert calls == [([0], [1.0])]
    torch.testing.assert_close(output, torch.tensor([[1.0], [0.0]]))
    assert stats["alternate_branch_rescued_candidates"] == 1
    assert stats["alternative_branches_checked"] == 1
    assert stats["validated_unique_window_sequences"] == 1


def test_alternative_branches_skip_time_invariant_invalid_paths():
    selected = torch.zeros(2, 1)
    options = [dict(parameters=torch.tensor([value]), objective=float(index),
                    window_signature=(index,))
               for index, value in enumerate((0.0, 1.0))]
    calls = []
    def validate(indices, parameters):
        calls.extend(indices.tolist())
        return torch.ones(len(indices), dtype=torch.bool)
    output, stats = validate_alternative_branches(
        {0: options, 1: options}, selected, torch.tensor([False, False]),
        validate, budget=4, ineligible=torch.tensor([True, False]),
    )
    assert calls == [1] and stats["alternate_branch_rescued_candidates"] == 1
    torch.testing.assert_close(output, torch.tensor([[0.0], [1.0]]))


from test_space_time_guidance import (
    TENSOR_ARGS, _Dataset, _DynamicField, _PlanningTask, _SpatialGuide, _moving_gate_world,
    _problem,
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


def test_request_local_safe_intervals_preserve_all_three_branch_preferences():
    times = np.arange(0.0, 5.1, 0.1)
    safe = np.ones((3, len(times)), dtype=bool)
    safe[1, (times > 1.0) & (times < 3.0)] = False
    intervals = [_safe_intervals(row, times, 0.05) for row in safe]
    arrivals = np.array([0.0, 2.0, 4.0])
    for preference in (0.0, -1.0, 1.0):
        expected = _reachable_branch(safe, arrivals, times, np.array([1, 1]), 3.0, 0.05, preference)
        cached = _reachable_branch(
            safe, arrivals, times, np.array([1, 1]), 3.0, 0.05, preference,
            interval_rows=intervals,
        )
        assert cached == expected


def test_vectorized_dp_matches_scalar_predecessor_and_tie_rules():
    def scalar_reference(safe, phase_times, times, minimum_steps, duration_min, margin, preference):
        intervals = [_safe_intervals(row, times, margin) for row in safe]
        if intervals[0][1][0] < 0:
            return None
        score = np.full(len(times), np.inf)
        score[0] = 0.0
        parents = np.full(safe.shape, -1, dtype=np.int32)
        for phase in range(1, len(safe)):
            previous_best = np.minimum.accumulate(score)
            previous_arg = np.zeros(len(times), dtype=np.int32)
            best = 0
            for index in range(len(times)):
                if score[index] < score[best]:
                    best = index
                previous_arg[index] = best
            next_score = np.full(len(times), np.inf)
            target = phase_times[phase] + preference * math.sin(math.pi * phase / (len(safe) - 1))
            for index in np.flatnonzero(intervals[phase][1] >= 0):
                predecessor = index - int(minimum_steps[phase - 1])
                if predecessor < 0 or not np.isfinite(previous_best[predecessor]):
                    continue
                next_score[index] = previous_best[predecessor] + (times[index] - target) ** 2 / len(safe)
                parents[phase, index] = previous_arg[predecessor]
            score = next_score
        score[times < duration_min - 1e-9] = np.inf
        final = int(np.argmin(score))
        if not np.isfinite(score[final]):
            return None
        selected = np.empty(len(safe), dtype=np.int32)
        selected[-1] = final
        for phase in range(len(safe) - 1, 0, -1):
            selected[phase - 1] = parents[phase, selected[phase]]
        if selected[0] != 0:
            return None
        return tuple(intervals[i][0][intervals[i][1][selected[i]]] for i in range(len(safe)))

    rng = np.random.default_rng(153)
    times = np.linspace(0.0, 6.0, 61)
    for _ in range(100):
        safe = rng.random((8, len(times))) > rng.uniform(0.0, 0.6)
        safe[0, 0] = True
        phase_times = np.linspace(0.0, 5.0, len(safe))
        minimum_steps = rng.integers(1, 4, size=len(safe) - 1)
        for preference in (0.0, -1.0, 1.0):
            expected = scalar_reference(safe, phase_times, times, minimum_steps, 2.0, 0.05, preference)
            actual = _reachable_branch(safe, phase_times, times, minimum_steps, 2.0, 0.05, preference)
            assert actual == expected


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


@pytest.mark.parametrize("backend", [refine_time_corridor, refine_time_corridor_batch])
def test_nonzero_endpoint_derivative_skips_optional_refinement_without_fault(backend):
    settings = SpaceTimeGuidanceSettings.from_mapping(dict(
        mode="phase5_joint", duration_min=1.0, duration_max=4.0,
        nominal_duration=2.0, corridor_a_enabled=True,
    ))
    guide = InferenceOnlySpaceTimeGuide(
        _SpatialGuide(), _PlanningTask(), _Dataset(),
        _DynamicField(_moving_gate_world()), settings, TENSOR_ARGS,
    )
    guide.planning_task.parametric_trajectory.q_vel_start = torch.tensor([0.1], **TENSOR_ARGS)
    guide.reset(1)
    paths = torch.zeros((1, 7, 1), **TENSOR_ARGS)
    original = guide.timing_spline.linear_control_points(2.0, batch_shape=(1,))
    refined, stats = backend(guide, paths, original, factorized=False)
    torch.testing.assert_close(refined, original)
    assert stats["skipped_nonzero_endpoint_derivatives"] is True


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
        experimental = parser.parse_args(base + [
            "--corridor-a", "--corridor-a-backend", "batch_time_table",
            "--corridor-a-dp-init", "--corridor-a-k-best", "4",
        ])
        assert experimental.corridor_a_dp_init and experimental.corridor_a_k_best == 4
        assert experimental.corridor_a_backend == "batch_time_table"


@pytest.mark.parametrize("chunk_size", [1, 2, 8])
def test_phase5_batch_exact_matches_serial_candidate_outputs(chunk_size):
    settings = SpaceTimeGuidanceSettings.from_mapping(dict(
        mode="phase5_joint", duration_min=1.0, duration_max=4.0,
        nominal_duration=2.0, corridor_a_enabled=True,
        corridor_a_phase_points=16, corridor_a_time_step_s=0.1,
        corridor_a_steps=3, corridor_a_chunk_size=chunk_size,
    ))
    guide = InferenceOnlySpaceTimeGuide(
        _SpatialGuide(), _PlanningTask(), _Dataset(),
        _DynamicField(_moving_gate_world()), settings, TENSOR_ARGS,
    )
    paths = torch.zeros((3, 7, 1), **TENSOR_ARGS)
    guide.reset(3)
    controls = guide.timing_spline.linear_control_points(2.0, batch_shape=(3,))
    controls[1, 3] += 0.1
    reference, reference_stats = refine_time_corridor(guide, paths, controls, factorized=False)
    batched, batch_stats = refine_time_corridor_batch(guide, paths, controls, factorized=False)
    torch.testing.assert_close(batched, reference, atol=2e-5, rtol=1e-4)
    for key in ("candidates", "attempted", "changed", "no_reachable_branch", "branches_tested"):
        assert batch_stats[key] == reference_stats[key]


@pytest.mark.parametrize("representation", ["c", "tau_r"])
@pytest.mark.parametrize("chunk_size", [1, 2])
def test_f1_batch_exact_matches_serial_six_latents(representation, chunk_size):
    guide, paths, latent = factorized_problem(representation)
    guide.settings = SpaceTimeGuidanceSettings.from_mapping(dict(
        duration_min=1.0, duration_max=4.0, nominal_duration=2.0,
        corridor_a_enabled=True, corridor_a_phase_points=16,
        corridor_a_time_step_s=0.1, corridor_a_steps=3,
        corridor_a_chunk_size=chunk_size,
    ))
    guide.cost_evaluator.settings = guide.settings
    reference, reference_stats = refine_time_corridor(guide, paths, latent, factorized=True)
    batched, batch_stats = refine_time_corridor_batch(guide, paths, latent, factorized=True)
    torch.testing.assert_close(batched, reference, atol=2e-5, rtol=1e-4)
    assert batched.shape == latent.shape == (2, 6)
    assert batch_stats["branches_tested"] == reference_stats["branches_tested"]


def test_time_distance_table_interpolation_and_domain_guard():
    world = _moving_gate_world()
    positions = torch.zeros((2, 5, 1, 3), **TENSOR_ARGS)
    positions[1, :, 0, 0] = 0.12
    grid = torch.linspace(0.0, 4.0, 41, **TENSOR_ARGS)
    table = _time_distance_table(world, positions, grid, candidate_chunk_size=1)
    arrivals = torch.tensor([[0.05, 0.35, 1.25, 2.65, 4.0],
                             [0.15, 0.45, 1.35, 2.75, 3.95]],
                            **TENSOR_ARGS).requires_grad_(True)
    interpolated = _interpolate_time_distance_table(table, grid, arrivals)
    assert interpolated.shape == (2, 5, 1)
    assert torch.isfinite(interpolated).all()
    interpolated.sum().backward()
    assert torch.isfinite(arrivals.grad).all()
    with pytest.raises(ValueError, match="outside"):
        _interpolate_time_distance_table(table, grid, arrivals.detach() + 0.2)


def test_exact_distance_override_preserves_mean_cvar_and_time_gradient():
    _, evaluator, controls, q, q_s, q_ss, positions = _problem()
    controls = controls.clone().requires_grad_(True)
    timing = evaluator.timing_spline.evaluate(controls, q=q, q_s=q_s, q_ss=q_ss)
    distance = evaluator.dynamic_world.minimum_signed_distance(
        positions, trajectory_times=timing.time_from_start,
    )
    direct, direct_breakdown, _ = evaluator(
        controls, q=q, q_s=q_s, q_ss=q_ss, collision_sphere_positions=positions,
    )
    overridden, override_breakdown, _ = evaluator(
        controls, q=q, q_s=q_s, q_ss=q_ss, collision_sphere_positions=positions,
        minimum_distance_override=distance,
    )
    torch.testing.assert_close(direct, overridden)
    for name in ("dynamic_collision", "dynamic_collision_mean", "dynamic_collision_cvar"):
        torch.testing.assert_close(direct_breakdown[name], override_breakdown[name])
    direct_gradient = torch.autograd.grad(direct.sum(), controls, retain_graph=True)[0]
    override_gradient = torch.autograd.grad(overridden.sum(), controls)[0]
    torch.testing.assert_close(direct_gradient, override_gradient)


def test_sphere_event_intervals_do_not_accept_exactly_blocked_grid_cells():
    world = _moving_gate_world()
    positions = torch.tensor([[[[0.0, 0.0, 0.0]], [[0.1, 0.0, 0.0]]]], **TENSOR_ARGS)
    margins = torch.tensor([0.08], **TENSOR_ARGS)
    grid = torch.linspace(0.0, 4.0, 401, **TENSOR_ARGS)
    analytic = _constant_sphere_interval_grid(world, positions, margins, grid)
    points = positions[:, :, None].expand(-1, -1, len(grid), -1, -1)
    points = points.reshape(2, len(grid), 1, 3)
    times = grid[None].expand(2, -1)
    exact = (world.minimum_signed_distance(points, trajectory_times=times)
             .reshape(1, 2, len(grid), 1) >= margins).all(dim=-1).numpy()
    assert not np.any(analytic & ~exact)
    assert np.any(~analytic)
    world.shape_code[0] = 1
    assert _constant_sphere_interval_grid(world, positions, margins, grid) is None


@pytest.mark.parametrize("representation", [None, "c", "tau_r"])
def test_optional_time_table_backend_keeps_original_or_exact_reranks(representation):
    if representation is None:
        settings = SpaceTimeGuidanceSettings.from_mapping(dict(
            mode="phase5_joint", duration_min=1.0, duration_max=4.0,
            nominal_duration=2.0, corridor_a_enabled=True,
            corridor_a_phase_points=16, corridor_a_time_step_s=0.1,
            corridor_a_steps=2, corridor_a_chunk_size=2,
            corridor_a_backend="batch_time_table",
        ))
        guide = InferenceOnlySpaceTimeGuide(
            _SpatialGuide(), _PlanningTask(), _Dataset(),
            _DynamicField(_moving_gate_world()), settings, TENSOR_ARGS,
        )
        paths = torch.zeros((2, 7, 1), **TENSOR_ARGS)
        guide.reset(2)
        initial = guide.timing_spline.linear_control_points(2.0, batch_shape=(2,))
    else:
        guide, paths, initial = factorized_problem(representation)
        guide.settings = SpaceTimeGuidanceSettings.from_mapping(dict(
            duration_min=1.0, duration_max=4.0, nominal_duration=2.0,
            corridor_a_enabled=True, corridor_a_phase_points=16,
            corridor_a_time_step_s=0.1, corridor_a_steps=2,
            corridor_a_chunk_size=2, corridor_a_backend="batch_time_table",
        ))
        guide.cost_evaluator.settings = guide.settings
    output, stats = refine_time_corridor_batch(
        guide, paths, initial, factorized=representation is not None,
    )
    assert output.shape == initial.shape
    assert torch.isfinite(output).all()
    assert stats["backend"] == "batch_time_table"
    assert stats["exact_rerank_branches"] > 0


def test_kbest_finds_distinct_window_sequences_with_finite_budget():
    times = np.linspace(0.0, 8.0, 81)
    safe = np.ones((3, len(times)), dtype=bool)
    safe[1, (times > 1.0) & (times < 1.5)] = False
    safe[1, (times > 2.0) & (times < 2.5)] = False
    safe[1, (times > 3.0) & (times < 3.5)] = False
    safe[1, (times > 4.0) & (times < 4.5)] = False
    branches = _reachable_branches_kbest(
        safe, np.array([0.0, 2.5, 6.0]), times,
        np.array([1, 1]), 5.0, 0.01, 4,
    )
    assert len(branches) == 4
    assert len({branch[1] for branch, _ in branches}) == 4
    for branch, schedule in branches:
        assert len(branch) == 3
        assert np.all(np.diff(schedule) > 0)


@pytest.mark.parametrize("representation", ["c", "tau_r"])
def test_f1_kbest_uses_same_six_latent_contract(representation):
    guide, paths, initial = factorized_problem(representation)
    guide.settings = SpaceTimeGuidanceSettings.from_mapping(dict(
        duration_min=1.0, duration_max=4.0, nominal_duration=2.0,
        corridor_a_enabled=True, corridor_a_phase_points=16,
        corridor_a_time_step_s=0.1, corridor_a_steps=2,
        corridor_a_chunk_size=2, corridor_a_backend="batch_exact",
        corridor_a_k_best=4,
    ))
    guide.cost_evaluator.settings = guide.settings
    output, stats, catalog = refine_time_corridor_batch(
        guide, paths, initial, factorized=True, return_branches=True,
    )
    assert output.shape == initial.shape == (2, 6)
    assert torch.isfinite(output).all()
    assert stats["branches_tested"] >= 1
    assert stats["optimized_unique_window_sequences"] >= 0
    for candidate, options in catalog.items():
        assert 0 <= candidate < len(paths)
        assert all(options[i]["objective"] <= options[i + 1]["objective"]
                   for i in range(len(options) - 1))
        assert all(len(option["window_signature"]) == 16 for option in options)


def test_selective_k_skips_time_invariant_invalid_paths_before_branch_search():
    guide, paths, initial = factorized_problem("c")
    guide.settings = SpaceTimeGuidanceSettings.from_mapping(dict(
        duration_min=1.0, duration_max=4.0, nominal_duration=2.0,
        corridor_a_enabled=True, corridor_a_phase_points=16,
        corridor_a_time_step_s=0.1, corridor_a_steps=2,
        corridor_a_chunk_size=2, corridor_a_backend="batch_exact",
        corridor_a_k_best=4, corridor_a_selective_k=True,
    ))
    guide.cost_evaluator.settings = guide.settings
    output, stats, catalog = refine_time_corridor_batch(
        guide, paths, initial, factorized=True, return_branches=True,
        eligible_candidate_mask=torch.zeros(len(initial), dtype=torch.bool),
    )
    torch.testing.assert_close(output, initial)
    assert stats["time_invariant_skipped"] > 0
    assert stats["branches_tested"] == 0
    assert stats["k_candidates_expanded"] == 0
    assert catalog == {}


def test_early_stop_requires_dense_checker_and_never_accepts_rejected_rows():
    guide, paths, initial = factorized_problem("c")
    guide.settings = SpaceTimeGuidanceSettings.from_mapping(dict(
        duration_min=1.0, duration_max=4.0, nominal_duration=2.0,
        corridor_a_enabled=True, corridor_a_phase_points=16,
        corridor_a_time_step_s=0.1, corridor_a_steps=8,
        corridor_a_chunk_size=2, corridor_a_backend="batch_exact",
        corridor_a_dp_init=True, corridor_a_early_stop=True,
    ))
    guide.cost_evaluator.settings = guide.settings
    with pytest.raises(ValueError, match="DenseCheck"):
        refine_time_corridor_batch(guide, paths, initial, factorized=True)
    output, stats = refine_time_corridor_batch(
        guide, paths, initial, factorized=True,
        validate_branch_rows=lambda indices, parameters: torch.zeros(
            len(indices), dtype=torch.bool, device=parameters.device,
        ),
    )
    assert output.shape == initial.shape
    assert stats["early_stopped_branches"] == 0
    assert stats["early_saved_iterations"] == 0


@pytest.mark.parametrize("representation", [None, "c", "tau_r"])
def test_dp_initializer_keeps_shape_and_reports_exact_acceptance(representation):
    if representation is None:
        settings = SpaceTimeGuidanceSettings.from_mapping(dict(
            mode="phase5_joint", duration_min=1.0, duration_max=4.0,
            nominal_duration=2.0, corridor_a_enabled=True,
            corridor_a_phase_points=16, corridor_a_time_step_s=0.1,
            corridor_a_steps=2, corridor_a_chunk_size=2,
            corridor_a_backend="batch_exact", corridor_a_dp_init=True,
        ))
        guide = InferenceOnlySpaceTimeGuide(
            _SpatialGuide(), _PlanningTask(), _Dataset(),
            _DynamicField(_moving_gate_world()), settings, TENSOR_ARGS,
        )
        paths = torch.zeros((2, 7, 1), **TENSOR_ARGS)
        guide.reset(2)
        initial = guide.timing_spline.linear_control_points(2.0, batch_shape=(2,))
    else:
        guide, paths, initial = factorized_problem(representation)
        guide.settings = SpaceTimeGuidanceSettings.from_mapping(dict(
            duration_min=1.0, duration_max=4.0, nominal_duration=2.0,
            corridor_a_enabled=True, corridor_a_phase_points=16,
            corridor_a_time_step_s=0.1, corridor_a_steps=2,
            corridor_a_chunk_size=2, corridor_a_backend="batch_exact",
            corridor_a_dp_init=True,
        ))
        guide.cost_evaluator.settings = guide.settings
    output, stats = refine_time_corridor_batch(
        guide, paths, initial, factorized=representation is not None,
    )
    assert output.shape == initial.shape
    assert torch.isfinite(output).all()
    assert stats["dp_initializers_accepted"] + stats["dp_initializers_rejected"] == stats["branches_tested"]
    assert stats["dp_fit_abs_error_s_mean"] is not None
    assert stats["dp_fit_abs_error_s_mean"] >= 0.0
