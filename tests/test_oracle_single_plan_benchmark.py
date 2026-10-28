"""Protocol tests: common motion, temporal gradients and swept base exclusion."""
import numpy as np
import pytest
import torch
import yaml

from mpd.inference.dynamic_collision import DynamicWorldError
from mpd.inference.jointdual_world import OracleForecastWorld
from mpd.inference.prescribed_world import PrescribedDynamicWorld
from scripts.benchmarks.oracle_sampling import arrival_phases, balanced_choices, base_sweep_clear, sample_world
from scripts.benchmarks.oracle_single_plan import DEFAULT_CONFIG, verify_suite


def test_arrival_quota_and_phase_semantics():
    rng = np.random.default_rng(10)
    choices = balanced_choices(60, list(range(6)), [.1, .15, .2, .25, .2, .1], rng)
    assert np.bincount(choices).tolist() == [6, 9, 12, 15, 12, 6]
    same = arrival_phases(5, "simultaneous", rng)
    assert np.ptp(same) == 0
    sequential = arrival_phases(5, "sequential", rng)
    assert np.min(np.diff(np.sort(sequential))) >= .06 - 1e-12
    grouped = arrival_phases(5, "grouped", rng)
    assert len(np.unique(grouped)) == 2
    with pytest.raises(ValueError):
        arrival_phases(5, "sequential", rng, gap=.3)


@pytest.mark.parametrize("count", range(6))
def test_sampler_matches_reference_phase_and_saves_exact_event_knots(count):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    phase = np.linspace(0, 1, 64)
    centers = np.zeros((64, 4, 3))
    centers[:, :, 0] = phase[:, None]
    centers[:, :, 2] = .7
    world, meta = sample_world(centers, [2, 3], count, "sequential", np.random.default_rng(7), config)
    assert world["positions"].shape == (len(world["times"]), count, 3)
    assert meta["inflation"] == "none"
    for i, (fraction, event, anchor) in enumerate(zip(meta["phases"], meta["event_times"], meta["anchors"])):
        assert event == pytest.approx(fraction * 10)
        assert anchor[0] == pytest.approx(fraction)
        at = np.flatnonzero(world["times"] == event)
        assert len(at) == 1
        np.testing.assert_allclose(world["positions"][at[0], i], anchor, atol=1e-12)


def test_base_exclusion_checks_segments_not_only_saved_frames():
    world = dict(shapes=np.array([0]), sizes=np.array([[.05, 0, 0]]),
                 positions=np.array([[[-1., 0, 0]], [[1., 0, 0]]]))
    assert not base_sweep_clear(world, [-.1] * 3, [.1] * 3, .005)
    world["positions"][:, 0, 2] = 1.
    assert base_sweep_clear(world, [-.1] * 3, [.1] * 3, .005)


@pytest.mark.parametrize("shape,size", [(0, [.1, 0, 0]), (1, [.1, .2, .3]), (2, [.1, .2, 0])])
def test_prescribed_world_agrees_with_jointdual_oracle_and_has_time_gradient(shape, size):
    world = PrescribedDynamicWorld(8, trajectory_duration_s=14.,
        tensor_args=dict(device="cpu", dtype=torch.float64), capacity_buckets_enabled=True,
        fused_reduction_enabled=True, shape_grouping_enabled=True)
    times = np.array([0., 7., 14.])
    positions = np.array([[[0., 0, 0]], [[1., 0, 0]], [[0., 0, 0]]])
    world.load_forecast(times, positions, [shape], [size])
    oracle = OracleForecastWorld(times, positions, np.array([shape]), np.array([size]))
    qtime = torch.tensor([[1., 3., 9.]], dtype=torch.float64, requires_grad=True)
    points = torch.tensor([[[[.9, .4, .2]], [[.9, .4, .2]], [[.9, .4, .2]]]],
                          dtype=torch.float64, requires_grad=True)
    actual = world.minimum_signed_distance(points, qtime)
    expected = oracle.minimum_signed_distance(points, qtime)
    torch.testing.assert_close(actual, expected)
    gp, gt = torch.autograd.grad(actual.sum(), (points, qtime))
    assert torch.isfinite(gp).all() and gt.abs().sum() > 0
    fixed = world.minimum_signed_distance(points)
    implicit_times = torch.linspace(0, 14, 3, dtype=torch.float64)[None]
    torch.testing.assert_close(fixed, oracle.minimum_signed_distance(points, implicit_times))
    assert not world.base_inflation.any() and not world.horizon_inflation_rate.any()
    a, b, c = world.features()
    x, y, z = oracle.features()
    np.testing.assert_allclose(a, x); np.testing.assert_array_equal(b, y); assert c == z
    with pytest.raises(DynamicWorldError, match="coverage"):
        world.minimum_signed_distance(points, qtime.detach() + 20, validate_times=False)


def test_empty_world_and_capacity_padding():
    world = PrescribedDynamicWorld(16, trajectory_duration_s=10.,
        tensor_args=dict(device="cpu", dtype=torch.float32), capacity_buckets_enabled=True,
        fused_reduction_enabled=True, shape_grouping_enabled=True)
    world.load_forecast([0., 14.], np.empty((2, 0, 3)), [], np.empty((0, 3)))
    assert torch.isinf(world.minimum_signed_distance(torch.zeros(2, 4, 3, 3))).all()
    world.load_forecast([0., 14.], np.ones((2, 5, 3)), [0] * 5, [[.1, 0, 0]] * 5)
    result = world.minimum_signed_distance(torch.zeros(2, 4, 3, 3))
    assert result.shape == (2, 4, 3) and torch.isfinite(result).all()


def test_suite_refuses_mutated_manifest(tmp_path):
    from scripts.runtime.infer_once import _canonical_json_sha256
    import json
    suite = dict(schema="oracle_single_plan_suite_v1", status="ready", cases=[])
    suite["suite_hash"] = _canonical_json_sha256(suite)
    (tmp_path / "suite.json").write_text(json.dumps(suite))
    assert verify_suite(tmp_path)["status"] == "ready"
    suite["cases"].append({"id": "tampered"})
    (tmp_path / "suite.json").write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="changed"):
        verify_suite(tmp_path)


def test_online_budget_contract_requires_explicit_benchmark_opt_in():
    from dotmap import DotMap
    from scripts.runtime.infer_once import _validate_runtime_config, ConfigurationError
    config = DotMap(yaml.safe_load((DEFAULT_CONFIG.parents[2] / "inference/cfgs/config_EnvWarehouse-RobotPanda-runtime.yaml").read_text()))
    config.n_trajectory_samples = 4
    with pytest.raises(ConfigurationError, match="n_trajectory_samples"):
        _validate_runtime_config(config)
    _validate_runtime_config(config, allow_benchmark_budgets=True)
    config.n_trajectory_samples = 0
    with pytest.raises(ConfigurationError):
        _validate_runtime_config(config, allow_benchmark_budgets=True)
