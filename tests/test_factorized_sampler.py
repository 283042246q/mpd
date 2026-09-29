import pytest
import torch
from dataclasses import replace

from mpd.inference.factorized_sampler import FactorizedSampler, FactorizedSettings, ddim_update, forward_noise, schedule
from mpd.timing_training.model import TimingDiffusion, TimingDenoiser


def test_measured_default_optimizations_remain_overridable():
    assert [(FactorizedSettings(method=method).timing_grad_only,
             FactorizedSettings(method=method).fixed_path_cache,
             FactorizedSettings(method=method).path_encoding_cache)
            for method in ("f1", "f2", "f3")] == [
                (True, True, True), (True, True, True), (True, True, True)]
    assert FactorizedSettings(method="f1", timing_grad_only=False,
                              fixed_path_cache=False, path_encoding_cache=False).timing_grad_only is False


class Space:
    def __init__(self, alphas):
        self.alphas_cumprod = alphas
    def predict_x_recon(self, x, t, context):
        return x.tanh() * .5


class Guide:
    def __init__(self):
        self.calls = []
        self.conditions = []
    def condition(self, p):
        self.conditions.append(p.clone())
        return p
    def nominal(self, p):
        return p.new_zeros(len(p), 6)
    def refine(self, p, z, *, active, weak=False):
        self.calls.append((active, weak, p.clone(), z.clone()))
        return (p - .01 if active in ("space", "joint") else p), (z - .01 if active in ("timing", "joint") else z)


def sampler(method="f1", **kwargs):
    model = TimingDiffusion(TimingDenoiser(dof=2, num_spatial_control_points=8,
        spatial_degree=3, num_phase_points=8, path_width=8, path_embedding_dim=8,
        hidden_dim=8, time_embedding_dim=8, num_residual_blocks=1), num_diffusion_steps=20)
    guide = Guide()
    return FactorizedSampler(Space(model.alphas_cumprod), model, guide,
        FactorizedSettings(method=method, space_steps=5, timing_steps=10,
                           timing_grad_only=False, fixed_path_cache=False,
                           path_encoding_cache=False, **kwargs))


def test_ddim_oracle_partial_noise_roundtrip():
    clean, noise = torch.randn(3, 6), torch.randn(3, 6)
    alpha = torch.tensor(.3)
    noisy = forward_noise(clean, alpha, noise)
    recovered = ddim_update(noisy, clean, alpha, torch.tensor(1.))
    torch.testing.assert_close(recovered, clean)
    middle = ddim_update(noisy, clean, alpha, torch.tensor(.7))
    torch.testing.assert_close(middle, forward_noise(clean, torch.tensor(.7), noise))
    with pytest.raises(ValueError):
        schedule(10, 11)


def test_f1_has_fixed_path_timing_stage_and_clean_guidance_gate():
    s = sampler(refinement_steps=0)
    p, z = s.sample((3, 8, 2), {}, {0: torch.zeros(3, 2)}, device="cpu")
    assert p.shape == (3, 8, 2) and z.shape == (3, 6)
    assert torch.isfinite(z).all()
    assert all(torch.equal(c, p) for c in s.guide.conditions)
    assert torch.equal(p[:, 0], torch.zeros(3, 2))
    assert len(s.statistics) == 15
    for record in s.statistics:
        assert record["guided"] == (record["timestep"] <= 5)
    assert all(weak for active, weak, _, _ in s.guide.calls if active == "space")


def test_f1_eta_zero_is_seed_reproducible():
    s = sampler(refinement_steps=2)
    torch.manual_seed(10)
    first = s.sample((2, 8, 2), {}, {}, device="cpu")
    torch.manual_seed(10)
    second = s.sample((2, 8, 2), {}, {}, device="cpu")
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_f2_partial_blocks_freeze_other_state_and_begin_at_forward_noise_level():
    s = sampler("f2", refinement_steps=0, alternating_rounds=2)
    s.sample((2, 8, 2), {}, {}, device="cpu")
    for iteration in range(2):
        pr = [r for r in s.statistics if r["stage"] == f"f2_space_{iteration}"]
        tr = [r for r in s.statistics if r["stage"] == f"f2_timing_{iteration}"]
        assert pr[0]["timestep"] == tr[0]["timestep"] == 5
        assert pr[-1]["timestep"] == tr[-1]["timestep"] == 0
        assert all(r["guided"] for r in pr + tr)
    full_space_calls = sum(r["guided"] for r in s.statistics if r["stage"] == "space")
    full_timing_calls = sum(r["guided"] for r in s.statistics if r["stage"] == "timing")
    calls = s.guide.calls[full_space_calls + full_timing_calls:]
    for start in (0, 5):
        a, b = calls[start:start + 2]
        assert a[0] == b[0] == "space" and not a[1] and not b[1]
        torch.testing.assert_close(a[3], b[3])
        timing_calls = calls[start + 2:start + 5]
        assert all(c[0] == "timing" for c in timing_calls)
        assert all(torch.equal(c[2], timing_calls[0][2]) for c in timing_calls)


@pytest.mark.parametrize("ratio", [1, 2])
def test_f3_alternates_unique_low_steps_and_conditions_on_updated_clean_path(ratio):
    s = sampler("f3", refinement_steps=0, timing_steps_per_space_step=ratio)
    s.sample((2, 8, 2), {}, {}, device="cpu")
    low = [r for r in s.statistics if r["stage"].endswith("_low")]
    assert [r["branch"] for r in low] == (["space"] + ["timing"] * ratio) * 2
    for branch in ("space", "timing"):
        steps = [r["timestep"] for r in s.statistics if r["branch"] == branch]
        assert steps == sorted(set(steps), reverse=True)
    # First low space uses nominal timing; all following space steps use the
    # low-noise timing estimate updated by the preceding timing block.
    space_calls = [c for c in s.guide.calls if c[0] == "space"]
    assert [c[1] for c in space_calls] == [True, False]
    assert not torch.equal(s.guide.conditions[-1], s.guide.conditions[0])
    for r in low:
        assert r["timestep"] <= 5


@pytest.mark.parametrize("method,expected_calls", [("f1", 1), ("f2", 2), ("f3", 3)])
def test_path_encoding_cache_is_block_scoped_and_keeps_sample(method, expected_calls):
    s = sampler(method, refinement_steps=0)
    encoder = s.timing_model.denoiser.path_encoder
    original_forward = encoder.forward
    calls = []
    def counted(path):
        calls.append(path.data_ptr())
        return original_forward(path)
    encoder.forward = counted
    torch.manual_seed(17)
    baseline = s.sample((2, 8, 2), {}, {}, device="cpu")
    assert len(calls) > expected_calls
    calls.clear()
    s.settings = replace(s.settings, path_encoding_cache=True)
    torch.manual_seed(17)
    cached = s.sample((2, 8, 2), {}, {}, device="cpu")
    for actual, expected in zip(cached, baseline):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(calls) == expected_calls


@pytest.mark.parametrize("method,expected_blocks", [("f1", 1), ("f2", 2), ("f3", 2)])
def test_fixed_path_cache_uses_guided_timing_blocks_only(method, expected_blocks):
    s = sampler(method, refinement_steps=0)
    prepared = []
    cleared = []
    s.guide.prepare_fixed_path = lambda p: prepared.append(p.clone())
    s.guide.clear_fixed_path = lambda: cleared.append(True)
    s.settings = replace(s.settings, fixed_path_cache=True, timing_grad_only=True)
    s.sample((2, 8, 2), {}, {}, device="cpu")
    assert len(prepared) == len(cleared) == expected_blocks
    if method == "f3":
        assert not torch.equal(prepared[0], prepared[1])
