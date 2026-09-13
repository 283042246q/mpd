from types import SimpleNamespace
import pytest
import torch

from mpd.inference.factorized_sampler import FactorizedSampler, FactorizedSettings, ddim_update, forward_noise, schedule
from mpd.timing_training.model import TimingDiffusion, TimingDenoiser


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
        FactorizedSettings(method=method, space_steps=5, timing_steps=10, **kwargs))


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
