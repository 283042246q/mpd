"""Explicit clean-estimate DDIM sampling for independent space/timing priors.

Existing spatial/timing model classes and their public samplers are unchanged.
All physical guidance operates on clean predictions, never noisy states.
"""
from dataclasses import dataclass
import math
import time

import torch


@dataclass(frozen=True)
class FactorizedSettings:
    method: str = "f1"
    space_steps: int = 32
    timing_steps: int = 100
    space_guide_fraction: float = .3
    timing_guide_fraction: float = .3
    space_lr: float = .01
    timing_lr: float = .02
    refinement_steps: int = 5
    weak_dynamic_scale: float = .2
    eta: float = 0.
    alternating_rounds: int = 1
    timing_steps_per_space_step: int = 1

    def __post_init__(self):
        if self.method not in ("f1", "f2", "f3"):
            raise ValueError("method must be f1, f2 or f3")
        if min(self.space_steps, self.timing_steps, self.alternating_rounds) < 1:
            raise ValueError("step counts and alternating rounds must be positive")
        if self.refinement_steps < 0 or self.timing_steps_per_space_step not in (1, 2):
            raise ValueError("invalid refinement count or timing:space step ratio")
        if not all(0. < v <= 1. for v in (self.space_guide_fraction, self.timing_guide_fraction)):
            raise ValueError("guide fractions must lie in (0,1]")
        if not all(math.isfinite(v) and v >= 0 for v in (self.space_lr, self.timing_lr, self.weak_dynamic_scale)):
            raise ValueError("learning rates and weak dynamic scale must be finite and nonnegative")
        if not 0. <= self.eta <= 1.:
            raise ValueError("DDIM eta must lie in [0,1]")


def schedule(total, steps, *, start=None):
    start = total - 1 if start is None else int(start)
    if not 0 <= start < total or not 1 <= steps <= start + 1:
        raise ValueError("reverse schedule exceeds available diffusion levels")
    levels = torch.linspace(start, 0, steps).round().long().tolist()
    return list(zip(levels, levels[1:] + [-1]))


def forward_noise(clean, alpha, noise):
    return alpha.sqrt() * clean + (1. - alpha).sqrt() * noise


def ddim_update(noisy, clean, alpha, next_alpha, *, eta=0., noise=None):
    epsilon = (noisy - alpha.sqrt() * clean) / (1. - alpha).sqrt().clamp_min(1e-12)
    variance = ((1. - next_alpha) / (1. - alpha) * (1. - alpha / next_alpha)).clamp_min(0.)
    sigma = float(eta) * variance.sqrt()
    result = next_alpha.sqrt() * clean + (1. - next_alpha - sigma.square()).clamp_min(0.).sqrt() * epsilon
    if eta:
        result = result + sigma * (torch.randn_like(noisy) if noise is None else noise)
    return result


class FactorizedSampler:
    """The guide implements condition(P), nominal(P), refine(P,z,active,weak).

    The spatial prior need only expose predict_x_recon and alphas_cumprod.
    Timing state z has shape [B,6] for either checkpoint representation.
    """
    def __init__(self, space_model, timing_model, guide, settings):
        self.space_model, self.timing_model, self.guide = space_model, timing_model, guide
        self.settings = settings
        self.statistics = []

    def _project(self, x):
        x = x.clamp(-1., 1.)
        for index, value in self.hard_conds.items():
            x[:, index, :] = value
        return x.detach()

    def _record(self, branch, step, stage, guided):
        self.statistics.append(dict(branch=branch, timestep=int(step), stage=stage, guided=bool(guided)))

    def _space_step(self, x, pair, context, *, z=None, guided=False, stage="space", weak=False):
        step, following = pair
        t = torch.full((len(x),), step, dtype=torch.long, device=x.device)
        clean = self._project(self.space_model.predict_x_recon(x, t, context))
        if guided:
            current_z = self.guide.nominal(clean) if weak else z
            clean, _ = self.guide.refine(clean, current_z, active="space", weak=weak)
            clean = self._project(clean)
        alpha = self.space_model.alphas_cumprod[step]
        next_alpha = x.new_tensor(1.) if following < 0 else self.space_model.alphas_cumprod[following]
        out = ddim_update(x, clean, alpha, next_alpha, eta=self.settings.eta)
        # Hard constraints apply to noisy states too, without clipping their noise.
        for index, value in self.hard_conds.items():
            out[:, index, :] = value
        self._record("space", step, stage, guided)
        return out.detach(), clean

    def _timing_step(self, z, pair, p, *, guided=False, stage="timing"):
        step, following = pair
        t = torch.full((len(z),), step, dtype=torch.long, device=z.device)
        path = self.guide.condition(p.detach())
        eps = self.timing_model.denoiser(z, t, path)
        clean = self.timing_model.predict_clean_from_noise(z, t, eps).detach()
        if guided:
            _, clean = self.guide.refine(p.detach(), clean, active="timing")
        alpha = self.timing_model.alphas_cumprod[step]
        next_alpha = z.new_tensor(1.) if following < 0 else self.timing_model.alphas_cumprod[following]
        out = ddim_update(z, clean, alpha, next_alpha, eta=self.settings.eta)
        self._record("timing", step, stage, guided)
        return out.detach(), clean.detach()

    def _full_separated(self, p, context, ps, ts):
        for pair in ps:
            p, _ = self._space_step(p, pair, context, guided=pair[0] <= self.p_gate, weak=True)
        z = torch.randn((len(p), 6), device=p.device, dtype=p.dtype)
        for pair in ts:
            z, _ = self._timing_step(z, pair, p, guided=pair[0] <= self.t_gate)
        return p, z

    def _partial_alternating(self, p, z, context, ps, ts):
        """F2: preserve the other clean block during each partial reverse."""
        cfg = self.settings
        pn, tn = len(self.space_model.alphas_cumprod), len(self.timing_model.alphas_cumprod)
        psteps = sum(pair[0] <= self.p_gate for pair in ps)
        tsteps = sum(pair[0] <= self.t_gate for pair in ts)
        partial_p = schedule(pn, psteps, start=self.p_gate)
        partial_t = schedule(tn, tsteps, start=self.t_gate)
        for iteration in range(cfg.alternating_rounds):
            p = forward_noise(p, self.space_model.alphas_cumprod[self.p_gate], torch.randn_like(p))
            for index, value in self.hard_conds.items():
                p[:, index, :] = value
            for pair in partial_p:
                p, _ = self._space_step(p, pair, context, z=z, guided=True,
                                         stage=f"f2_space_{iteration}")
            z = forward_noise(z, self.timing_model.alphas_cumprod[self.t_gate], torch.randn_like(z))
            for pair in partial_t:
                z, _ = self._timing_step(z, pair, p, guided=True, stage=f"f2_timing_{iteration}")
        return p, z

    def _fine_alternating(self, p, context, ps, ts):
        """F3: independently warm both chains, then cross-guide clean estimates.

        The low timing schedule is explicitly aligned to the spatial schedule;
        it uses distinct training timesteps, never repeats a reverse edge.
        """
        low_p = [pair for pair in ps if pair[0] <= self.p_gate]
        high_p = [pair for pair in ps if pair[0] > self.p_gate]
        ratio = self.settings.timing_steps_per_space_step
        low_t_count = len(low_p) * ratio
        tn = len(self.timing_model.alphas_cumprod)
        if low_t_count > self.t_gate + 1:
            raise ValueError("F3 timing low-noise levels cannot accommodate the requested step ratio; reduce space steps or increase timing guide fraction")
        low_t = schedule(tn, low_t_count, start=self.t_gate)
        # Split a COMPLETE timing chain at its low-noise boundary. Rebuild the
        # high->low bridge so z really has the level used by the first low step.
        high_levels = [pair[0] for pair in ts if pair[0] > self.t_gate]
        high_t = list(zip(high_levels, high_levels[1:] + [self.t_gate]))
        p_clean = None
        for pair in high_p:
            p, p_clean = self._space_step(p, pair, context, stage="f3_space_high")
        # rho=1 leaves no high steps: obtain a conditioning clean estimate, but
        # do not advance the spatial noisy state before its first scheduled edge.
        if p_clean is None:
            t = torch.full((len(p),), low_p[0][0], dtype=torch.long, device=p.device)
            p_clean = self._project(self.space_model.predict_x_recon(p, t, context))
            self._record("space", low_p[0][0], "f3_condition_bootstrap", False)
        z = p.new_empty((len(p), 6)).normal_()
        z_clean = None
        for pair in high_t:
            z, z_clean = self._timing_step(z, pair, p_clean, stage="f3_timing_high")
        if z_clean is None:
            t = torch.full((len(z),), self.t_gate, dtype=torch.long, device=z.device)
            eps = self.timing_model.denoiser(z, t, self.guide.condition(p_clean))
            z_clean = self.timing_model.predict_clean_from_noise(z, t, eps)
            self._record("timing", self.t_gate, "f3_condition_bootstrap", False)
        for index, pair in enumerate(low_p):
            # Each clean condition was estimated at a *high* level until the
            # first low update. Do not cross-guide space with that timing yet.
            p, p_clean = self._space_step(p, pair, context, z=z_clean,
                guided=True, weak=index == 0, stage="f3_space_low")
            for timing_pair in low_t[index * ratio:(index + 1) * ratio]:
                z, z_clean = self._timing_step(z, timing_pair, p_clean,
                    guided=True, stage="f3_timing_low")
        return p, z

    @torch.no_grad()
    def sample(self, shape, context, hard_conds, *, device, dtype=torch.float32):
        self.statistics = []
        self.hard_conds = hard_conds
        cfg = self.settings
        pn, tn = len(self.space_model.alphas_cumprod), len(self.timing_model.alphas_cumprod)
        self.p_gate = min(pn - 1, max(0, math.ceil(cfg.space_guide_fraction * pn) - 1))
        self.t_gate = min(tn - 1, max(0, math.ceil(cfg.timing_guide_fraction * tn) - 1))
        ps, ts = schedule(pn, cfg.space_steps), schedule(tn, cfg.timing_steps)
        p = torch.randn(shape, device=device, dtype=dtype)
        for index, value in hard_conds.items():
            p[:, index, :] = value
        if cfg.method == "f3":
            p, z = self._fine_alternating(p, context, ps, ts)
        else:
            p, z = self._full_separated(p, context, ps, ts)
            if cfg.method == "f2":
                p, z = self._partial_alternating(p, z, context, ps, ts)
        for _ in range(cfg.refinement_steps):
            p, z = self.guide.refine(p, z, active="joint")
            p = self._project(p)
        if not torch.isfinite(p).all() or not torch.isfinite(z).all():
            raise ValueError("factorized sampler produced nonfinite states")
        return p.detach(), z.detach()


class FactorizedModelAdapter:
    """Instance-local adapter for GenerativeOptimizationPlanner.run_inference."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __getattr__(self, name):
        return getattr(self.sampler.space_model, name)

    def run_inference(self, *, context_d, hard_conds, n_samples, horizon,
                      return_chain=False, return_chain_x_recon=False, results_ns=None, **kwargs):
        if return_chain_x_recon:
            raise ValueError("factorized adapter exposes final paths, not legacy reconstruction chains")
        context = {k: v.unsqueeze(0).expand(n_samples, *v.shape) for k, v in context_d.items()}
        conditions = {k: v.unsqueeze(0).expand(n_samples, *v.shape) for k, v in hard_conds.items()}
        device = self.sampler.timing_model.betas.device
        started = time.perf_counter()
        p, z = self.sampler.sample((n_samples, horizon, self.sampler.guide.dataset.control_points_dim[-1]),
            context, conditions, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if results_ns is not None:
            results_ns.t_generator += time.perf_counter() - started
        self.sampler.guide.timing_control_points = z
        return p.unsqueeze(0) if return_chain else p
