"""Synchronous P/z DDIM with one authoritative timing state."""
from dataclasses import dataclass
import math
import time
import torch

from mpd.inference.factorized_sampler import schedule, ddim_update


@dataclass(frozen=True)
class JointDualSettings:
    steps: int = 32
    guide_fraction: float = .3
    refinement_steps: int = 5
    space_lr: float = .01
    timing_lr: float = .02
    space_clip: float = 2.
    timing_clip: float = 1.
    cfg_scale: float = 1.
    eta: float = 0.

    def __post_init__(self):
        if self.steps < 1 or self.refinement_steps < 0 or not 0 <= self.guide_fraction <= 1:
            raise ValueError("invalid joint sampling budget")
        values = (self.space_lr, self.timing_lr, self.space_clip, self.timing_clip, self.cfg_scale, self.eta)
        if not all(math.isfinite(v) and v >= 0 for v in values) or self.eta > 1:
            raise ValueError("invalid joint guidance settings")


@dataclass
class JointSampleResult:
    p: torch.Tensor
    z: torch.Tensor
    statistics: list


class JointDualSampler:
    def __init__(self, model, settings=None):
        self.model = model
        self.settings = settings or JointDualSettings()

    @torch.no_grad()
    def sample(self, task, *, world_features=None, world_mask=None, world_state=None,
               guide=None, generator=None, initial_state=None, encoded_world=None):
        if initial_state is not None:
            raise ValueError("joint warm-start is not implemented; provide neither P nor z")
        settings, model = self.settings, self.model
        model.eval()
        n = len(task); args = dict(device=task.device, dtype=task.dtype, generator=generator)
        p, z = torch.randn((n,16,7), **args), torch.randn((n,6), **args)
        def now():
            if task.is_cuda:torch.cuda.synchronize(task.device)
            return time.perf_counter()
        world_started=now()
        encoded = encoded_world if encoded_world is not None else model.denoiser.encode_world(world_features, world_mask, world_state)
        null = None
        if encoded is not None and settings.cfg_scale != 1:
            null = model.denoiser.encode_world(world_features, world_mask, torch.full_like(world_state,2))
        statistics = [dict(stage="world_encoding",elapsed_s=now()-world_started,denoiser_evaluations=0)]
        pairs = schedule(model.num_diffusion_steps, settings.steps)
        for index, (step, following) in enumerate(pairs):
            k = torch.full((n,),step,device=p.device,dtype=torch.long)
            denoiser_started=now()
            ep,ez = model.denoiser(p,z,k,task,encoded_world=encoded)
            evaluations = 1
            if null is not None:
                up,uz = model.denoiser(p,z,k,task,encoded_world=null)
                ep,ez = up+settings.cfg_scale*(ep-up), uz+settings.cfg_scale*(ez-uz)
                evaluations += 1
            denoiser_s=now()-denoiser_started
            a = model.alphas_cumprod[step]
            p0 = ((p-(1-a).sqrt()*ep)/a.sqrt()).clamp(-1,1)
            z0 = (z-(1-a).sqrt()*ez)/a.sqrt()
            guided = guide is not None and index >= len(pairs)-math.ceil(len(pairs)*settings.guide_fraction)
            guide_started=now()
            if guided:
                p0,z0 = guide.refine(p0,z0,settings)
            guide_s=now()-guide_started if guided else 0.
            next_a = model.alphas_cumprod[following] if following >= 0 else a.new_tensor(1.)
            p = ddim_update(p,p0,a,next_a,eta=settings.eta,
                            noise=torch.randn(p.shape,**args) if settings.eta else None)
            z = ddim_update(z,z0,a,next_a,eta=settings.eta,
                            noise=torch.randn(z.shape,**args) if settings.eta else None)
            if not torch.isfinite(p).all() or not torch.isfinite(z).all():
                raise ValueError("nonfinite joint sampler state")
            statistics.append(dict(step=step,following=following,guided=guided,denoiser_evaluations=evaluations,
                                   denoiser_s=denoiser_s,guide_s=guide_s))
        p = p.clamp(-1,1)
        if guide is not None:
            for _ in range(settings.refinement_steps):
                guide_started=now()
                p,z = guide.refine(p,z,settings)
                statistics.append(dict(stage="final_refinement",guided=True,denoiser_evaluations=0,guide_s=now()-guide_started))
        return JointSampleResult(p.detach(),z.detach(),statistics)
