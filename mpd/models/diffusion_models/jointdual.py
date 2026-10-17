"""Joint noisy-space/noisy-time denoiser; KIN/WORLD share the same backbone."""
import math
import torch
from torch import nn

from mpd.models.layers.layers import ResidualTemporalBlock
from mpd.models.diffusion_models.world_encoder import WorldEncoder
from mpd.timing_training.model import SinusoidalTimeEmbedding, cosine_beta_schedule


class JointDualDenoiser(nn.Module):
    def __init__(self, variant="jd_kin", channels=(32, 64, 128, 256), timing_width=128,
                 joint_width=256, world_width=128, attention_heads=4, world_dropout=.1):
        super().__init__()
        if variant not in ("jd_kin", "jd_world") or len(channels) != 4:
            raise ValueError("invalid JointDual variant or channel pyramid")
        if not 0 <= world_dropout < 1:
            raise ValueError("invalid world dropout")
        self.variant, self.world_dropout = variant, world_dropout
        w = timing_width
        self.time = nn.Sequential(SinusoidalTimeEmbedding(w), nn.Linear(w, w), nn.Mish(), nn.Linear(w, w))
        self.task = nn.Sequential(nn.Linear(19, w), nn.Mish(), nn.Linear(w, w))
        self.timing = nn.Sequential(nn.Linear(6 + 2*w, w), nn.Mish(), nn.Linear(w, w))
        self.input = nn.Linear(7, channels[0])
        pos = torch.arange(16)[:, None] * torch.exp(torch.linspace(0, -math.log(10000), channels[0]//2))[None]
        self.register_buffer("position", torch.cat((pos.sin(), pos.cos()), -1))
        self.encoder, self.down = nn.ModuleList(), nn.ModuleList()
        previous = channels[0]
        for i, ch in enumerate(channels):
            self.encoder.append(nn.ModuleList([ResidualTemporalBlock(previous, ch, 3*w, 16 // 2**i),
                                               ResidualTemporalBlock(ch, ch, 3*w, 16 // 2**i)]))
            if i < 3:
                self.down.append(nn.Conv1d(ch, ch, 4, 2, 1))
            previous = ch
        if variant == "jd_world":
            self.world_encoder = WorldEncoder(world_width)
            self.timing_attention = nn.MultiheadAttention(w, attention_heads, kdim=world_width,
                                                         vdim=world_width, batch_first=True)
            self.spatial_attention = nn.MultiheadAttention(channels[-1], attention_heads,
                kdim=world_width, vdim=world_width, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(channels[-1]*2 + 3*w, joint_width), nn.Mish(),
                                    nn.Linear(joint_width, joint_width))
        self.decoder, self.up = nn.ModuleList(), nn.ModuleList()
        previous = channels[-1]
        for i, ch in enumerate(reversed(channels[:-1])):
            self.up.append(nn.ConvTranspose1d(previous, ch, 4, 2, 1))
            self.decoder.append(ResidualTemporalBlock(ch*2, ch, 3*w+joint_width, 4 * 2**i))
            previous = ch
        self.spatial_head = nn.Conv1d(channels[0], 7, 1)
        self.timing_head = nn.Sequential(nn.Linear(joint_width+w, w), nn.Mish(), nn.Linear(w, 6))
        self.timing_residual = nn.Linear(w, 6)

    def encode_world(self, features, mask, state):
        if self.variant == "jd_kin":
            return None
        return self.world_encoder(features, mask, state)

    def forward(self, p, z, step, task, world_features=None, world_mask=None, world_state=None,
                encoded_world=None):
        if p.shape[1:] != (16, 7) or z.shape != (len(p), 6) or task.shape != (len(p), 19):
            raise ValueError("JointDual expects P16, z6, and task19")
        t, c = self.time(step), self.task(task)
        hz = self.timing(torch.cat((z, t, c), -1))
        if self.variant == "jd_world":
            if encoded_world is None:
                if world_features is None:
                    raise ValueError("WORLD requires explicit KNOWN/EMPTY/NULL world")
                state = world_state.clone()
                if self.training and self.world_dropout:
                    state[torch.rand(len(p), device=p.device) < self.world_dropout] = 2
                encoded_world = self.encode_world(world_features, world_mask, state)
            tokens, valid = encoded_world
            hz = hz + self.timing_attention(hz[:, None], tokens, tokens,
                                             key_padding_mask=~valid, need_weights=False)[0][:, 0]
        condition = torch.cat((t, c, hz), -1)
        x = (self.input(p) + self.position.to(p)).transpose(1, 2)
        skips = []
        for i, (one, two) in enumerate(self.encoder):
            x = two(one(x, condition), condition); skips.append(x)
            if i < 3:
                x = self.down[i](x)
        if self.variant == "jd_world":
            query = x.transpose(1, 2)
            x = (query + self.spatial_attention(query, tokens, tokens,
                 key_padding_mask=~valid, need_weights=False)[0]).transpose(1, 2)
        joint = self.fusion(torch.cat((x.flatten(1), hz, t, c), -1))
        decoder_condition = torch.cat((condition, joint), -1)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips[:-1])):
            x = block(torch.cat((up(x), skip), 1), decoder_condition)
        return self.spatial_head(x).transpose(1, 2), self.timing_head(torch.cat((joint, hz), -1)) + self.timing_residual(hz)


class JointDiffusion(nn.Module):
    def __init__(self, denoiser, num_diffusion_steps=100):
        super().__init__()
        self.denoiser = denoiser
        self.num_diffusion_steps = num_diffusion_steps
        betas = cosine_beta_schedule(num_diffusion_steps)
        self.register_buffer("alphas_cumprod", torch.cumprod(1 - betas.float(), 0))

    def training_loss(self, batch, *, generator=None):
        p, z = batch["p"], batch["z"]
        step = torch.randint(self.num_diffusion_steps, (len(p),), device=p.device, generator=generator)
        ep = torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=generator)
        ez = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
        a = self.alphas_cumprod[step]
        pk = a[:, None, None].sqrt()*p + (1-a[:, None, None]).sqrt()*ep
        zk = a[:, None].sqrt()*z + (1-a[:, None]).sqrt()*ez
        pp, pz = self.denoiser(pk, zk, step, batch["task"],
            batch.get("world_features"), batch.get("world_mask"), batch.get("world_state"))
        lp, lz = (pp-ep).square().mean(), (pz-ez).square().mean()
        return lp + lz, dict(p=lp.detach(), z=lz.detach(), tau=(pz[:, 0]-ez[:, 0]).square().mean().detach(),
                             r=(pz[:, 1:]-ez[:, 1:]).square().mean().detach())
