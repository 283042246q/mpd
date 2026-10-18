"""Joint noisy-space/noisy-time denoiser; KIN/WORLD share the same backbone."""
import math
import torch
from torch import nn

from mpd.models.layers.layers import ResidualTemporalBlock
from mpd.datasets.spacetime_legacy import open_uniform_knots
from mpd.models.diffusion_models.world_encoder import WorldEncoder
from mpd.timing_training.model import SinusoidalTimeEmbedding, cosine_beta_schedule


class ResidualFusion(nn.Module):
    """Preserve the ordered input through a projected residual path."""
    def __init__(self, input_width, width):
        super().__init__()
        self.projection = nn.Linear(input_width, width)
        self.residual = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                      nn.Mish(), nn.Linear(width, width))

    def forward(self, value):
        value = self.projection(value)
        return value + self.residual(value)


class GatedWorldAttention(nn.Module):
    """Pre-LN cross-attention and FFN, gated separately by diffusion/task context."""
    def __init__(self, width, world_width, condition_width, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.world_norm = nn.LayerNorm(world_width)
        self.attention = nn.MultiheadAttention(width, heads, kdim=world_width,
                                               vdim=world_width, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2*width), nn.Mish(), nn.Linear(2*width, width))
        self.gates = nn.Sequential(nn.Mish(), nn.Linear(condition_width, 2*width))
        # Zero only the gates: branch outputs remain nonzero so gates learn on step one.
        nn.init.zeros_(self.gates[-1].weight)
        nn.init.zeros_(self.gates[-1].bias)

    def forward(self, query, tokens, valid, condition):
        attention_gate, ffn_gate = self.gates(condition).chunk(2, dim=-1)
        world = self.world_norm(tokens)
        attended = self.attention(self.query_norm(query), world, world,
                                 key_padding_mask=~valid, need_weights=False)[0]
        query = query + attention_gate[:, None] * attended
        return query + ffn_gate[:, None] * self.ffn(self.ffn_norm(query))


class JointDualDenoiser(nn.Module):
    def __init__(self, variant="jd_kin", channels=(32, 64, 128, 256), timing_width=128,
                 joint_width=256, world_width=128, attention_heads=4, world_dropout=.1,
                 architecture_version="v1"):
        super().__init__()
        if variant not in ("jd_kin", "jd_world") or len(channels) != 4:
            raise ValueError("invalid JointDual variant or channel pyramid")
        if not 0 <= world_dropout < 1:
            raise ValueError("invalid world dropout")
        if architecture_version not in ("v1", "v2"):
            raise ValueError("unsupported JointDual architecture_version")
        if channels[0] % 2:
            raise ValueError("spatial position encoding requires an even input width")
        self.variant, self.world_dropout = variant, world_dropout
        # Missing version in historical checkpoints must retain the exact v1 graph.
        self.architecture_version = architecture_version
        w = timing_width
        self.time = nn.Sequential(SinusoidalTimeEmbedding(w), nn.Linear(w, w), nn.Mish(), nn.Linear(w, w))
        self.task = nn.Sequential(nn.Linear(19, w), nn.Mish(), nn.Linear(w, w))
        self.timing = nn.Sequential(nn.Linear(6 + 2*w, w), nn.Mish(), nn.Linear(w, w))
        self.input = nn.Linear(7, channels[0])
        if architecture_version == "v1":
            pos = torch.arange(16)[:, None] * torch.exp(torch.linspace(0, -math.log(10000), channels[0]//2))[None]
        else:
            # P_free[:15] maps to full P21 indices 3..17. The final variable ties
            # indices 18..20 and represents the endpoint. These are spline-order
            # coordinates, NOT q(s) samples or physical execution timestamps.
            knots = torch.as_tensor(open_uniform_knots(21, 5), dtype=torch.float32)
            phase = torch.cat((torch.stack([knots[i+1:i+6].mean() for i in range(3, 18)]),
                               torch.ones(1)))
            self.register_buffer("spatial_phase", phase)
            pos = phase[:, None] * (math.pi * 2**torch.linspace(0, 3, channels[0]//2))[None]
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
            self.world_encoder = WorldEncoder(world_width, encoding_version=architecture_version)
            if architecture_version == "v1":
                self.timing_attention = nn.MultiheadAttention(w, attention_heads, kdim=world_width,
                                                             vdim=world_width, batch_first=True)
                self.spatial_attention = nn.MultiheadAttention(channels[-1], attention_heads,
                    kdim=world_width, vdim=world_width, batch_first=True)
            else:
                self.timing_attention = GatedWorldAttention(w, world_width, 2*w, attention_heads)
                self.spatial_attention = GatedWorldAttention(channels[-1], world_width, 2*w, attention_heads)
        if architecture_version == "v1":
            self.fusion = nn.Sequential(nn.Linear(channels[-1]*2 + 3*w, joint_width), nn.Mish(),
                                        nn.Linear(joint_width, joint_width))
        else:
            self.fusion = ResidualFusion(channels[-1]*2 + 3*w, joint_width)
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
        t, c = self.time(step), self.task(task)  # t embeds diffusion k, never physical seconds.
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
            if self.architecture_version == "v1":
                hz = hz + self.timing_attention(hz[:, None], tokens, tokens,
                                                 key_padding_mask=~valid, need_weights=False)[0][:, 0]
            else:
                hz = self.timing_attention(hz[:, None], tokens, valid, torch.cat((t, c), -1))[:, 0]
        condition = torch.cat((t, c, hz), -1)
        x = (self.input(p) + self.position.to(p)).transpose(1, 2)
        skips = []
        for i, (one, two) in enumerate(self.encoder):
            x = two(one(x, condition), condition); skips.append(x)
            if i < 3:
                x = self.down[i](x)
        if self.variant == "jd_world":
            query = x.transpose(1, 2)
            if self.architecture_version == "v1":
                x = (query + self.spatial_attention(query, tokens, tokens,
                     key_padding_mask=~valid, need_weights=False)[0]).transpose(1, 2)
            else:
                x = self.spatial_attention(query, tokens, valid, torch.cat((t, c), -1)).transpose(1, 2)
        # Channel-major flatten retains both bottleneck positions; never mean-pool.
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
