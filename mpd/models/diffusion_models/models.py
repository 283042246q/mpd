import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpd.models.layers.layers import (
    Downsample1d,
    Conv1dBlock,
    Upsample1d,
    ResidualTemporalBlock,
    TimeEncoder,
    MLP,
    group_norm_n_groups,
    LinearAttention,
    PreNorm,
    Residual,
)
from mpd.models.layers.layers_attention import SpatialTransformer
from torch_robotics.torch_utils.torch_timer import TimerCUDA

import numpy as np

UNET_DIM_MULTS = {
    0: (1, 2, 4),
    1: (1, 2, 4, 8),
    2: (1, 2),
}


class TemporalUnet(nn.Module):

    def __init__(
        self,
        n_support_points=None,
        state_dim=None,
        unet_input_dim=32,
        dim_mults=(1, 2, 4, 8),
        time_emb_dim=32,
        self_attention=False,
        conditioning_embed_dim=4,
        conditioning_type=None,
        attention_num_heads=2,
        attention_dim_head=32,
        **kwargs,
    ):
        super().__init__()

        self.n_support_points = n_support_points
        self.state_dim = state_dim
        input_dim = state_dim

        # Conditioning
        if conditioning_type is None or conditioning_type == "None":
            conditioning_type = None
        elif conditioning_type == "concatenate":
            if self.state_dim < conditioning_embed_dim // 4:
                # Embed the state in a latent space HxF if the conditioning embedding is much larger than the state
                state_emb_dim = conditioning_embed_dim // 4
                self.state_encoder = MLP(state_dim, state_emb_dim, hidden_dim=state_emb_dim, n_layers=2, act="mish")
            else:
                state_emb_dim = state_dim
                self.state_encoder = nn.Identity()
            input_dim = state_emb_dim + conditioning_embed_dim
        elif conditioning_type == "attention":
            pass
        elif conditioning_type == "default":
            pass
        else:
            raise NotImplementedError
        self.conditioning_type = conditioning_type

        dims = [input_dim, *map(lambda m: unet_input_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.horizon_multiple = 2 ** max(0, len(in_out) - 1)
        padded_n_support_points = (
            (n_support_points + self.horizon_multiple - 1) // self.horizon_multiple
        ) * self.horizon_multiple
        network_n_support_points = padded_n_support_points
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        # Networks
        self.time_mlp = TimeEncoder(32, time_emb_dim)

        # conditioning dimension (time + context)
        cond_dim = time_emb_dim + (conditioning_embed_dim if conditioning_type == "default" else 0)

        # Unet
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, cond_dim, n_support_points=network_n_support_points),
                        ResidualTemporalBlock(dim_out, dim_out, cond_dim, n_support_points=network_n_support_points),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))) if self_attention else nn.Identity(),
                        (
                            SpatialTransformer(
                                dim_out,
                                attention_num_heads,
                                attention_dim_head,
                                depth=1,
                                context_dim=conditioning_embed_dim,
                            )
                            if conditioning_type == "attention"
                            else None
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                network_n_support_points = network_n_support_points // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, cond_dim, n_support_points=network_n_support_points
        )
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if self_attention else nn.Identity()
        self.mid_attention = (
            SpatialTransformer(
                mid_dim, attention_num_heads, attention_dim_head, depth=1, context_dim=conditioning_embed_dim
            )
            if conditioning_type == "attention"
            else nn.Identity()
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, cond_dim, n_support_points=network_n_support_points
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2, dim_in, cond_dim, n_support_points=network_n_support_points
                        ),
                        ResidualTemporalBlock(dim_in, dim_in, cond_dim, n_support_points=network_n_support_points),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))) if self_attention else nn.Identity(),
                        (
                            SpatialTransformer(
                                dim_in,
                                attention_num_heads,
                                attention_dim_head,
                                depth=1,
                                context_dim=conditioning_embed_dim,
                            )
                            if conditioning_type == "attention"
                            else None
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                network_n_support_points = network_n_support_points * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(unet_input_dim, unet_input_dim, kernel_size=5, n_groups=group_norm_n_groups(unet_input_dim)),
            nn.Conv1d(unet_input_dim, state_dim, 1),
        )

    def forward(self, x, time, context):
        """
        x : [ batch x horizon x state_dim ]
        context: [batch x context_dim]
        """
        b, horizon, d = x.shape
        if horizon != self.n_support_points:
            raise ValueError(f"expected horizon {self.n_support_points}, got {horizon}")

        t_emb = self.time_mlp(time)
        c_emb = t_emb
        if self.conditioning_type == "concatenate":
            x_emb = self.state_encoder(x)
            context = einops.repeat(context, "m n -> m h n", h=horizon)
            x = torch.cat((x_emb, context), dim=-1)
        elif self.conditioning_type == "attention":
            # reshape to keep the interface
            context = einops.rearrange(context, "b d -> b 1 d")
        elif self.conditioning_type == "default":
            c_emb = torch.cat((t_emb, context), dim=-1)

        # swap horizon and channels (state_dim)
        x = einops.rearrange(x, "b h c -> b c h")  # batch, horizon, channels (state_dim)
        pad_right = (-horizon) % self.horizon_multiple
        if pad_right:
            x = F.pad(x, (0, pad_right))

        skips = []
        for resnet, resnet2, attn_self, attn_conditioning, downsample in self.downs:
            x = resnet(x, c_emb)
            # if self.conditioning_type == 'attention':
            #     x = attention1(x, context=conditioning_emb)
            x = resnet2(x, c_emb)
            x = attn_self(x)
            if self.conditioning_type == "attention":
                x = attn_conditioning(x, context=context)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c_emb)
        x = self.mid_attn(x)
        if self.conditioning_type == "attention":
            x = self.mid_attention(x, context=context)
        x = self.mid_block2(x, c_emb)

        for resnet, resnet2, attn_self, attn_conditioning, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, c_emb)
            x = resnet2(x, c_emb)
            x = attn_self(x)
            if self.conditioning_type == "attention":
                x = attn_conditioning(x, context=context)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b c h -> b h c")
        x = x[:, :horizon]

        return x


class CrossArmMixer(nn.Module):
    """Attention between the two arms at each matching trajectory index."""

    def __init__(self, channels, num_heads=4, ff_multiplier=2, dropout=0.0):
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * ff_multiplier, channels),
        )

    def forward(self, x, batch_size):
        if x.shape[0] != batch_size * 2:
            raise ValueError("cross-arm mixer expects arms stacked as a [B*2, C, H] batch")
        tokens = einops.rearrange(x, "(b a) c h -> (b h) a c", b=batch_size, a=2)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.feed_forward(self.norm2(tokens))
        return einops.rearrange(tokens, "(b h) a c -> (b a) c h", b=batch_size, a=2)


class CrossArmTimeBottleneck(nn.Module):
    """Global attention over a pair token and all left/right time tokens."""

    def __init__(
        self,
        channels,
        context_token_dim,
        n_support_points,
        num_heads=4,
        num_layers=1,
        ff_multiplier=2,
        dropout=0.0,
    ):
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.n_support_points = n_support_points
        self.pair_projection = nn.Linear(context_token_dim, channels)
        self.arm_embedding = nn.Parameter(torch.empty(2, channels))
        self.time_embedding = nn.Parameter(torch.empty(n_support_points, channels))
        nn.init.normal_(self.arm_embedding, std=0.02)
        nn.init.normal_(self.time_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(channels),
        )

    def forward(self, x, pair_context, batch_size):
        if x.shape[0] != batch_size * 2:
            raise ValueError("global mixer expects arms stacked as a [B*2, C, H] batch")
        horizon = x.shape[-1]
        if horizon != self.n_support_points:
            raise ValueError(f"expected bottleneck horizon {self.n_support_points}, got {horizon}")
        arm_time = einops.rearrange(x, "(b a) c h -> b a h c", b=batch_size, a=2)
        arm_time = arm_time + self.arm_embedding[None, :, None, :]
        arm_time = arm_time + self.time_embedding[None, None, :, :]
        arm_time = einops.rearrange(arm_time, "b a h c -> b (a h) c")
        pair = self.pair_projection(pair_context).unsqueeze(1)
        mixed = self.transformer(torch.cat((pair, arm_time), dim=1))[:, 1:]
        return einops.rearrange(mixed, "b (a h) c -> (b a) c h", a=2, h=horizon)


class CoupledBimanualTemporalUnet(nn.Module):
    """Variant-D denoiser with shared arm streams and explicit coupling.

    The external contract remains [B, H, 14] -> [B, H, 14]. Internally the
    left and right seven-joint paths share all temporal convolution weights.
    Same-index arm attention is applied at every U-Net scale and global
    arm-by-time attention is applied at the bottleneck.
    """

    def __init__(
        self,
        n_support_points=None,
        state_dim=14,
        arm_state_dim=7,
        unet_input_dim=32,
        dim_mults=(1, 2, 4, 8),
        time_emb_dim=32,
        conditioning_embed_dim=None,
        context_token_dim=128,
        conditioning_type="default",
        attention_num_heads=4,
        attention_layers=1,
        attention_ff_multiplier=2,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        if state_dim != arm_state_dim * 2:
            raise ValueError("coupled Marvin U-Net requires state_dim=14 as two seven-joint arms")
        if conditioning_type != "default":
            raise ValueError("coupled Marvin U-Net supports conditioning_type='default' only")
        if conditioning_embed_dim != context_token_dim * 3:
            raise ValueError("variant D context must contain flattened [left, right, pair] tokens")

        self.n_support_points = n_support_points
        self.state_dim = state_dim
        self.arm_state_dim = arm_state_dim
        self.context_token_dim = context_token_dim
        self.conditioning_type = conditioning_type
        dims = [arm_state_dim, *map(lambda m: unet_input_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        for channels in dims[1:]:
            if channels % attention_num_heads:
                raise ValueError(
                    f"U-Net channel {channels} must be divisible by attention_num_heads={attention_num_heads}"
                )

        self.horizon_multiple = 2 ** max(0, len(in_out) - 1)
        padded_horizon = (
            (n_support_points + self.horizon_multiple - 1) // self.horizon_multiple
        ) * self.horizon_multiple
        network_horizon = padded_horizon
        print(f"[ models/coupled-bimanual-temporal ] Shared-arm channel dimensions: {in_out}")

        self.time_mlp = TimeEncoder(32, time_emb_dim)
        # Each arm sees its own token as well as the relation/pair token.
        cond_dim = time_emb_dim + context_token_dim * 2
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index >= num_resolutions - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, cond_dim, n_support_points=network_horizon),
                        ResidualTemporalBlock(dim_out, dim_out, cond_dim, n_support_points=network_horizon),
                        CrossArmMixer(
                            dim_out,
                            num_heads=attention_num_heads,
                            ff_multiplier=attention_ff_multiplier,
                            dropout=attention_dropout,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
            if not is_last:
                network_horizon //= 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim, n_support_points=network_horizon)
        self.global_mixer = CrossArmTimeBottleneck(
            mid_dim,
            context_token_dim,
            network_horizon,
            num_heads=attention_num_heads,
            num_layers=attention_layers,
            ff_multiplier=attention_ff_multiplier,
            dropout=attention_dropout,
        )
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim, n_support_points=network_horizon)

        for index, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = index >= num_resolutions - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_out * 2, dim_in, cond_dim, n_support_points=network_horizon),
                        ResidualTemporalBlock(dim_in, dim_in, cond_dim, n_support_points=network_horizon),
                        CrossArmMixer(
                            dim_in,
                            num_heads=attention_num_heads,
                            ff_multiplier=attention_ff_multiplier,
                            dropout=attention_dropout,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
            if not is_last:
                network_horizon *= 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(
                unet_input_dim,
                unet_input_dim,
                kernel_size=5,
                n_groups=group_norm_n_groups(unet_input_dim),
            ),
            nn.Conv1d(unet_input_dim, arm_state_dim, 1),
        )

    def forward(self, x, time, context):
        if x.ndim != 3 or x.shape[-1] != self.state_dim:
            raise ValueError(f"expected x with shape [B, H, {self.state_dim}]")
        batch_size, horizon, _ = x.shape
        if horizon != self.n_support_points:
            raise ValueError(f"expected horizon {self.n_support_points}, got {horizon}")
        if context.ndim != 2 or context.shape != (batch_size, self.context_token_dim * 3):
            raise ValueError(
                f"expected flattened [left, right, pair] context [B, {self.context_token_dim * 3}]"
            )

        context_tokens = context.reshape(batch_size, 3, self.context_token_dim)
        arm_context = context_tokens[:, :2]
        pair_context = context_tokens[:, 2]
        time_context = self.time_mlp(time)
        time_context = time_context[:, None, :].expand(-1, 2, -1)
        pair_per_arm = pair_context[:, None, :].expand(-1, 2, -1)
        conditioning = torch.cat((time_context, arm_context, pair_per_arm), dim=-1)
        conditioning = conditioning.reshape(batch_size * 2, -1)

        x = x.reshape(batch_size, horizon, 2, self.arm_state_dim)
        x = einops.rearrange(x, "b h a d -> (b a) d h")
        pad_right = (-horizon) % self.horizon_multiple
        if pad_right:
            x = F.pad(x, (0, pad_right))

        skips = []
        for resnet, resnet2, arm_mixer, downsample in self.downs:
            x = resnet(x, conditioning)
            x = resnet2(x, conditioning)
            x = arm_mixer(x, batch_size)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, conditioning)
        x = self.global_mixer(x, pair_context, batch_size)
        x = self.mid_block2(x, conditioning)

        for resnet, resnet2, arm_mixer, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, conditioning)
            x = resnet2(x, conditioning)
            x = arm_mixer(x, batch_size)
            x = upsample(x)

        x = self.final_conv(x)
        x = x[..., :horizon]
        return einops.rearrange(x, "(b a) d h -> b h (a d)", b=batch_size, a=2)


if __name__ == "__main__":
    import torch
    import time

    device = "cuda:0"

    batch_size = 1000
    n_support_points = 16

    model = TemporalUnet(
        n_support_points=n_support_points,
        state_dim=7,
        unet_input_dim=32,
        dim_mults=UNET_DIM_MULTS[1],
        time_emb_dim=32,
        self_attention=True,
        conditioning_embed_dim=128,
        conditioning_type="default",
    )
    model.to(device)

    x = torch.randn(batch_size, n_support_points, 7, device=device)  # batch_size x horizon x state_dim
    t = torch.randn(batch_size, device=device)  # batch_size x horizon
    context = torch.randn(batch_size, 128, device=device)  # batch_size x context_dim

    t_elapsed_l = []
    for i in range(20):
        with TimerCUDA() as t_forward:
            output = model(x, t, context)
        # print("Time taken (CUDA):", t_forward.elapsed)
        t_elapsed_l.append(t_forward.elapsed)
    print("Time taken average (CUDA):", np.mean(t_elapsed_l[10:]))
