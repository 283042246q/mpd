"""Permutation-invariant world tokens, with distinct KNOWN/EMPTY/NULL states."""
import math
import torch
from torch import nn


class WorldEncoder(nn.Module):
    def __init__(self, width=128, encoding_version="v1"):
        super().__init__()
        if encoding_version not in ("v1", "v2"):
            raise ValueError("unsupported world encoding_version")
        self.encoding_version = encoding_version
        self.encoder = nn.Sequential(nn.Linear(28 if encoding_version == "v1" else 29, width),
                                     nn.Mish(), nn.Linear(width, width))
        self.state = nn.Embedding(3, width)
        if encoding_version == "v1":
            self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 8)))
        else:
            # Fixed physical units: position/size in m, velocity in m/s. Time is
            # relative to plan start, scaled by the fixed WORLD horizon, not T.
            self.register_buffer("time_scale_seconds", torch.tensor(14.))
            self.register_buffer("frequencies", math.pi * 2**torch.linspace(0, 3, 8))

    def encode_time(self, seconds):
        if self.encoding_version == "v1":
            angles = seconds * self.frequencies.to(seconds)
            return torch.cat((angles.sin(), angles.cos()), -1)
        phase = seconds / self.time_scale_seconds.to(seconds)
        angles = phase * self.frequencies.to(seconds)
        # Raw normalized time distinguishes endpoints independently of periodic PE.
        return torch.cat((phase, angles.sin(), angles.cos()), -1)

    def forward(self, features, mask, state):
        if features.shape[-1] != 13 or mask.shape != features.shape[:2]:
            raise ValueError("world feature/mask dimensions disagree")
        tokens = self.encoder(torch.cat((features[..., :12], self.encode_time(features[..., -1:])), -1))
        # A valid state token avoids all-masked attention, even for empty worlds.
        tokens = torch.cat((self.state(state)[:, None], tokens), 1)
        valid = mask & (state[:, None] == 0)
        valid = torch.cat((torch.ones_like(state[:, None], dtype=torch.bool), valid), 1)
        return tokens, valid
