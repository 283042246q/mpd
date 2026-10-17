"""Permutation-invariant world tokens, with distinct KNOWN/EMPTY/NULL states."""
import math
import torch
from torch import nn


class WorldEncoder(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28, width), nn.Mish(), nn.Linear(width, width))
        self.state = nn.Embedding(3, width)
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 8)))

    def forward(self, features, mask, state):
        if features.shape[-1] != 13 or mask.shape != features.shape[:2]:
            raise ValueError("world feature/mask dimensions disagree")
        time = features[..., -1:] * self.frequencies.to(features)
        tokens = self.encoder(torch.cat((features[..., :12], time.sin(), time.cos()), -1))
        # A valid state token avoids all-masked attention, even for empty worlds.
        tokens = torch.cat((self.state(state)[:, None], tokens), 1)
        valid = mask & (state[:, None] == 0)
        valid = torch.cat((torch.ones_like(state[:, None], dtype=torch.bool), valid), 1)
        return tokens, valid
