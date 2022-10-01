from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ..activation import SiLU
from ..base import Linear


class TimestepEmbedding(nn.Module):
    def __init__(
        self, *, channel: int, time_embed_dim: int, use_silu: bool = True,
    ) -> None:
        super().__init__()

        self.channel = channel
        self.time_embed_dim = time_embed_dim
        self.use_silu = use_silu

        self.linear_1 = Linear(channel, time_embed_dim)
        self.linear_2 = Linear(time_embed_dim, time_embed_dim)

        self.act = SiLU() if use_silu else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_1(x)
        x = self.act(x)

        return self.linear_2(x)
