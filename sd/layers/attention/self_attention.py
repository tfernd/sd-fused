from __future__ import annotations
from typing import Optional

import math

import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange
from einops import rearrange

from ..base import GroupNorm, Linear
from .attention import attention


class SelfAttention(nn.Module):
    attention_chunks: Optional[int] = None  # ! TODO Auto?

    def __init__(
        self,
        *,
        num_channels: int,
        num_head_channels: Optional[int],
        num_groups: int,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_head_channels = num_head_channels
        self.num_groups = num_groups

        num_head_channels = num_head_channels or num_channels
        num_heads = num_channels // num_head_channels

        self.scale = math.pow(num_head_channels, -1 / 4)

        self.group_norm = GroupNorm(num_groups, num_channels)

        self.query = Linear(num_channels)
        self.key = Linear(num_channels)
        self.value = Linear(num_channels)

        self.proj_attn = Linear(num_channels)

        self.channel_last_and_spatial_join = Rearrange("B C H W -> B (H W) C")
        self.heads_to_batch = Rearrange(
            "B HW (heads C) -> (B heads) HW C", heads=num_heads,
        )
        self.heads_to_channel = Rearrange(
            "(B heads) HW C -> B HW (heads C)", heads=num_heads
        )

    def __call__(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        xin = x

        # normalization
        # TODO group norm and query,... into single module?
        x = self.group_norm(x)

        x = self.channel_last_and_spatial_join(x)

        # key, query, value projections
        q = self.heads_to_batch(self.query(x))
        k = self.heads_to_batch(self.key(x))
        v = self.heads_to_batch(self.value(x))
        del x

        # scale
        q = q * self.scale
        k = k * self.scale

        # attention score
        x = attention(q, k, v, self.attention_chunks)
        del q, k, v
        x = self.proj_attn(self.heads_to_channel(x))

        # output
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)

        return xin + x
