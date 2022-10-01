from __future__ import annotations
from typing import Optional

import math

import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange
from einops import rearrange

from ...utils import softmax_
from ..base import GroupNorm, Linear


class AttentionBlock(nn.Module):
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
        self.separate_heads = Rearrange(
            "B HW (heads C) -> B heads HW C", heads=num_heads,
        )
        self.join_heads = Rearrange("B heads HW C -> B HW (heads C)")

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        xin = x

        # normalization
        x = self.group_norm(x)
        x = self.channel_last_and_spatial_join(x)

        # key, query, value projections
        q = self.separate_heads(self.query(x)).mul_(self.scale)
        k = self.separate_heads(self.key(x)).mul_(self.scale)
        v = self.separate_heads(self.value(x))
        del x

        attn = softmax_(q @ k.transpose(-1, -2), dim=-1)
        del q, k

        # projection
        x = self.proj_attn(self.join_heads(attn @ v))
        del attn, v

        # output
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)
        x += xin
        del xin

        return x
