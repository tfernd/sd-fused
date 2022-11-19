from __future__ import annotations
from typing import Optional

from torch import Tensor

from ...external import Rearrange
from ...base import Module
from ...basic import GroupNorm, Linear
from .base_attention import BaseAttention


class SelfAttention(BaseAttention, Module):
    def __init__(
        self,
        *,
        in_features: int,
        head_features: Optional[int],
        num_groups: int,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.head_features = head_features
        self.num_groups = num_groups

        head_features = head_features or in_features
        num_heads = in_features // head_features

        # TODO These 4+1 operations can be fused
        self.group_norm = GroupNorm(num_groups, in_features)
        self.query = Linear(in_features)
        self.key = Linear(in_features)
        self.value = Linear(in_features)

        self.proj_attn = Linear(in_features)

        self.channel_last = Rearrange("B C H W -> B H W C")
        self.channel_first = Rearrange("B H W C -> B C H W")
        self.heads_to_batch = Rearrange("B H W (heads C) -> B heads H W C", heads=num_heads)
        self.heads_to_channel = Rearrange("B heads H W C -> B H W (heads C)")

    def __call__(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        xin = x
        x = self.group_norm(x)
        x = self.channel_last(x)

        # key, query, value projections
        q = self.heads_to_batch(self.query(x))
        k = self.heads_to_batch(self.key(x))
        v = self.heads_to_batch(self.value(x))

        x = self.attention(q, k, v)
        del q, k, v
        x = self.proj_attn(self.heads_to_channel(x))

        # output
        x = self.channel_first(x, H=H, W=W)

        return xin + x
