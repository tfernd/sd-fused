from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange
from einops import rearrange

from ..base import Conv2d, GroupNorm
from .basic_transformer import BasicTransformer


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_heads: int,
        dim_head: int,
        depth: int,
        num_groups: int,
        context_dim: Optional[int],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.depth = depth
        self.num_groups = num_groups
        self.context_dim = context_dim

        inner_dim = num_heads * dim_head

        self.norm = GroupNorm(num_groups, in_channels)

        self.proj_in = Conv2d(in_channels, inner_dim)
        self.proj_out = Conv2d(inner_dim, in_channels)

        self.transformer_blocks = nn.ModuleList()
        for _ in range(depth):
            self.transformer_blocks.append(
                BasicTransformer(
                    dim=inner_dim,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    context_dim=context_dim,
                )
            )

        self.channel_last_and_spatial_join = Rearrange("B C H W -> B (H W) C")

    def __call__(
        self, x: Tensor, *, context: Optional[Tensor] = None
    ) -> Tensor:
        B, C, H, W = x.shape

        xin = x

        x = self.norm(x)
        x = self.proj_in(x)
        x = self.channel_last_and_spatial_join(x)

        for block in self.transformer_blocks:
            assert isinstance(block, BasicTransformer)

            x = block(x, context=context)
        del context

        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)

        return xin + self.proj_out(x)
