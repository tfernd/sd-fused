from __future__ import annotations
from typing import Optional

from torch import Tensor

from ....layers.basic import Conv2d
from ...external import Rearrange
from ...base import Module, ModuleList
from ..basic import GroupNormConv2d
from .basic_transformer import BasicTransformer


class SpatialTransformer(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_heads: int,
        head_features: int,
        depth: int,
        num_groups: int,
        context_features: Optional[int],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_features = head_features
        self.depth = depth
        self.num_groups = num_groups
        self.context_features = context_features

        inner_dim = num_heads * head_features

        self.proj_in = GroupNormConv2d(num_groups, in_channels, inner_dim)
        self.proj_out = Conv2d(inner_dim, in_channels)

        self.transformer_blocks = ModuleList()
        for _ in range(depth):
            self.transformer_blocks.append(
                BasicTransformer(
                    in_features=inner_dim,
                    num_heads=num_heads,
                    head_features=head_features,
                    context_features=context_features,
                )
            )

        self.channel_last = Rearrange("B C H W -> B H W C")
        self.channel_first = Rearrange("B H W C -> B C H W")

    def __call__(
        self,
        x: Tensor,
        *,
        context: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        B, C, H, W = x.shape

        xin = x

        x = self.proj_in(x)
        x = self.channel_last(x)

        for block in self.transformer_blocks:
            assert isinstance(block, BasicTransformer)

            x = block(x, context=context, weights=weights)
        x = self.channel_first(x, H=H, W=W)

        return xin + self.proj_out(x)
