from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ..base import LayerNorm
from .cross_attention import CrossAttention
from .feed_forward import FeedForward


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        dim_head: int,
        context_dim: Optional[int],
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.context_dim = context_dim

        self.attn1 = CrossAttention(
            query_dim=dim,
            num_heads=num_heads,
            dim_head=dim_head,
            context_dim=None,
        )
        self.attn2 = CrossAttention(
            query_dim=dim,
            num_heads=num_heads,
            dim_head=dim_head,
            context_dim=context_dim,
        )

        self.ff = FeedForward(dim, dim_out=None, mult=4)

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.norm3 = LayerNorm(dim)

    def forward(
        self, x: Tensor, *, context: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.attn1(self.norm1(x)).add_(x)
        x = self.attn2(self.norm2(x), context=context).add_(x)
        del context
        x = self.ff(self.norm3(x)).add_(x)

        return x
