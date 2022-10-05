from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ..base import LayerNorm, InPlace
from .cross_attention import CrossAttention
from .feed_forward import FeedForward

class BasicTransformer(InPlace, nn.Module):
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
        out = self.attn1(self.norm1(x))
        x = out.add_(x) if self.inplace else out + x

        out = self.attn2(self.norm2(x), context=context)
        x = out.add_(x) if self.inplace else out + x
        del context

        out = self.ff(self.norm3(x))
        x = out.add_(x) if self.inplace else out + x
        del out

        return x
