from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from .cross_attention import CrossAttention
from .feed_forward import FeedForward


class BasicTransformer(nn.Module):
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

        self.ff = FeedForward(dim, mult=4)

    def __call__(
        self,
        x: Tensor,
        *,
        context: Optional[Tensor] = None,
        context_weights: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.attn1(x)
        x = self.attn2(x, context=context, context_weights=context_weights)
        del context
        x = self.ff(x)

        return x
