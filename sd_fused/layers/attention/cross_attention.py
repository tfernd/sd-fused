from __future__ import annotations
from typing import Optional

import math

import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange

from ..base import Linear, LayerNorm
from .attention import attention


class CrossAttention(nn.Module):
    attention_chunks: Optional[int] = None

    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: Optional[int],
        num_heads: int,
        dim_head: int,
    ) -> None:
        super().__init__()

        context_dim = context_dim or query_dim
        inner_dim = dim_head * num_heads

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.dim_head = dim_head

        # TODO These 4+1 operations can be fused
        self.norm = LayerNorm(query_dim)
        self.to_q = Linear(query_dim, inner_dim, bias=False)
        self.to_k = Linear(context_dim, inner_dim, bias=False)
        self.to_v = Linear(context_dim, inner_dim, bias=False)
        self.scale = math.pow(dim_head, -1 / 4)

        self.to_out = Linear(inner_dim, query_dim)

        self.heads_to_batch = Rearrange(
            "B T (heads C) -> (B heads) T C", heads=num_heads
        )

        self.heads_to_channel = Rearrange(
            "(B heads) T C -> B T (heads C)", heads=num_heads
        )

    def __call__(
        self, x: Tensor, *, context: Optional[Tensor] = None,
    ) -> Tensor:

        xin = x

        x = self.norm(x)

        # key, query, value projections
        q = self.heads_to_batch(self.to_q(x))
        context = context if context is not None else x
        k = self.heads_to_batch(self.to_k(context))
        v = self.heads_to_batch(self.to_v(context))
        del x, context

        q = q * self.scale
        k = k * self.scale

        # attention score
        x = attention(q, k, v, self.attention_chunks)
        del q, k, v
        x = self.heads_to_channel(x)

        return xin + self.to_out(x)