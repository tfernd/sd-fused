from __future__ import annotations
from typing import Optional

from torch import Tensor

from ...external import Rearrange
from ...base import Module
from ...basic import LayerNorm, Linear
from .base_attention import BaseAttention


class CrossAttention(BaseAttention, Module):
    def __init__(
        self,
        *,
        query_features: int,
        context_features: Optional[int],
        head_features: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        is_cross_attention = context_features is not None

        context_features = context_features or query_features
        inner_dim = head_features * num_heads

        self.query_features = query_features
        self.context_features = context_features
        self.head_features = head_features
        self.num_heads = num_heads

        # TODO These 4+1 operations can be fused
        self.norm = LayerNorm(query_features)
        self.to_q = Linear(query_features, inner_dim, bias=False)
        self.to_k = Linear(context_features, inner_dim, bias=False)
        self.to_v = Linear(context_features, inner_dim, bias=False)

        self.to_out = Linear(inner_dim, query_features)

        self.heads_to_batch = Rearrange("B H W (heads C) -> B heads H W C", heads=num_heads)
        self.heads_to_batch2 = self.heads_to_batch
        if is_cross_attention:
            self.heads_to_batch2 = Rearrange("B T (heads C) -> B heads T C", heads=num_heads)

        self.heads_to_channel = Rearrange("B heads H W C -> B H W (heads C)")

    def __call__(
        self,
        x: Tensor,
        *,
        context: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:

        xin = x
        x = self.norm(x)
        context = context if context is not None else x

        # key, query, value projections
        q = self.heads_to_batch(self.to_q(x))
        k = self.heads_to_batch2(self.to_k(context))
        v = self.heads_to_batch2(self.to_v(context))

        x = self.attention(q, k, v, weights)
        del q, k, v
        x = self.heads_to_channel(x)

        return xin + self.to_out(x)
