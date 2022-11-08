from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange

from ....utils.typing import Literal
from ...base import Linear, LayerNorm
from .fn import attention, ChunkType


class CrossAttention(nn.Module):
    attention_chunks: Optional[int | Literal["auto"]] = None
    chunk_type: Optional[ChunkType] = None
    use_flash_attention: bool = False
    tome_r: Optional[int | float] = None

    def __init__(
        self,
        *,
        query_features: int,
        context_features: Optional[int],
        head_features: int,
        num_heads: int,
    ) -> None:
        super().__init__()

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

        self.heads_to_batch = Rearrange(
            "B T (heads C) -> (B heads) T C", heads=num_heads
        )

        self.heads_to_channel = Rearrange(
            "(B heads) T C -> B T (heads C)", heads=num_heads
        )

    def __call__(
        self,
        x: Tensor,
        *,
        context: Optional[Tensor] = None,
        context_weights: Optional[Tensor] = None,
    ) -> Tensor:

        xin = x

        x = self.norm(x)
        context = context if context is not None else x

        # key, query, value projections
        q = self.heads_to_batch(self.to_q(x))
        k = self.heads_to_batch(self.to_k(context))
        v = self.heads_to_batch(self.to_v(context))
        del x, context

        x = attention(
            q,
            k,
            v,
            chunks=self.attention_chunks,
            chunk_type=self.chunk_type,
            weights=context_weights,
            use_flash_attention=self.use_flash_attention,
            tome_r=self.tome_r,
        )
        del q, k, v
        x = self.heads_to_channel(x)

        return xin + self.to_out(x)
