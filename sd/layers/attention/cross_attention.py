from __future__ import annotations
from typing import Optional

import math

import torch
import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange

from ...utils import softmax_
from ..base import Linear


class CrossAttention(nn.Module):
    split_attention_chunks: Optional[int] = None

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

        self.scale = math.pow(dim_head, -1 / 4)

        self.to_q = Linear(query_dim, inner_dim, bias=False)
        self.to_k = Linear(context_dim, inner_dim, bias=False)
        self.to_v = Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(Linear(inner_dim, query_dim))

        self.heads_to_batch = Rearrange(
            "B T (heads C) -> (B heads) T C", heads=num_heads
        )
        # pre-transpose to avoid transposing afterwards
        self.heads_to_batch_t = Rearrange(
            "B T (heads C) -> (B heads) C T", heads=num_heads
        )
        self.heads_to_channel = Rearrange(
            "(B heads) T C -> B T (heads C)", heads=num_heads
        )

    def forward(
        self, x: Tensor, *, context: Optional[Tensor] = None,
    ) -> Tensor:
        device = x.device
        B, T, C = x.shape

        context = context if context is not None else x

        # key, query, value projections
        q = self.heads_to_batch(self.to_q(x).mul_(self.scale))
        del x
        k = self.heads_to_batch_t(self.to_k(context).mul_(self.scale))
        v = self.heads_to_batch(self.to_v(context))
        del context

        # attention score
        if self.split_attention_chunks is None:
            attn = softmax_(q @ k, dim=-1)
            del q, k

            # projection
            x = self.heads_to_channel(attn @ v)
            del attn, v

            return self.to_out(x)

        # split-attention score
        shape = (*q.shape[0:2], v.shape[2])
        x = torch.zeros(shape, device=q.device, dtype=q.dtype)
        for i in range(0, k.shape[0], self.split_attention_chunks):
            idx = slice(i, i + self.split_attention_chunks)

            attn = softmax_(q[idx] @ k[idx], dim=-1)
            x[idx] = attn @ v[idx]
            del attn

            # TODO delete q[idx], k[idx], v[idx]
        x = self.heads_to_channel(x)

        return self.to_out(x)
