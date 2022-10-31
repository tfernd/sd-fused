from __future__ import annotations
from typing import Optional

import math

import torch
from torch import Tensor

from einops import repeat

from ...utils import softmax

try:
    from xformers.ops import memory_efficient_attention  # type: ignore
except ImportError:
    memory_efficient_attention = None


def attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
    *,
    chunks: Optional[int] = None,
    weights: Optional[Tensor] = None,
    flash_attention: bool = False,
) -> Tensor:
    assert q.ndim == k.ndim == v.ndim == 3
    assert q.shape[0] == k.shape[0] == v.shape[0]
    assert q.shape[2] == k.shape[2] == v.shape[2]
    assert k.shape[1] == v.shape[1]

    Tl, C = k.shape[1:]

    if flash_attention and memory_efficient_attention is not None:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out = memory_efficient_attention(q, k, v)

        return out

    # scale q, v
    scale = math.pow(C, -1 / 4)
    q = q * scale
    k = k * scale

    if weights is not None:
        assert weights.ndim == 2
        assert weights.shape[1] == Tl

        # qkv has heads combined into the batch
        num_heads = q.shape[0] // weights.shape[0]
        weights = repeat(weights, "b t -> (b h) 1 t", h=num_heads)

    # pre-transpose
    k = k.transpose(1, 2)

    if chunks is None:
        attn = softmax(q @ k, dim=-1)
        del q, k

        if weights is not None:
            attn *= weights

        return attn @ v

    # TODO auto, see how much free memory is available?

    # split-attention score
    out = torch.empty_like(q)
    for i in range(0, len(k), chunks):
        s = slice(i, i + chunks)

        attn = softmax(q[s] @ k[s], dim=-1)
        if weights is not None:
            attn *= weights[s]

        out[s] = attn @ v[s]
        del attn

    return out
