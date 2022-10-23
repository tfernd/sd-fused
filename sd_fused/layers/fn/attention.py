from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from einops import repeat

from ...utils import softmax


def attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
    *,
    chunks: Optional[int] = None,
    weights: Optional[Tensor] = None
) -> Tensor:
    assert q.ndim == k.ndim == v.ndim == 3
    assert q.shape[0] == k.shape[0] == v.shape[0]
    assert q.shape[2] == k.shape[2] == v.shape[2]
    assert k.shape[1] == v.shape[1]

    if weights is not None:
        assert weights.ndim == 2
        assert weights.shape[1] == k.shape[1]

        # qkv has heads combined into the batch
        num_heads = q.shape[0] // weights.shape[0]
        weights = repeat(weights, "b t -> (b h) 1 t", h=num_heads)

    k = k.transpose(1, 2)

    if chunks is None:
        attn = softmax(q @ k, dim=-1)
        del q, k

        if weights is not None:
            attn *= weights

        return attn @ v

    # TODO auto, see how much free memory is available?

    # split-attention score
    shape = (q.size(0), q.size(1), k.size(1))
    out = torch.empty(shape, device=q.device, dtype=q.dtype)
    for i in range(0, len(k), chunks):
        s = slice(i, i + chunks)

        attn = softmax(q[s] @ k[s], dim=-1)
        if weights is not None:
            attn *= weights[s]

        out[s] = attn @ v[s]
        del attn

    return out
