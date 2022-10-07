from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from ...utils import softmax


def attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T, C')
    v: Tensor,  # (B, T', C)
    chunks: Optional[int] = None,
) -> Tensor:
    k = k.transpose(1, 2)

    if chunks is None:
        attn = softmax(q @ k, dim=-1)
        del q, k

        return attn @ v

    # split-attention score
    shape = (*q.shape[:2], k.shape[1])
    out = torch.empty(shape, device=q.device, dtype=q.dtype)
    for i in range(0, k.shape[0], chunks):
        s = slice(i, i + chunks)

        attn = softmax(q[s] @ k[s], dim=-1)
        out[s] = attn @ v[s]
        del attn

    return out
