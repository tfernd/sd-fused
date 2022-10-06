from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from ...utils import softmax


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    inplace: bool = False,
    chunks: Optional[int] = None,
) -> Tensor:
    if chunks is None:
        attn = softmax(q @ k, dim=-1, inplace=inplace)
        del q, k

        return attn @ v

    # split-attention score
    shape = (*q.shape[:2], k.shape[1])
    x = torch.empty(shape, device=q.device, dtype=q.dtype)
    for i in range(0, k.shape[0], chunks):
        s = slice(i, i + chunks)

        attn = softmax(q[s] @ k[s], dim=-1, inplace=inplace)
        x[s] = attn @ v[s]
        del attn

        # TODO split and delete q[s], k[s], v[s] to save memory?

    return x
