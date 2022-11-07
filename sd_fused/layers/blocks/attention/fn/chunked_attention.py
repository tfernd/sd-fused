from __future__ import annotations
from typing import Optional

import math
import torch
from torch import Tensor

from .....utils.tensors import softmax
from .scale_qk import scale_qk


def chunked_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
    chunks: int,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Chunked attention computation."""

    assert chunks >= 1

    q, k = scale_qk(q, k)

    kT = k.transpose(1, 2)

    out = torch.empty_like(q)
    for i in range(0, len(k), chunks):
        s = slice(i, min(i + chunks, len(k)))

        attn = softmax(q[s] @ kT[s], dim=-1)

        out[s] = attn @ v[s]
        del attn

    return out
