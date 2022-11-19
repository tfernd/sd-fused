from __future__ import annotations
from typing import Optional

import torch.nn.functional as F
from torch import Tensor

from .scale_qk import scale_qk


def standard_attention(
    q: Tensor,  # (B, heads, T, C)
    k: Tensor,  # (B, heads, T', C)
    v: Tensor,  # (B, heads, T', C)
    bias: Optional[Tensor] = None,  # ? shape?
) -> Tensor:
    """Standard attention computation."""

    q, k = scale_qk(q, k)
    score = q @ k.transpose(-1, -2)
    if bias is not None:
        score += bias
    attn = F.softmax(score, dim=-1, dtype=q.dtype)
    del score

    return attn @ v
