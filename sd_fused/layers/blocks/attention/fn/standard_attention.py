from __future__ import annotations
from typing import Optional

from torch import Tensor

from .....utils.tensors import softmax
from .scale_qk import scale_qk


def standard_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Standard attention computation."""

    q, k = scale_qk(q, k)
    score = q @ k.transpose(1, 2)
    if bias is not None:
        score += bias
    attn = softmax(score, dim=2)
    del score

    return attn @ v
