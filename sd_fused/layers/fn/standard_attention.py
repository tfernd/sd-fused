from __future__ import annotations
from typing import Optional

from torch import Tensor

from ...utils import softmax
from .scale_qk import scale_qk


def standard_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
    bias: Optional[Tensor] = None
) -> Tensor:
    """Standard attention computation."""

    q, k = scale_qk(q, k)
    attn = softmax(q @ k.transpose(1, 2), dim=2)
    del q, k

    return attn @ v

