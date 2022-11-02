from __future__ import annotations

from torch import Tensor

from ...utils import softmax
from .scale_qk import scale_qk


def standard_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
) -> Tensor:
    """Standard attention computation."""

    q, k = scale_qk(q, k)
    attn = softmax(q @ k.transpose(1, 2), dim=-1)
    del q, k

    return attn @ v
