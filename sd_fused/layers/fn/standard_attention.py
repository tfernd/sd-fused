from __future__ import annotations

import torch
from torch import Tensor

from ...utils import softmax
from .scale_qk import scale_qk


def standard_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
) -> Tensor:
    """Standard attention computation."""

    # ! Faster to use baddbmm?
    # score = torch.baddbmm()

    q, k = scale_qk(q, k)
    attn = softmax(q @ k.transpose(1, 2), dim=-1)
    del q, k

    return attn @ v
