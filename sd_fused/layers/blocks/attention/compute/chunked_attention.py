from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .scale_qk import scale_qk


def batch_chunked_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
    chunks: int,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Batch-chunked attention computation."""

    assert chunks >= 1
    B, T, C = q.shape

    q, k = scale_qk(q, k)
    kT = k.transpose(1, 2)

    out = torch.empty_like(q)
    for i in range(0, B, chunks):
        s = slice(i, min(i + chunks, B))

        score = q[s] @ kT[s]
        if bias is not None:
            score += bias[s]
        attn = F.softmax(score, dim=2, dtype=q.dtype)
        del score

        out[s] = attn @ v[s]
        del attn

    return out


def sequence_chunked_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
    chunks: int,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Sequence-chunked attention computation."""

    # https://github.com/Doggettx/stable-diffusion/blob/main/ldm/modules/attention.py#L209

    assert chunks >= 1
    B, T, C = q.shape

    q, k = scale_qk(q, k)
    kT = k.transpose(1, 2)

    out = torch.empty_like(q)
    for i in range(0, T, chunks):
        s = slice(i, min(i + chunks, T))

        score = q[:, s] @ kT
        if bias is not None:
            score += bias[:, s]  # ?
        attn = F.softmax(score, dim=-1, dtype=q.dtype)
        del score

        out[:, s] = attn @ v
        del attn

    return out
