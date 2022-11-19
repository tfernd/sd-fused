from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .scale_qk import scale_qk


def batch_chunked_attention(
    q: Tensor,  # (B, heads, T, C)
    k: Tensor,  # (B, heads, T', C)
    v: Tensor,  # (B, heads, T', C)
    chunks: int,
    bias: Optional[Tensor] = None,  # ? shape
) -> Tensor:
    """Batch-chunked attention computation."""

    assert chunks >= 1
    B, heads, T, C = q.shape

    # join batch-heads
    q = q.flatten(0, 1)
    k = k.flatten(0, 1)
    v = v.flatten(0, 1)

    q, k = scale_qk(q, k)
    kT = k.transpose(-1, -2)

    out = torch.empty_like(q)
    for i in range(0, B * heads, chunks):
        s = slice(i, min(i + chunks, B * heads))

        score = q[s] @ kT[s]
        if bias is not None:
            score += bias[s]
        attn = F.softmax(score, dim=-1, dtype=q.dtype)
        del score

        out[s] = attn @ v[s]
        del attn

    return out.unflatten(0, (B, heads))


def sequence_chunked_attention(
    q: Tensor,  # (B, heads, T, C)
    k: Tensor,  # (B, heads, T', C)
    v: Tensor,  # (B, heads, T', C)
    chunks: int,
    bias: Optional[Tensor] = None,  # ? shape
) -> Tensor:
    """Sequence-chunked attention computation."""

    # https://github.com/Doggettx/stable-diffusion/blob/main/ldm/modules/attention.py#L209

    assert chunks >= 1
    B, heads, T, C = q.shape

    # join batch-heads
    q = q.flatten(0, 1)
    k = k.flatten(0, 1)
    v = v.flatten(0, 1)

    q, k = scale_qk(q, k)
    kT = k.transpose(-1, -2)

    out = torch.empty_like(q)
    for i in range(0, T, chunks):
        s = slice(i, min(i + chunks, T))

        score = q[:, s] @ kT
        if bias is not None:
            score += bias[:, s]
        attn = F.softmax(score, dim=-1, dtype=q.dtype)
        del score

        out[:, s] = attn @ v
        del attn

    return out.unflatten(0, (B, heads))
