from __future__ import annotations
from typing import Optional

import math

import torch
from torch import Tensor

from einops import repeat

from ...utils import softmax
from ...utils.typing import Literal
from .auto_chunk_size import auto_chunk_size

try:
    from xformers.ops import memory_efficient_attention  # type: ignore
except ImportError:
    memory_efficient_attention = None


def scale_qk(*xs: Tensor) -> tuple[Tensor, Tensor]:
    """Scale the qk tensor by the channel-dimension."""

    assert len(xs) == 2

    return tuple(x * math.pow(x.size(-1), -1 / 4) for x in xs)


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


def chunked_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
    chunks: int,
) -> Tensor:
    """Chunked attention computation."""

    assert chunks >= 1

    q, k = scale_qk(q, k)

    out = torch.empty_like(q)
    for i in range(0, len(k), chunks):
        s = slice(i, min(i + chunks, len(k)))

        attn = softmax(q[s] @ k[s].transpose(1, 2), dim=-1)

        out[s] = attn @ v[s]
        del attn

    return out


def flash_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
) -> Tensor:
    """xformers flash-attention computation."""

    assert memory_efficient_attention is not None

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    return memory_efficient_attention(q, k, v)


def weight_modify_v(x: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    """Modify the `values` by a given token-weight."""

    if weights is None:
        return x

    assert weights.ndim == 2
    assert weights.shape[1] == x.shape[1]

    # qkv has heads combined into the batch
    num_heads = x.shape[0] // weights.shape[0]
    weights = repeat(weights, "b t -> (b h) t 1", h=num_heads)

    return x * weights


def attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
    *,
    weights: Optional[Tensor] = None,  # (B, T')
    chunks: Optional[int | Literal["auto"]] = None,
    use_flash_attention: bool = False,
) -> Tensor:
    """General attention computation."""

    assert q.ndim == k.ndim == v.ndim == 3
    assert q.shape[0] == k.shape[0] == v.shape[0]
    assert q.shape[2] == k.shape[2] == v.shape[2]
    assert k.shape[1] == v.shape[1]

    B, T, C = q.shape
    Tl = k.shape[1]
    dtype = q.dtype

    # k = weight_modify_v(k, weights) # gives crappy result
    v = weight_modify_v(v, weights)

    if chunks == "auto":
        chunks = auto_chunk_size(B, T, Tl, C, dtype)

    if chunks is not None:
        assert not use_flash_attention
        return chunked_attention(q, k, v, chunks)

    if use_flash_attention:
        assert chunks is None
        return flash_attention(q, k, v)

    return standard_attention(q, k, v)
