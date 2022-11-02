from __future__ import annotations
from typing import Optional

import math

import torch
from torch import Tensor

from einops import repeat

from ...utils import softmax
from ...utils.typing import Literal
from .auto_chunk_size import auto_chunk_size
from .scale_qk import scale_qk
from .standard_attention import standard_attention
from .chunked_attention import chunked_attention
from .flash_attention import flash_attention
from .weight_modify_v import weight_modify_v


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
