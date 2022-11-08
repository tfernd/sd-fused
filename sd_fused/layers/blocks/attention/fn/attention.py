from __future__ import annotations
from typing import Optional

from torch import Tensor

from .....utils.typing import Literal
from .auto_chunk_size import auto_chunk_size, ChunkType
from .standard_attention import standard_attention
from .chunked_attention import (
    batch_chunked_attention,
    sequence_chunked_attention,
)
from .flash_attention import flash_attention
from .weighted_values import weighted_values
from .tome import token_average


def attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
    *,
    weights: Optional[Tensor] = None,  # (B, T')
    chunks: Optional[int | Literal["auto"]] = None,
    chunk_type: Optional[ChunkType] = None,
    use_flash_attention: bool = False,
    tome_r: Optional[int | float] = None,
) -> Tensor:
    """General attention computation."""

    assert q.ndim == k.ndim == v.ndim == 3
    assert q.shape[0] == k.shape[0] == v.shape[0]
    assert q.shape[2] == k.shape[2] == v.shape[2]
    assert k.shape[1] == v.shape[1]

    B, T, C = q.shape
    B, Tl, C = k.shape
    dtype = q.dtype

    v = weighted_values(v, weights)

    if T == Tl and tome_r is not None:
        k, v, bias = token_average(k, v, tome_r)
    bias = None  # ! bias not used yet

    if chunks == "auto":
        assert chunk_type is not None
        chunks = auto_chunk_size(B, T, Tl, C, dtype, chunk_type)

    if chunks is not None:
        assert not use_flash_attention
        assert chunk_type is not None
        if chunk_type == "batch":
            return batch_chunked_attention(q, k, v, chunks, bias)
        else:
            return sequence_chunked_attention(q, k, v, chunks, bias)

    if use_flash_attention:
        assert chunks is None
        assert chunk_type is None
        assert tome_r is None
        return flash_attention(q, k, v, bias)

    return standard_attention(q, k, v, bias)
