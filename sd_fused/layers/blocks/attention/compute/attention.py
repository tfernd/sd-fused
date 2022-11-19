from __future__ import annotations
from typing import Optional

from torch import Tensor

from .....utils.typing import Literal
from .join_spatial_dim import join_spatial_dim
from .auto_chunk_size import auto_chunk_size, ChunkType
from .standard_attention import standard_attention
from .chunked_attention import batch_chunked_attention, sequence_chunked_attention
from .flash_attention import flash_attention
from .weighted_values import weighted_values
from .tome import token_average


def attention(
    q: Tensor,  # (B, heads, H, W, C)
    k: Tensor,  # (B, heads, *[H,W] | T', C)
    v: Tensor,  # (B, heads, *[H,W] | T', C)
    *,
    weights: Optional[Tensor] = None,  # (B, T')
    chunks: Optional[int | Literal["auto"]] = None,
    chunk_type: Optional[ChunkType] = None,
    use_flash_attention: bool = False,
    tome_r: Optional[int | float] = None,
) -> Tensor:
    """General attention computation."""

    # header
    is_self_attention = q.shape == k.shape
    dtype = q.dtype
    B, heads, H, W, C = q.shape
    T = H * W

    if weights is not None:
        assert not is_self_attention

        v = weighted_values(v, weights) # ?keys is vad?

    q, k, v = join_spatial_dim(q, k, v)
    Tl = k.size(2)

    chunks = auto_chunk_size(chunks, B, heads, T, Tl, C, dtype, chunk_type)

    if is_self_attention and tome_r is not None:
        k, v, bias = token_average(k, v, tome_r)
    bias = None  # ! not used for now.

    if chunks is not None:
        assert not use_flash_attention
        assert chunk_type is not None  # ! default?

        if chunk_type == "batch":
            out = batch_chunked_attention(q, k, v, chunks, bias)
        else:
            out = sequence_chunked_attention(q, k, v, chunks, bias)

    elif use_flash_attention:
        assert chunk_type is None
        assert tome_r is None

        out = flash_attention(q, k, v, bias)
    else:
        out = standard_attention(q, k, v, bias)

    out = out.unflatten(2, (H, W))  # separate-spatial-dim

    return out
