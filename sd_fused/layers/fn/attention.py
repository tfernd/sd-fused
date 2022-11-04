from __future__ import annotations
from typing import Optional

from torch import Tensor

from ...utils.typing import Literal
from .auto_chunk_size import auto_chunk_size
from .standard_attention import standard_attention
from .chunked_attention import chunked_attention
from .flash_attention import flash_attention
from .weight_modify_v import weight_modify_v

# TODO tome debug
# from tome.merge import  bipartite_soft_matching, merge_wavg


def attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
    *,
    weights: Optional[Tensor] = None,  # (B, T')
    chunks: Optional[int | Literal["auto"]] = None,
    use_flash_attention: bool = False,
    # use_tome: bool = False, # TODO implement
) -> Tensor:
    """General attention computation."""

    assert q.ndim == k.ndim == v.ndim == 3
    assert q.shape[0] == k.shape[0] == v.shape[0]
    assert q.shape[2] == k.shape[2] == v.shape[2]
    assert k.shape[1] == v.shape[1]

    B, T, C = q.shape
    B, Tl, C = k.shape
    dtype = q.dtype

    v = weight_modify_v(v, weights)

    # TODO ToMe?
    # if T == Tl:
    #     merge, unmerge = bipartite_soft_matching(q, C//2)

    #     # q, size = merge_wavg(merge, q)
    #     k, size = merge_wavg(merge, k)
    #     v, size = merge_wavg(merge, v)

    if chunks == "auto":
        chunks = auto_chunk_size(B, T, Tl, C, dtype)

    if chunks is not None:
        assert not use_flash_attention
        return chunked_attention(q, k, v, chunks)

    if use_flash_attention:
        assert chunks is None
        return flash_attention(q, k, v)

    return standard_attention(q, k, v)
