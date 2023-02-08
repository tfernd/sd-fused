from __future__ import annotations
from typing import Optional
from typing_extensions import Literal

from torch import Tensor

from ....functional.attention import attention, ChunkType


class BaseAttention:
    attention_chunks: Optional[int | Literal["auto"]] = None
    chunk_type: Optional[ChunkType] = None
    use_flash_attention: bool = False
    tome_r: Optional[int | float] = None

    def attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        return attention(
            q,
            k,
            v,
            weights=weights,
            chunks=self.attention_chunks,
            chunk_type=self.chunk_type,
            use_flash_attention=self.use_flash_attention,
            tome_r=self.tome_r,
        )
