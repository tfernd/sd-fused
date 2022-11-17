from __future__ import annotations
from typing import Optional
from typing_extensions import Self

from ...layers.base import Module
from ...utils.typing import Literal
from ...layers.blocks.attention import CrossAttention, SelfAttention
from ...layers.blocks.attention.compute import ChunkType


class SplitAttentionModel(Module):
    def split_attention(
        self,
        chunks: Optional[int | Literal["auto"]] = "auto",
        chunk_type: Optional[ChunkType] = None,
    ) -> Self:
        """Split cross/self-attention computation into chunks."""

        for name, module in self.named_modules().items():
            if isinstance(module, (CrossAttention, SelfAttention)):
                module.attention_chunks = chunks
                module.chunk_type = chunk_type

                if chunks is not None:
                    module.use_flash_attention = False

        return self
