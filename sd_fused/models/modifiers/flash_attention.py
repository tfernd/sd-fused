from __future__ import annotations
from typing_extensions import Self

from ...layers.base import Module
from ...layers.blocks.attention import CrossAttention, SelfAttention


class FlashAttentionModel(Module):
    def flash_attention(self, use: bool = True) -> Self:
        """Use xformers flash-attention."""

        for name, module in self.named_modules().items():
            if isinstance(module, (CrossAttention, SelfAttention)):
                module.use_flash_attention = use

                if use:
                    module.attention_chunks = None
                    module.chunk_type = None

        return self
