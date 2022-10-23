from __future__ import annotations
from typing import Optional

import torch.nn as nn

from ..blocks.attention import CrossAttention


class SplitAttentionModel(nn.Module):
    def split_attention(
        self, cross_attention_chunks: Optional[int] = None
    ) -> None:
        """Split cross/self-attention computation into chunks."""

        if cross_attention_chunks is not None:
            assert cross_attention_chunks >= 1

        for name, module in self.named_modules():
            if isinstance(module, CrossAttention):
                module.attention_chunks = cross_attention_chunks
