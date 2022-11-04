from __future__ import annotations
from typing import Optional

import torch.nn as nn

from ...utils.typing import Literal
from ..blocks.attention import CrossAttention, SelfAttention


class SplitAttentionModel(nn.Module):
    def split_attention(
        self,
        chunks: Optional[int | Literal["auto"]] = "auto",
    ) -> None:
        """Split cross/self-attention computation into chunks."""

        for name, module in self.named_modules():
            if isinstance(module, (CrossAttention, SelfAttention)):
                module.attention_chunks = chunks

                if chunks is not None:
                    module.use_flash_attention = False
