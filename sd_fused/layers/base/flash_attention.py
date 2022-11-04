from __future__ import annotations

import torch.nn as nn

from ..blocks.attention import CrossAttention, SelfAttention


class FlashAttentionModel(nn.Module):
    def flash_attention(self, use: bool = True) -> None:
        """Use xformers flash-attention."""

        for name, module in self.named_modules():
            if isinstance(module, (CrossAttention, SelfAttention)):
                module.use_flash_attention = use

                if use:
                    module.attention_chunks = None
