from __future__ import annotations
from typing import Optional

import torch.nn as nn

from ..blocks.attention import CrossAttention, SelfAttention


class ToMe(nn.Module):
    # https://arxiv.org/abs/2210.09461
    # https://github.com/facebookresearch/ToMe/blob/main/tome/merge.py

    def tome(self, r: Optional[int | float] = None) -> None:
        """Merge similar tokens."""

        for name, module in self.named_modules():
            if isinstance(module, (CrossAttention, SelfAttention)):
                module.tome_r = r

                module.use_flash_attention = False  # ! for now
