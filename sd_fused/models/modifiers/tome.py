from __future__ import annotations
from typing import Optional
from typing_extensions import Self

from ...layers.base import Module
from ...layers.blocks.attention import CrossAttention, SelfAttention


class ToMeModel(Module):
    # https://arxiv.org/abs/2210.09461
    # https://github.com/facebookresearch/ToMe/blob/main/tome/merge.py

    def tome(self, r: Optional[int | float] = None) -> Self:
        """Merge similar tokens."""

        for name, module in self.named_modules().items():
            if isinstance(module, (CrossAttention, SelfAttention)):
                module.tome_r = r

        return self
