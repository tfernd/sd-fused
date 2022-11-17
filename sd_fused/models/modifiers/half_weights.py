from __future__ import annotations
from typing_extensions import Self

from ...layers.base import Module
from ...layers.blocks.attention import CrossAttention, SelfAttention
from ...layers.modifiers import HalfWeightsModule


class HalfWeightsModel(Module):
    def half_weights(self, use: bool = True) -> Self:
        """Store the weights in half-precision but
        compute forward pass in full precision.
        Useful for GPUs that gives NaN when used in half-precision.
        """

        for name, module in self.named_modules().items():
            if isinstance(module, HalfWeightsModule):
                module.half_weights(use)

        return self
