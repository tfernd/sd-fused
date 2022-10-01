from __future__ import annotations
from typing_extensions import Self

import torch.nn as nn
from torch import Tensor


class HalfWeights(nn.Module):
    """Store the weights in half-precision but
    compute forward pass in full precision.
    """

    use_half_weights: bool = False

    def half_weights(self) -> None:
        self.use_half_weights = True
        self.half()

    def forward(self, x: Tensor) -> Tensor:
        if not self.use_half_weights:
            return super().forward(x)

        self.float()
        out = super().forward(x)
        self.half()

        return out


class HalfWeightsModel(nn.Module):
    def half_weights(self) -> Self:
        for name, module in self.named_modules():
            if isinstance(module, HalfWeights):
                module.half_weights()

        return self
