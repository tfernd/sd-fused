from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ..base import InPlace


class SiLU(InPlace, nn.Module):
    """SiLU activation function (sigmoid(x)*x)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.sigmoid().mul_(x)
        return x * x.sigmoid()
