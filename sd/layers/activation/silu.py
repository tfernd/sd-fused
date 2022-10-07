from __future__ import annotations

import torch.nn as nn
from torch import Tensor

class SiLU(nn.Module):
    """SiLU activation function (sigmoid(x)*x)."""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid().mul_(x)
        # return x * x.sigmoid()
