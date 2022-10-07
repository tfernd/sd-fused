from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import Linear


class GEGLU(nn.Module):
    """GEGLU activation function (x*gelu(x))."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.proj = Linear(dim_in, 2 * dim_out)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)

        return F.gelu(gate).mul_(x)
        # return x * F.gelu(gate)
