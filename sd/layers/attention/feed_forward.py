from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ..base import Linear, LayerNorm
from ..activation import GEGLU


class FeedForward(nn.Sequential):
    def __init__(
        self, dim: int, dim_out: Optional[int], *, mult: float,
    ) -> None:

        self.dim = dim
        self.dim_out = dim_out
        self.mult = mult

        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        layers = (
            LayerNorm(dim),
            GEGLU(dim, inner_dim),
            Linear(inner_dim, dim_out),
        )

        super().__init__(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)
