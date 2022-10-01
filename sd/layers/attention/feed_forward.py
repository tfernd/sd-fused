from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ..base import Linear
from ..activation import GEGLU


class FeedForward(nn.Module):
    def __init__(
        self, dim: int, dim_out: Optional[int], *, mult: float,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.mult = mult

        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        self.net = nn.Sequential(
            GEGLU(dim, inner_dim),
            nn.Identity(),  # Dropout removed
            Linear(inner_dim, dim_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
