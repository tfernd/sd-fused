from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ..base import Linear, LayerNorm
from ..activation import GEGLU

# TODO Rename as LayerNormGEGLULinear and add to blocks
class FeedForward(nn.Sequential):
    def __init__(self, dim: int, *, mult: float,) -> None:

        self.dim = dim
        self.mult = mult

        inner_dim = int(dim * mult)
        dim = dim or dim

        layers = (
            LayerNorm(dim),
            GEGLU(dim, inner_dim),
            Linear(inner_dim, dim),
        )

        super().__init__(*layers)

    def __call__(self, x: Tensor) -> Tensor:
        return x + super().forward(x)
