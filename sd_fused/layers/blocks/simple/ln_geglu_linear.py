from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ...base import Linear, LayerNorm
from ...activation import GEGLU


class LayerNormGEGLULinear(nn.Sequential):
    def __init__(self, dim: int, *, expand: float) -> None:

        self.dim = dim
        self.expand = expand

        inner_dim = int(dim * expand)
        dim = dim or dim

        layers = (
            LayerNorm(dim),
            GEGLU(dim, inner_dim),
            Linear(inner_dim, dim),
        )

        super().__init__(*layers)

    def __call__(self, x: Tensor) -> Tensor:
        return x + super().forward(x)
