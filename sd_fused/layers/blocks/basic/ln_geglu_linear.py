from __future__ import annotations

from torch import Tensor

from ....layers.basic import LayerNorm, Linear
from ....layers.activation import GEGLU
from ...base import Sequential


class LayerNormGEGLULinear(Sequential):
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
        return x + super().__call__(x)
