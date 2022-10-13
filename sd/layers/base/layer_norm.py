from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .half_weights import HalfWeights


class LayerNorm(HalfWeights, nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ) -> None:
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def __call__(self, x: Tensor) -> Tensor:
        return super().half_weights_forward(x)
