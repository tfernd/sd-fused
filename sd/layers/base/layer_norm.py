from __future__ import annotations

import torch.nn as nn



from .ops import Ops

class LayerNorm(Ops, nn.LayerNorm):
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
