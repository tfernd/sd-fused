from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .half_weights import HalfWeights


class GroupNorm(HalfWeights, nn.GroupNorm):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        *,
        eps: float = 1e-6,
        affine: bool = True,
    ) -> None:
        super().__init__(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
        )

    def __call__(self, x: Tensor) -> Tensor:
        return super().half_weights_forward(x)
