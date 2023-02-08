from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..base import Module
from ..basic import Linear


class GEGLU(Module):
    """Applies the SiLU (Sigmoid-weighted Linear Unit) activation function element-wise."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.proj = Linear(in_features, 2 * out_features)

    def __call__(self, x: Tensor) -> Tensor:
        with torch.set_grad_enabled(self.training):
            x, gate = self.proj(x).chunk(2, dim=-1)

            return F.gelu(gate).mul(x)
