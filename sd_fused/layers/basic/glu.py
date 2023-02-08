from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..base import Module
from ..modifiers import HalfWeightsModule, half_weights

class GLU(HalfWeightsModule, Module):
    def __init__(
        self,
        dim: int = -1
    ) -> None:
        super().__init__()

        self.dim = dim

    @half_weights
    def __call__(self, x: Tensor) -> Tensor:
        with torch.set_grad_enabled(self.training):
            return F.glu(x, self.dim)
