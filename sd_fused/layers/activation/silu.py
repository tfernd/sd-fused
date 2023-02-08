from __future__ import annotations

import torch
from torch import Tensor

from ..base import Module


class SiLU(Module):
    def __call__(self, x: Tensor) -> Tensor:
        with torch.set_grad_enabled(self.training):
            return x.sigmoid().mul(x)
