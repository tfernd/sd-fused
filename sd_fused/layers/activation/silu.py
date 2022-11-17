from __future__ import annotations

from torch import Tensor

from ..base import Module


class SiLU(Module):
    def __call__(self, x: Tensor) -> Tensor:
        return x.sigmoid().mul_(x)
