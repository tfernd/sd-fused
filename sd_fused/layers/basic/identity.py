from __future__ import annotations

from torch import Tensor

from ..base import Module


class Identity(Module):
    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x
