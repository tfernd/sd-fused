from __future__ import annotations

from torch import Tensor

from .module import Module
from .module_list import ModuleList, T


class Sequential(ModuleList[T], Module):
    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        return x
