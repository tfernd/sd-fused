from __future__ import annotations

from torch import Tensor
from einops import rearrange

from ..base import Module


class Rearrange(Module):
    def __init__(self, pattern: str, **axes_length: int) -> None:
        self.pattern = pattern
        self.axes_length = axes_length

    def __call__(self, x: Tensor, **axes_length: int) -> Tensor:
        return rearrange(x, self.pattern, **self.axes_length, **axes_length)

    def make_inverse(self) -> Rearrange:
        left, right = self.pattern.split('->')

        new = Rearrange(f'{right} -> {left}')

        return new

