from __future__ import annotations
from typing_extensions import Self

import torch
from torch import Tensor
from einops import rearrange

from ..base import Module


class Rearrange(Module):
    def __init__(self, pattern: str, **axes_length: int) -> None:
        super().__init__()

        self.pattern = pattern
        self.axes_length = axes_length

    def __call__(self, x: Tensor, **axes_length: int) -> Tensor:
        with torch.set_grad_enabled(self.training):
            return rearrange(x, self.pattern, **self.axes_length, **axes_length)

    def make_inverse(self) -> Self:
        left, right = self.pattern.split("->")

        return Rearrange(f"{right} -> {left}")
