from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from ..base import Module, Parameter
from ..modifiers import HalfWeightsModule, half_weights


class LayerNorm(HalfWeightsModule, Module):
    shape: tuple[int, ...]

    def __init__(
        self,
        shape: int | tuple[int, ...],
        *,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.shape = shape = shape if isinstance(shape, tuple) else (shape,)
        self.eps = eps

        empty = partial(torch.empty, dtype=self.dtype, device=self.device)
        parameter = partial(Parameter, requires_grad=False)

        self.weight = parameter(empty(shape))
        self.bias = parameter(empty(shape))

    @half_weights
    def __call__(self, x: Tensor) -> Tensor:
        with torch.set_grad_enabled(self.training):
            return F.layer_norm(x, self.shape, self.weight, self.bias, self.eps)
