from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import Module
from ..modifiers import HalfWeightsModule, half_weights


class LayerNorm(HalfWeightsModule, Module):
    def __init__(
        self,
        shape: int | tuple[int, ...],
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        self.shape = shape = shape if isinstance(shape, tuple) else (shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        empty = partial(torch.empty, dtype=self.dtype, device=self.device)
        parameter = partial(nn.Parameter, requires_grad=False)

        self.weight = parameter(empty(shape)) if elementwise_affine else NotImplemented
        self.bias = parameter(empty(shape)) if elementwise_affine else NotImplemented

    @half_weights
    def __call__(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.shape, self.weight, self.bias, self.eps)
