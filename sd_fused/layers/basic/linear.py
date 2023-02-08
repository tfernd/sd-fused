from __future__ import annotations
from typing import Optional

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from ..base import Module, Parameter
from ..modifiers import HalfWeightsModule, half_weights


class Linear(HalfWeightsModule, Module):
    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features = out_features or in_features

        empty = partial(torch.empty, dtype=self.dtype, device=self.device)
        parameter = partial(Parameter, requires_grad=False)

        self.weight = parameter(empty(out_features, in_features))
        self.bias = parameter(empty(out_features)) if bias else None

    @half_weights
    def __call__(self, x: Tensor) -> Tensor:
        with torch.set_grad_enabled(self.training):
            return F.linear(x, self.weight, self.bias)
