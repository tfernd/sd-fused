from __future__ import annotations
from typing import Optional

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import Module
from ..modifiers import HalfWeightsModule, half_weights


class Conv2d(HalfWeightsModule, Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        *,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        assert in_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels = out_channels or in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        # TODO duplication
        empty = partial(torch.empty, dtype=self.dtype, device=self.device)
        parameter = partial(nn.Parameter, requires_grad=False)

        w = empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        self.weight = parameter(w)
        self.bias = parameter(empty(out_channels)) if bias else None

    @half_weights
    def __call__(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
