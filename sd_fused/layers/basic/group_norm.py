from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import Module
from ..modifiers import HalfWeightsModule, half_weights


class GroupNorm(HalfWeightsModule, Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        *,
        eps: float = 1e-6,
        affine: bool = True,
    ) -> None:
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        empty = partial(torch.empty, dtype=self.dtype, device=self.device)
        parameter = partial(nn.Parameter, requires_grad=False)

        self.weight = parameter(empty(num_channels)) if affine else None
        self.bias = parameter(empty(num_channels)) if affine else None

    @half_weights
    def __call__(self, x: Tensor) -> Tensor:
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
