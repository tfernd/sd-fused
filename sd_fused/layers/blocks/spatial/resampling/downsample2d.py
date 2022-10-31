from __future__ import annotations
from typing import Callable, Optional

from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ....base import Conv2d


class Downsample2D(nn.Module):
    pad: Callable[[Tensor], Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int],
        *,
        padding: int,
    ) -> None:
        super().__init__()

        self.channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        out_channels = out_channels or in_channels
        stride = 2

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
        )

        self.pad = lambda x: x
        if padding == 0:
            # ? Why?
            self.pad = partial(
                F.pad,
                pad=(0, 1, 0, 1),
                mode="constant",
                value=0,
            )

    def __call__(self, x: Tensor) -> Tensor:
        x = self.pad(x)

        return self.conv(x)
