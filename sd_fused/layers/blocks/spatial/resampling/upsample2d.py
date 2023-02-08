from __future__ import annotations
from typing import Optional

from functools import partial

import torch.nn.functional as F
from torch import Tensor

from ....base import Module
from ....basic import Conv2d


class Upsample2D(Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels

        out_channels = out_channels or channels

        self.conv = Conv2d(channels, out_channels, kernel_size=3, padding=1)
        self.upscale = partial(F.interpolate, mode="nearest")

    def __call__(
        self,
        x: Tensor,
        *,
        size: Optional[tuple[int, int]],
    ) -> Tensor:
        if size is not None:
            x = self.upscale(x, size=size)
        else:
            x = self.upscale(x, scale_factor=2)

        return self.conv(x)
