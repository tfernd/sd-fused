from __future__ import annotations
from typing import Optional

from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ....base import Conv2d


class Upsample2D(nn.Module):
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
        self.upscale = partial(F.interpolate, mode="nearest", scale_factor=2)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.upscale(x)

        return self.conv(x)
