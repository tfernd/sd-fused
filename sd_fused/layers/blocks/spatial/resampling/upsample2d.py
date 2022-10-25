from __future__ import annotations
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ....base import Conv2d


class Upsample2D(nn.Module):
    def __init__(
        self, channels: int, out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels

        out_channels = out_channels or channels

        self.conv = Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def __call__(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, mode="nearest", scale_factor=2)

        return self.conv(x)
