from __future__ import annotations
from typing import Optional

import torch.nn as nn

from ...base import GroupNorm, Conv2d
from ...activation import SiLU


class GroupNormSiLUConv2d(nn.Sequential):
    def __init__(
        self,
        num_groups: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        *,
        kernel_size: int = 1,
        padding: int = 0,
    ) -> None:
        layers = (
            GroupNorm(num_groups, in_channels),
            SiLU(),
            Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )

        super().__init__(*layers)
