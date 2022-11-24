from __future__ import annotations
from typing import Optional

from ...base import Sequential
from ...basic import GroupNorm, Conv2d

# TODO improve types by using ModuleTuple
class GroupNormConv2d(Sequential[GroupNorm | Conv2d]):
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
            Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )

        super().__init__(*layers)
