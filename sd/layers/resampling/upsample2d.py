from __future__ import annotations
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...utils.typing import Literal
from ..base import Conv2d


class Upsample2D(nn.Module):
    kind: Literal["conv", "transpose", "none"]

    def __init__(
        self,
        *,
        channels: int,
        kind: Literal["conv", "transpose", "none"],
        out_channels: Optional[int],
    ) -> None:
        super().__init__()

        self.channels = channels
        self.kind = kind
        self.out_channels = out_channels

        out_channels = out_channels or channels

        if kind == "conv":
            self.conv = Conv2d(
                channels, out_channels, kernel_size=3, padding=1
            )
        elif kind == "transpose":
            # never used for SD
            self.conv = nn.ConvTranspose2d(
                channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        else:
            assert kind == "none"

            self.conv = nn.Identity()

    def forward(
        self, x: Tensor, *, size: Optional[tuple[int, int]] = None
    ) -> Tensor:
        if self.kind == "transpose":
            return self.conv(x)

        kwargs = dict(scale_factor=2) if size is None else dict(size=size)
        x = F.interpolate(x, mode="nearest", **kwargs)

        return self.conv(x)
