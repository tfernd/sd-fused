from __future__ import annotations
from typing import Optional

from functools import partial

import torch.nn as nn
from torch import Tensor

from ..activation import SiLU
from ..base import Conv2d, Linear, GroupNorm


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int],
        temb_channels: Optional[int],
        groups: int,
        groups_out: Optional[int],
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.groups = groups
        self.groups_out = groups_out

        out_channels = out_channels or in_channels
        groups_out = groups_out or groups

        conv3 = partial(Conv2d, kernel_size=3, padding=1)

        self.norm1 = GroupNorm(groups, in_channels)
        self.norm2 = GroupNorm(groups_out, out_channels)

        self.conv1 = conv3(in_channels, out_channels)
        self.conv2 = conv3(out_channels)

        if temb_channels is not None:
            self.time_emb_proj = Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.nonlinearity = SiLU()

        if in_channels != out_channels:
            self.conv_shortcut = Conv2d(in_channels, out_channels)
        else:
            self.conv_shortcut = nn.Identity()

    def forward(self, x: Tensor, *, temb: Optional[Tensor] = None) -> Tensor:
        xin = self.conv_shortcut(x)

        # TODO create own layer
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)

        if self.time_emb_proj is not None:
            assert temb is not None

            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)
            assert isinstance(temb, Tensor)  # needed for type checking below
            temb = temb[..., None, None]

            x = x + temb
            del temb

        # TODO create own layer
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)

        return xin + x
