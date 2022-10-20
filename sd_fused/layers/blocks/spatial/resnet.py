from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange

from ...activation import SiLU
from ...base import Conv2d, Linear
from ..simple import GroupNormSiLUConv2d


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int],
        temb_channels: Optional[int],
        num_groups: int,
        num_out_groups: Optional[int],
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.num_groups = num_groups
        self.num_out_groups = num_out_groups

        out_channels = out_channels or in_channels
        num_out_groups = num_out_groups or num_groups

        if in_channels != out_channels:
            self.conv_shortcut = Conv2d(in_channels, out_channels)
        else:
            self.conv_shortcut = nn.Identity()

        self.pre_process = GroupNormSiLUConv2d(
            num_groups, in_channels, out_channels, kernel_size=3, padding=1
        )

        self.post_process = GroupNormSiLUConv2d(
            num_out_groups,
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        # self.nonlinearity = SiLU()
        if temb_channels is not None:
            self.time_emb_proj = nn.Sequential(
                SiLU(),
                Linear(temb_channels, out_channels),
                Rearrange("b c -> b c 1 1"),
            )
            # self.time_emb_proj = Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

    def __call__(self, x: Tensor, *, temb: Optional[Tensor] = None) -> Tensor:
        xin = self.conv_shortcut(x)

        x = self.pre_process(x)

        if self.time_emb_proj is not None:
            assert temb is not None

            # temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)
            # temb = temb[..., None, None]

            x = x + temb
            del temb

        x = self.post_process(x)

        return xin + x
