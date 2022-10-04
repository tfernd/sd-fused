from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ..resampling import Downsample2D
from .resnet_block2d import ResnetBlock2D


class DownBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        temb_channels: Optional[int],
        num_layers: int,
        resnet_groups: int,
        add_downsample: bool,
        downsample_padding: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.num_layers = num_layers
        self.resnet_groups = resnet_groups
        self.add_downsample = add_downsample
        self.downsample_padding = downsample_padding

        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels

            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    groups_out=None,
                )
            )

        if add_downsample:
            self.downsamplers = nn.ModuleList()
            self.downsamplers.append(
                Downsample2D(
                    channels=in_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
            )
        else:
            self.downsamplers = None

    def forward(
        self, x: Tensor, *, temb: Optional[Tensor] = None
    ) -> tuple[Tensor, list[Tensor]]:
        states: list[Tensor] = []
        for resnet in self.resnets:
            x = resnet(x, temb=temb)
            states.append(x)
        del temb

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)

            states.append(x)

        return x, states
