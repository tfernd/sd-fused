from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ..resampling import Downsample2D
from ..resnet import ResnetBlock2D
from ..output_states import OutputStates


class DownBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        temb_channels: Optional[int],
        num_layers: int,
        num_groups: int,
        add_downsample: bool,
        downsample_padding: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.num_layers = num_layers
        self.num_groups = num_groups
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
                    num_groups=num_groups,
                    num_out_groups=None,
                )
            )

        if add_downsample:
            self.downsampler = Downsample2D(
                in_channels,
                out_channels,
                padding=downsample_padding,
            )
        else:
            self.downsampler = None

    def __call__(
        self,
        x: Tensor,
        *,
        temb: Optional[Tensor] = None,
    ) -> OutputStates:
        states: list[Tensor] = []
        for resnet in self.resnets:
            assert isinstance(resnet, ResnetBlock2D)

            x = resnet(x, temb=temb)
            states.append(x)

        if self.downsampler is not None:
            x = self.downsampler(x)

            states.append(x)

        return OutputStates(x, states)
