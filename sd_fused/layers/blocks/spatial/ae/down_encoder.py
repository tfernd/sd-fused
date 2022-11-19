from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ....base import Module, ModuleList
from ..resampling import Downsample2D
from ..resnet import ResnetBlock2D


class DownEncoderBlock2D(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        resnet_groups: int,
        add_downsample: bool,
        downsample_padding: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.resnet_groups = resnet_groups
        self.add_downsample = add_downsample
        self.downsample_padding = downsample_padding

        self.resnets = ModuleList[ResnetBlock2D]()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels

            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    num_groups=resnet_groups,
                    num_out_groups=None,
                )
            )

        if add_downsample:
            self.downsampler = Downsample2D(in_channels, out_channels, padding=downsample_padding)
        else:
            self.downsampler = nn.Identity()

    def __call__(self, x: Tensor) -> Tensor:
        for resnet in self.resnets:
            x = resnet(x)

        x = self.downsampler(x)

        return x
