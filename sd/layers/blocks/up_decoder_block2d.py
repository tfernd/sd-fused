from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ..resampling import Upsample2D
from .resnet_block2d import ResnetBlock2D


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        resnet_groups: int,
        add_upsample: bool,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.resnet_groups = resnet_groups
        self.add_upsample = add_upsample

        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            self.resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    groups=resnet_groups,
                    groups_out=None,
                )
            )

        # TODO remove ModuleList?
        self.upsamplers = nn.ModuleList()
        if add_upsample:
            self.upsamplers.append(
                Upsample2D(
                    channels=out_channels,
                    kind="conv",
                    out_channels=out_channels,
                )
            )

    def forward(
        self, x: Tensor, size: Optional[tuple[int, int]] = None
    ) -> Tensor:
        for resnet in self.resnets:
            x = resnet(x)

        for upsampler in self.upsamplers:
            x = upsampler(x, size=size)

        return x
