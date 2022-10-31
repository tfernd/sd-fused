from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ..resampling import Upsample2D
from ..resnet import ResnetBlock2D


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
                    num_groups=resnet_groups,
                    num_out_groups=None,
                )
            )

        if add_upsample:
            self.upsampler = Upsample2D(
                out_channels,
                out_channels,
            )
        else:
            self.upsampler = nn.Identity()

    def __call__(self, x: Tensor) -> Tensor:
        for resnet in self.resnets:
            assert isinstance(resnet, ResnetBlock2D)

            x = resnet(x)

        x = self.upsampler(x)

        return x
