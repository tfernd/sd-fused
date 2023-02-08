from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from ....base import Module, ModuleList
from ....basic import Identity
from ..resampling import Upsample2D
from ..resnet import ResnetBlock2D


class UpBlock2D(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: Optional[int],
        num_layers: int,
        resnet_groups: int,
        add_upsample: bool,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.prev_output_channel = prev_output_channel
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.num_layers = num_layers
        self.resnet_groups = resnet_groups
        self.add_upsample = add_upsample

        self.resnets = ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                res_skip_channels = in_channels
            else:
                res_skip_channels = out_channels

            if i == 0:
                resnet_in_channels = prev_output_channel
            else:
                resnet_in_channels = out_channels

            self.resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    num_groups=resnet_groups,
                    num_out_groups=None,
                )
            )

        if add_upsample:
            self.upsampler = Upsample2D(out_channels)
        else:
            self.upsampler = Identity()

    def __call__(
        self,
        x: Tensor,
        *,
        states: list[Tensor],
        temb: Optional[Tensor] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> Tensor:
        assert len(states) == self.num_layers

        for resnet, state in zip(self.resnets, states):
            assert isinstance(resnet, ResnetBlock2D)

            x = torch.cat([x, state], dim=1)
            x = resnet(x, temb=temb)

        x = self.upsampler(x, size=size)

        return x
