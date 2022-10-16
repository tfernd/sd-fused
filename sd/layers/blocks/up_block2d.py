from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..resampling import Upsample2D, TensorSize
from .resnet_block2d import ResnetBlock2D


class UpBlock2D(nn.Module):
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

        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            # TODO make more redable
            res_skip_channels = (
                in_channels if (i == num_layers - 1) else out_channels
            )
            resnet_in_channels = (
                prev_output_channel if i == 0 else out_channels
            )

            self.resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    groups_out=None,
                )
            )

        if add_upsample:
            self.upsampler = Upsample2D(
                channels=out_channels, kind="conv", out_channels=None,
            )
        else:
            self.upsampler = None  # TODO nn.Identity()

    def __call__(
        self,
        x: Tensor,
        *,
        states: list[Tensor],
        temb: Optional[Tensor] = None,
        # TODO remove size, not used
        size: Optional[TensorSize] = None,
    ) -> Tensor:
        assert len(states) == self.num_layers

        for resnet, state in zip(self.resnets, states):
            assert isinstance(resnet, ResnetBlock2D)

            x = torch.cat([x, state], dim=1)
            del state
            x = resnet(x, temb=temb)
        del states, temb

        if self.upsampler is not None:
            x = self.upsampler(x, size=size)

        return x
