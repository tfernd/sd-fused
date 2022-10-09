from __future__ import annotations

from functools import partial

import torch.nn as nn
from torch import Tensor

from ...layers.blocks import DownEncoderBlock2D, UNetMidBlock2D
from ..activation import SiLU
from ..base import Conv2d, GroupNorm


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        norm_num_groups: int,
        double_z: bool,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.norm_num_groups = norm_num_groups
        self.double_z = double_z

        num_blocks = len(block_out_channels)

        conv = partial(Conv2d, kernel_size=3, padding=1)

        self.conv_in = conv(in_channels, block_out_channels[0])

        # down
        output_channel = block_out_channels[0]
        self.down_blocks = nn.ModuleList()
        for i in range(num_blocks):
            is_final_block = i == num_blocks - 1

            input_channel = output_channel
            output_channel = block_out_channels[i]

            block = DownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block,
                resnet_groups=norm_num_groups,
                downsample_padding=0,
                add_downsample=not is_final_block,
            )
            self.down_blocks.append(block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            num_layers=1,
            resnet_groups=norm_num_groups,
            attn_num_head_channels=None,
        )

        # out
        self.conv_norm_out = GroupNorm(norm_num_groups, block_out_channels[-1])
        self.conv_act = SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = conv(block_out_channels[-1], conv_out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)

        # down
        for down_block in self.down_blocks:
            x = down_block(x)

        # middle
        x = self.mid_block(x)

        # post-process
        # TODO Join into single layer
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x
