from __future__ import annotations

from functools import partial

import torch.nn as nn
from torch import Tensor

from ..activation import SiLU
from ...layers.blocks import UpDecoderBlock2D, UNetMidBlock2D
from ..base import Conv2d, GroupNorm


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        norm_num_groups: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.norm_num_groups = norm_num_groups

        num_blocks = len(block_out_channels)

        conv = partial(Conv2d, kernel_size=3, padding=1)

        self.conv_in = conv(in_channels, block_out_channels[-1])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            num_layers=1,
            resnet_groups=norm_num_groups,
            attn_num_head_channels=None,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        self.up_blocks = nn.ModuleList()
        for i in range(num_blocks):
            is_final_block = i == num_blocks - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            block = UpDecoderBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                num_layers=layers_per_block + 1,
                resnet_groups=norm_num_groups,
                add_upsample=not is_final_block,
            )

            self.up_blocks.append(block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = GroupNorm(norm_num_groups, block_out_channels[0])
        self.conv_act = SiLU()
        self.conv_out = conv(block_out_channels[0], out_channels)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)

        # middle
        x = self.mid_block(x)

        # up
        for up_block in self.up_blocks:
            assert isinstance(up_block, UpDecoderBlock2D)

            x = up_block(x)

        # post-process
        # TODO join into single layer
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x
