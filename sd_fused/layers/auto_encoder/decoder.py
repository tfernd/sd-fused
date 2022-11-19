from __future__ import annotations

from torch import Tensor

from ..base import Module, ModuleList
from ..basic import Conv2d
from ..blocks.basic import GroupNormSiLUConv2d
from ..blocks.spatial import UpDecoderBlock2D, UNetMidBlock2DSelfAttention


class Decoder(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        resnet_groups: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.resnet_groups = resnet_groups

        num_blocks = len(block_out_channels)

        self.pre_process = Conv2d(in_channels, block_out_channels[-1], kernel_size=3, padding=1)

        # mid
        self.mid_block = UNetMidBlock2DSelfAttention(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            num_layers=1,
            resnet_groups=resnet_groups,
            attn_num_head_channels=None,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        self.up_blocks = ModuleList[UpDecoderBlock2D]()
        for i in range(num_blocks):
            is_final_block = i == num_blocks - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            block = UpDecoderBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                num_layers=layers_per_block + 1,
                resnet_groups=resnet_groups,
                add_upsample=not is_final_block,
            )

            self.up_blocks.append(block)
            prev_output_channel = output_channel

        # out
        self.post_process = GroupNormSiLUConv2d(
            resnet_groups,
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, x: Tensor) -> Tensor:
        x = self.pre_process(x)
        x = self.mid_block(x)

        for up_block in self.up_blocks:
            x = up_block(x)

        return self.post_process(x)
