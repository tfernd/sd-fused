from __future__ import annotations

from torch import Tensor

from ..base import Module, ModuleList
from ..basic import Conv2d
from ..blocks.basic import GroupNormSiLUConv2d
from ..blocks.spatial import DownEncoderBlock2D, UNetMidBlock2DSelfAttention


class Encoder(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        resnet_groups: int,
        double_z: bool,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.resnet_groups = resnet_groups
        self.double_z = double_z

        num_blocks = len(block_out_channels)

        self.pre_process = Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # down
        output_channel = block_out_channels[0]
        self.down_blocks = ModuleList()
        for i in range(num_blocks):
            is_final_block = i == num_blocks - 1

            input_channel = output_channel
            output_channel = block_out_channels[i]

            block = DownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block,
                resnet_groups=resnet_groups,
                downsample_padding=0,
                add_downsample=not is_final_block,
            )
            self.down_blocks.append(block)

        # mid
        self.mid_block = UNetMidBlock2DSelfAttention(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            num_layers=1,
            resnet_groups=resnet_groups,
            attn_num_head_channels=None,
        )

        # out
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.post_process = GroupNormSiLUConv2d(
            resnet_groups,
            block_out_channels[-1],
            conv_out_channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, x: Tensor) -> Tensor:
        x = self.pre_process(x)

        for down_block in self.down_blocks:
            assert isinstance(down_block, DownEncoderBlock2D)

            x = down_block(x)

        x = self.mid_block(x)

        return self.post_process(x)
