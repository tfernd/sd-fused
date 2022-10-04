from __future__ import annotations
from typing import Optional, Type
from typing_extensions import Self

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from ..layers.embedding import Timesteps, TimestepEmbedding
from ..layers.activation import SiLU
from ..layers.base import Conv2d, GroupNorm, HalfWeightsModel, InPlaceModel
from ..layers.attention import CrossAttention
from ..layers.blocks import (
    UNetMidBlock2DCrossAttn,
    DownBlock2D,
    UpBlock2D,
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
)
from ..utils import FLASH_ATTENTION


class UNet2DConditional(InPlaceModel, HalfWeightsModel, nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 4,
        out_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_blocks: tuple[
            Type[CrossAttnDownBlock2D] | Type[DownBlock2D], ...
        ] = (
            CrossAttnDownBlock2D,
            CrossAttnDownBlock2D,
            CrossAttnDownBlock2D,
            DownBlock2D,
        ),
        up_blocks: tuple[Type[CrossAttnUpBlock2D] | Type[UpBlock2D], ...] = (
            UpBlock2D,
            CrossAttnUpBlock2D,
            CrossAttnUpBlock2D,
            CrossAttnUpBlock2D,
        ),
        block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        norm_num_groups: int = 32,
        cross_attention_dim: int = 768,
        attention_head_dim: int = 8,
    ) -> None:
        super().__init__()

        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        # time
        self.time_proj = Timesteps(
            num_channels=block_out_channels[0],
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=freq_shift,
        )
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            channel=timestep_input_dim, time_embed_dim=time_embed_dim,
        )

        # down
        output_channel = block_out_channels[0]
        self.down_blocks = nn.ModuleList()
        for i, block in enumerate(down_blocks):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if block == CrossAttnDownBlock2D:
                self.down_blocks.append(
                    CrossAttnDownBlock2D(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        temb_channels=time_embed_dim,
                        num_layers=layers_per_block,
                        resnet_groups=norm_num_groups,
                        attn_num_head_channels=attention_head_dim,
                        cross_attention_dim=cross_attention_dim,
                        downsample_padding=downsample_padding,
                        add_downsample=not is_final_block,
                    )
                )
            elif block == DownBlock2D:
                self.down_blocks.append(
                    DownBlock2D(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        temb_channels=time_embed_dim,
                        num_layers=layers_per_block,
                        resnet_groups=norm_num_groups,
                        add_downsample=not is_final_block,
                        downsample_padding=downsample_padding,
                    )
                )

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            num_layers=1,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        self.up_blocks = nn.ModuleList()
        for i, block in enumerate(up_blocks):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            is_final_block = i == len(block_out_channels) - 1

            if block == CrossAttnUpBlock2D:
                self.up_blocks.append(
                    CrossAttnUpBlock2D(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        prev_output_channel=prev_output_channel,
                        num_layers=layers_per_block + 1,
                        temb_channels=time_embed_dim,
                        add_upsample=not is_final_block,
                        resnet_groups=norm_num_groups,
                        cross_attention_dim=cross_attention_dim,
                        attn_num_head_channels=attention_head_dim,
                    )
                )
            elif block == UpBlock2D:
                self.up_blocks.append(
                    UpBlock2D(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        prev_output_channel=prev_output_channel,
                        num_layers=layers_per_block + 1,
                        temb_channels=time_embed_dim,
                        add_upsample=not is_final_block,
                        resnet_groups=norm_num_groups,
                    )
                )
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = GroupNorm(norm_num_groups, block_out_channels[0])
        self.conv_act = SiLU()
        self.conv_out = Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    def forward(self, x: Tensor, timestep: int, *, context: Tensor) -> Tensor:
        B, C, H, W = x.shape

        # 1. time embedding
        timesteps = torch.tensor([timestep], device=x.device)
        timesteps = timesteps.expand(B)

        temb = self.time_proj(timesteps)
        temb = self.time_embedding(temb)

        # 2. pre-process
        x = self.conv_in(x)

        # 3. down
        # TODO it is possible to make it a list[list[Tensor]]? or is the number of elements wrong?
        all_states: list[Tensor] = [x]
        for block in self.down_blocks:
            assert isinstance(block, (CrossAttnDownBlock2D, DownBlock2D))

            if isinstance(block, CrossAttnDownBlock2D):
                x, states = block(x, temb=temb, context=context)
            elif isinstance(block, DownBlock2D):
                x, states = block(x, temb=temb)
            else:
                raise ValueError

            all_states.extend(states)
            del states

        # 4. mid
        x = self.mid_block(x, temb=temb, context=context)

        # 5. up
        for block in self.up_blocks:
            assert isinstance(block, (CrossAttnUpBlock2D, UpBlock2D))

            states = tuple(all_states.pop() for _ in range(block.num_layers))

            if isinstance(block, CrossAttnUpBlock2D):
                x = block(x, states=states, temb=temb, context=context)
            elif isinstance(block, UpBlock2D):
                x = block(x, states=states, temb=temb)
            else:
                raise ValueError

            del states
        del all_states

        # 6. post-process
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x

    @classmethod
    def load_sd(cls, path: str | Path) -> Self:
        """Load Stable-Diffusion."""

        path = Path(path)
        paths = list(path.glob("*.bin"))
        assert len(paths) == 1
        path = paths[0]

        state = torch.load(path, map_location="cpu")
        model = cls()
        model.load_state_dict(state)

        return model

    def split_attention(
        self, cross_attention_chunks: Optional[int] = None
    ) -> None:
        if cross_attention_chunks is not None:
            assert cross_attention_chunks >= 1

        for name, module in self.named_modules():
            if isinstance(module, CrossAttention):
                if cross_attention_chunks is not None:
                    module.flash_attention = False

                module.split_attention_chunks = cross_attention_chunks

    # TODO FIX
    # def flash_attention(self, flash: bool = True) -> None:
    #     global FLASH_ATTENTION
    #     for name, module in self.named_modules():
    #         if isinstance(module, CrossAttention):
    #             if flash:
    #                 # assert FLASH_ATTENTION # ! temp
    #                 module.split_attention_chunks = None

    #             module.flash_attention = flash
