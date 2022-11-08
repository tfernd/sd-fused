from __future__ import annotations
from typing import Optional, Type
from typing_extensions import Self

from pathlib import Path
import json

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.tensors import to_tensor
from ..layers.embedding import Timesteps, TimestepEmbedding
from ..layers.base import (
    Conv2d,
    HalfWeightsModel,
    SplitAttentionModel,
    FlashAttentionModel,
    ToMe,
)
from ..layers.blocks.simple import GroupNormSiLUConv2d
from ..layers.blocks.spatial import (
    UNetMidBlock2DCrossAttention,
    DownBlock2D,
    UpBlock2D,
    CrossAttentionDownBlock2D,
    CrossAttentionUpBlock2D,
)
from .config import UnetConfig
from .convert import diffusers2fused_unet
from .convert.states import debug_state_replacements


class UNet2DConditional(
    HalfWeightsModel,
    SplitAttentionModel,
    FlashAttentionModel,
    ToMe,
    nn.Module,
):
    @classmethod
    def from_config(cls, path: str | Path) -> Self:
        """Creates a model from a (diffusers) config file."""

        path = Path(path)
        if path.is_dir():
            path /= "config.json"
        assert path.suffix == ".json"

        db = json.load(open(path, "r"))
        config = UnetConfig(**db)

        return cls(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            flip_sin_to_cos=config.flip_sin_to_cos,
            freq_shift=config.freq_shift,
            down_blocks=config.down_blocks,
            up_blocks=config.up_blocks,
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            downsample_padding=config.downsample_padding,
            norm_num_groups=config.norm_num_groups,
            cross_attention_dim=config.cross_attention_dim,
            attention_head_dim=config.attention_head_dim,
        )

    def __init__(
        self,
        *,
        in_channels: int = 4,
        out_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_blocks: tuple[Type[CrossAttentionDownBlock2D] | Type[DownBlock2D], ...] = (
            CrossAttentionDownBlock2D,
            CrossAttentionDownBlock2D,
            CrossAttentionDownBlock2D,
            DownBlock2D,
        ),
        up_blocks: tuple[Type[CrossAttentionUpBlock2D] | Type[UpBlock2D], ...] = (
            UpBlock2D,
            CrossAttentionUpBlock2D,
            CrossAttentionUpBlock2D,
            CrossAttentionUpBlock2D,
        ),
        block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        norm_num_groups: int = 32,
        cross_attention_dim: int = 768,
        attention_head_dim: int = 8,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.freq_shift = freq_shift
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.downsample_padding = downsample_padding
        self.norm_num_groups = norm_num_groups
        self.cross_attention_dim = cross_attention_dim
        self.attention_head_dim = attention_head_dim

        time_embed_dim = block_out_channels[0] * 4

        # input
        self.pre_process = Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # time
        self.time_proj = Timesteps(
            num_channels=block_out_channels[0],
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=freq_shift,
        )
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(channel=timestep_input_dim, time_embed_dim=time_embed_dim)

        # down
        output_channel = block_out_channels[0]
        self.down_blocks = nn.ModuleList()
        for i, block in enumerate(down_blocks):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if block == CrossAttentionDownBlock2D:
                self.down_blocks.append(
                    CrossAttentionDownBlock2D(
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
                        num_groups=norm_num_groups,
                        add_downsample=not is_final_block,
                        downsample_padding=downsample_padding,
                    )
                )

        # mid
        self.mid_block = UNetMidBlock2DCrossAttention(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            num_layers=1,
        )

        # up
        reversed_block_out_channels = tuple(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        self.up_blocks = nn.ModuleList()
        for i, block in enumerate(up_blocks):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            if block == CrossAttentionUpBlock2D:
                self.up_blocks.append(
                    CrossAttentionUpBlock2D(
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
        self.post_process = GroupNormSiLUConv2d(
            norm_num_groups,
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(
        self,
        x: Tensor,
        timestep: int | Tensor,
        context: Tensor,
        context_weights: Optional[Tensor] = None,
    ) -> Tensor:
        B, C, H, W = x.shape

        # 1. time embedding
        timestep = to_tensor(timestep, device=x.device, dtype=x.dtype)
        if timestep.size(0) != B:
            assert timestep.size(0) == 1
            timestep = timestep.expand(B)

        temb = self.time_proj(timestep)
        temb = self.time_embedding(temb)

        # 2. pre-process
        x = self.pre_process(x)

        # 3. down
        # TODO it is possible to make it a list[list[Tensor]]? or is the number of elements wrong?
        all_states: list[Tensor] = [x]
        for block in self.down_blocks:
            if isinstance(block, CrossAttentionDownBlock2D):
                x, states = block(x, temb=temb, context=context, context_weights=context_weights)
            elif isinstance(block, DownBlock2D):
                x, states = block(x, temb=temb)
            else:
                raise ValueError

            all_states.extend(states)
            del states

        # 4. mid
        x = self.mid_block(x, temb=temb, context=context, context_weights=context_weights)

        # 5. up
        for block in self.up_blocks:
            assert isinstance(block, (CrossAttentionUpBlock2D, UpBlock2D))

            # ! I don't like the construction of this...
            states = list(all_states.pop() for _ in range(block.num_layers))

            if isinstance(block, CrossAttentionUpBlock2D):
                x = block(x, states=states, temb=temb, context=context, context_weights=context_weights)
            elif isinstance(block, UpBlock2D):
                x = block(x, states=states, temb=temb)

            del states
        del all_states

        # 6. post-process
        x = self.post_process(x)

        return x

    @classmethod
    def from_diffusers(cls, path: str | Path) -> Self:
        """Load Stable-Diffusion from diffusers checkpoint folder."""

        path = Path(path)
        model = cls.from_config(path)

        state_path = next(path.glob("*.bin"))
        old_state = torch.load(state_path, map_location="cpu")
        replaced_state = diffusers2fused_unet(old_state)

        # debug_state_replacements(model.state_dict(), replaced_state)

        model.load_state_dict(replaced_state)

        return model
