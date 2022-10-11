from __future__ import annotations
from typing import Optional, Type
from typing_extensions import Self

from pathlib import Path

import re

import torch
import torch.nn as nn
from torch import Tensor

from ..layers.embedding import Timesteps, TimestepEmbedding
from ..layers.activation import SiLU
from ..layers.base import Conv2d, GroupNorm, HalfWeightsModel
from ..layers.attention import CrossAttention
from ..layers.blocks import (
    UNetMidBlock2DCrossAttention,
    DownBlock2D,
    UpBlock2D,
    CrossAttentionDownBlock2D,
    CrossAttentionUpBlock2D,
)


class UNet2DConditional(HalfWeightsModel, nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 4,
        out_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_blocks: tuple[
            Type[CrossAttentionDownBlock2D] | Type[DownBlock2D], ...
        ] = (
            CrossAttentionDownBlock2D,
            CrossAttentionDownBlock2D,
            CrossAttentionDownBlock2D,
            DownBlock2D,
        ),
        up_blocks: tuple[
            Type[CrossAttentionUpBlock2D] | Type[UpBlock2D], ...
        ] = (
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
                        resnet_groups=norm_num_groups,
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
        self.conv_norm_out = GroupNorm(norm_num_groups, block_out_channels[0])
        self.conv_act = SiLU()
        self.conv_out = Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    def forward(
        self, x: Tensor, timestep: int | Tensor, *, context: Tensor
    ) -> Tensor:
        B, C, H, W = x.shape

        # 1. time embedding
        if isinstance(timestep, int):
            timesteps = torch.tensor([timestep], device=x.device)
            timesteps = timesteps.expand(B)
        else:
            assert timestep.shape == (B,)
            timesteps = timestep

        temb = self.time_proj(timesteps)
        temb = self.time_embedding(temb)

        # 2. pre-process
        x = self.conv_in(x)

        # 3. down
        # TODO it is possible to make it a list[list[Tensor]]? or is the number of elements wrong?
        all_states: list[Tensor] = [x]
        for block in self.down_blocks:
            assert isinstance(block, (CrossAttentionDownBlock2D, DownBlock2D))

            if isinstance(block, CrossAttentionDownBlock2D):
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
            assert isinstance(block, (CrossAttentionUpBlock2D, UpBlock2D))

            states = tuple(all_states.pop() for _ in range(block.num_layers))

            if isinstance(block, CrossAttentionUpBlock2D):
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

        changes: list[tuple[str, str]] = [
            # Cross-attention
            (
                r"transformer_blocks.(\d).norm([12]).(weight|bias)",
                r"transformer_blocks.\1.attn\2.norm.\3",
            ),
            ## FeedForward (norm)
            (
                r"transformer_blocks.(\d).norm3.(weight|bias)",
                r"transformer_blocks.\1.ff.0.\2",
            ),
            ## FeedForward (geglu)
            (r"ff.net.0.proj.(weight|bias)", r"ff.1.proj.\1",),
            ## FeedForward-Linear
            (r"ff.net.2.(weight|bias)", r"ff.2.\1",),
            # up/down samplers
            (r"(up|down)samplers.0", r"\1sampler"),
            # TimeEmbedding
            (r"time_embedding.linear_1.(weight|bias)", r"time_embedding.0.\1"),
            (r"time_embedding.linear_2.(weight|bias)", r"time_embedding.2.\1"),
        ]

        # modify state-dict
        for key in list(state.keys()):
            for (c1, c2) in changes:
                new_key = re.sub(c1, c2, key)
                if new_key != key:
                    # print(f"Changing {key} -> {new_key}")
                    value = state.pop(key)
                    state[new_key] = value

        # debug
        # old_keys = list(state.keys())
        # new_keys = list(model.state_dict().keys())

        # in_old = set(old_keys) - set(new_keys)
        # in_new = set(new_keys) - set(old_keys)

        # with open("in-old.txt", "w") as f:
        #     f.write("\n".join(sorted(list(in_old))))

        # with open("in-new.txt", "w") as f:
        #     f.write("\n".join(sorted(list(in_new))))

        model.load_state_dict(state)

        return model

    # TODO add to its own Class and add support for SelfAttention
    # TODO copy to the vae
    def split_attention(
        self, cross_attention_chunks: Optional[int] = None
    ) -> None:
        """Split cross/self-attention computation into chunks."""

        if cross_attention_chunks is not None:
            assert cross_attention_chunks >= 1

        for name, module in self.named_modules():
            if isinstance(module, CrossAttention):
                module.attention_chunks = cross_attention_chunks
