from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..resampling import Upsample2D
from ..attention import SpatialTransformer
from .resnet_block2d import ResnetBlock2D


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        num_layers: int,
        resnet_groups: int,
        attn_num_head_channels: int,
        cross_attention_dim: int,
        add_upsample: bool,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prev_output_channel = prev_output_channel
        self.temb_channels = temb_channels
        self.num_layers = num_layers
        self.resnet_groups = resnet_groups
        self.attn_num_head_channels = attn_num_head_channels
        self.cross_attention_dim = cross_attention_dim
        self.add_upsample = add_upsample

        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for i in range(num_layers):
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

            self.attentions.append(
                SpatialTransformer(
                    in_channels=out_channels,
                    num_heads=attn_num_head_channels,
                    dim_head=out_channels // attn_num_head_channels,
                    depth=1,
                    num_groups=resnet_groups,
                    context_dim=cross_attention_dim,
                )
            )

        # TODO remove ModuleList?
        self.upsamplers = nn.ModuleList()
        if add_upsample:
            self.upsamplers.append(
                Upsample2D(
                    channels=out_channels,
                    out_channels=out_channels,
                    kind="conv",
                )
            )

    def forward(
        self,
        x: Tensor,
        *,
        states: list[Tensor],
        temb: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> Tensor:
        assert len(states) == self.num_layers

        for resnet, attn, state in zip(self.resnets, self.attentions, states):
            x = torch.cat([x, state], dim=1)
            del state
            x = resnet(x, temb=temb)
            x = attn(x, context=context)
        del states, temb, context

        for upsampler in self.upsamplers:
            x = upsampler(x, size=size)

        return x
