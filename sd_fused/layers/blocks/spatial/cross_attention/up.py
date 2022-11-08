from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ...transformer import SpatialTransformer
from ..resampling import Upsample2D
from ..resnet import ResnetBlock2D


class CrossAttentionUpBlock2D(nn.Module):
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

            self.attentions.append(
                SpatialTransformer(
                    in_channels=out_channels,
                    num_heads=attn_num_head_channels,
                    head_features=out_channels // attn_num_head_channels,
                    depth=1,
                    num_groups=resnet_groups,
                    context_features=cross_attention_dim,
                )
            )

        if add_upsample:
            self.upsampler = Upsample2D(out_channels)
        else:
            self.upsampler = nn.Identity()

    def __call__(
        self,
        x: Tensor,
        *,
        states: list[Tensor],
        temb: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_weights: Optional[Tensor] = None,
    ) -> Tensor:
        assert len(states) == self.num_layers

        for resnet, attn, state in zip(self.resnets, self.attentions, states):
            assert isinstance(resnet, ResnetBlock2D)
            assert isinstance(attn, SpatialTransformer)

            x = torch.cat([x, state], dim=1)
            x = resnet(x, temb=temb)
            x = attn(x, context=context, context_weights=context_weights)

        x = self.upsampler(x)

        return x
