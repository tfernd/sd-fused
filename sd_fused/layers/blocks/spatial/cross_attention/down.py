from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ....base import Module, ModuleList
from ...transformer import SpatialTransformer
from ..resampling import Downsample2D
from ..resnet import ResnetBlock2D
from ..output_states import OutputStates


class CrossAttentionDownBlock2D(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int,
        resnet_groups: int,
        attn_num_head_channels: int,
        cross_attention_dim: int,
        downsample_padding: int,
        add_downsample: bool,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.num_layers = num_layers
        self.resnet_groups = resnet_groups
        self.attn_num_head_channels = attn_num_head_channels
        self.cross_attention_dim = cross_attention_dim
        self.downsample_padding = downsample_padding
        self.add_downsample = add_downsample

        self.resnets = ModuleList()
        self.attentions = ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels

            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
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

        if add_downsample:
            self.downsampler = Downsample2D(in_channels, out_channels, padding=downsample_padding)
        else:
            self.downsampler = None

    def __call__(
        self,
        x: Tensor,
        *,
        temb: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> OutputStates:
        states: list[Tensor] = []
        for resnet, attn in zip(self.resnets, self.attentions):
            assert isinstance(resnet, ResnetBlock2D)
            assert isinstance(attn, SpatialTransformer)

            x = resnet(x, temb=temb)
            x = attn(x, context=context, weights=weights)

            states.append(x)

        if self.downsampler is not None:
            x = self.downsampler(x)

            states.append(x)

        return OutputStates(x, states)
