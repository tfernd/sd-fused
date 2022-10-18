from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ..attention import SpatialTransformer
from .resnet_block2d import ResnetBlock2D


class UNetMidBlock2DCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        temb_channels: int,
        num_layers: int,
        resnet_groups: int,
        attn_num_head_channels: int,
        cross_attention_dim: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.temb_channels = temb_channels
        self.num_layers = num_layers
        self.resnet_groups = resnet_groups
        self.attn_num_head_channels = attn_num_head_channels
        self.cross_attention_dim = cross_attention_dim

        resnet_groups = resnet_groups or min(in_channels // 4, 32)

        self.attentions = nn.ModuleList()
        self.resnets = nn.ModuleList()
        for i in range(num_layers + 1):
            if i > 0:
                self.attentions.append(
                    SpatialTransformer(
                        in_channels=in_channels,
                        num_heads=attn_num_head_channels,
                        dim_head=in_channels // attn_num_head_channels,
                        depth=1,
                        num_groups=resnet_groups,
                        context_dim=cross_attention_dim,
                    )
                )

            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    groups_out=None,
                )
            )

    def __call__(
        self,
        x: Tensor,
        *,
        temb: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_weights: Optional[Tensor] = None,
    ) -> Tensor:
        first_resnet, *rest_resnets = self.resnets

        x = first_resnet(x, temb=temb)

        for attn, resnet in zip(self.attentions, rest_resnets):
            assert isinstance(attn, SpatialTransformer)
            assert isinstance(resnet, ResnetBlock2D)

            x = attn(x, context=context, context_weights=context_weights)
            x = resnet(x, temb=temb)
        del context, temb

        return x
