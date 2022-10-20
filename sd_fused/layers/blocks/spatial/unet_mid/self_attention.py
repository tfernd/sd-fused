from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ...attention import SelfAttention
from ..resnet import ResnetBlock2D


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        temb_channels: Optional[int],
        num_layers: int,
        resnet_groups: int,
        attn_num_head_channels: Optional[int],
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.temb_channels = temb_channels
        self.num_layers = num_layers
        self.resnet_groups = resnet_groups
        self.attn_num_head_channels = attn_num_head_channels

        resnet_groups = resnet_groups or min(in_channels // 4, 32)

        self.attentions = nn.ModuleList()
        self.resnets = nn.ModuleList()
        for i in range(num_layers + 1):
            if i > 0:
                self.attentions.append(
                    SelfAttention(
                        in_features=in_channels,
                        num_groups=resnet_groups,
                        head_features=attn_num_head_channels,
                    )
                )

            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    num_groups=resnet_groups,
                    num_out_groups=None,
                )
            )

    def __call__(self, x: Tensor, *, temb: Optional[Tensor] = None) -> Tensor:
        first_resnet, *rest_resnets = self.resnets

        x = first_resnet(x, temb=temb)

        for attn, resnet in zip(self.attentions, rest_resnets):
            assert isinstance(attn, SelfAttention)
            assert isinstance(resnet, ResnetBlock2D)

            x = attn(x)
            x = resnet(x, temb=temb)
        del temb

        return x
