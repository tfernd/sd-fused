from __future__ import annotations
from typing import Optional

from torch import Tensor

from ....base import Module, ModuleList
from ...transformer import SpatialTransformer
from ..resnet import ResnetBlock2D


class UNetMidBlock2DCrossAttention(Module):
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

        self.attentions = ModuleList()
        self.resnets = ModuleList()
        for i in range(num_layers + 1):
            if i > 0:
                self.attentions.append(
                    SpatialTransformer(
                        in_channels=in_channels,
                        num_heads=attn_num_head_channels,
                        head_features=in_channels // attn_num_head_channels,
                        depth=1,
                        num_groups=resnet_groups,
                        context_features=cross_attention_dim,
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

    def __call__(
        self,
        x: Tensor,
        *,
        temb: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        first_resnet, *rest_resnets = self.resnets

        x = first_resnet(x, temb=temb)

        for attn, resnet in zip(self.attentions, rest_resnets):
            assert isinstance(attn, SpatialTransformer)
            assert isinstance(resnet, ResnetBlock2D)

            x = attn(x, context=context, weights=weights)
            x = resnet(x, temb=temb)

        return x
