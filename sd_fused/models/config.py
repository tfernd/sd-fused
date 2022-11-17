from __future__ import annotations
from typing import Type

from dataclasses import dataclass

from ..layers.blocks.spatial import DownBlock2D, UpBlock2D, CrossAttentionDownBlock2D, CrossAttentionUpBlock2D


@dataclass
class VaeConfig:
    _class_name: str
    _diffusers_version: str
    act_fn: str
    in_channels: int
    latent_channels: int
    layers_per_block: int
    out_channels: int
    sample_size: int
    block_out_channels: list[int]
    down_block_types: list[str]
    up_block_types: list[str]
    norm_num_groups: int = 32

    def __post_init__(self) -> None:
        assert self._class_name == "AutoencoderKL"
        assert self.act_fn == "silu"

        for block in self.down_block_types:
            assert block == "DownEncoderBlock2D"

        for block in self.up_block_types:
            assert block == "UpDecoderBlock2D"


@dataclass
class UnetConfig:
    _class_name: str
    _diffusers_version: str
    act_fn: str
    attention_head_dim: int
    block_out_channels: list[int]
    center_input_sample: bool
    cross_attention_dim: int
    down_block_types: list[str]
    downsample_padding: int
    flip_sin_to_cos: bool
    freq_shift: int
    in_channels: int
    layers_per_block: int
    mid_block_scale_factor: int
    norm_eps: float
    norm_num_groups: int
    out_channels: int
    sample_size: int
    up_block_types: list[str]

    def __post_init__(self) -> None:
        assert self._class_name == "UNet2DConditionModel"
        assert self.act_fn == "silu"

        for block in self.down_block_types:
            assert block in ("CrossAttnDownBlock2D", "DownBlock2D")
        for block in self.up_block_types:
            assert block in ("UpBlock2D", "CrossAttnUpBlock2D")

    @property
    def down_blocks(
        self,
    ) -> tuple[Type[CrossAttentionDownBlock2D] | Type[DownBlock2D], ...]:
        def get_block(
            block: str,
        ) -> Type[CrossAttentionDownBlock2D] | Type[DownBlock2D]:
            if block == "CrossAttnDownBlock2D":
                return CrossAttentionDownBlock2D
            if block == "DownBlock2D":
                return DownBlock2D

            raise ValueError

        return tuple(get_block(block) for block in self.down_block_types)

    @property
    def up_blocks(
        self,
    ) -> tuple[Type[CrossAttentionUpBlock2D] | Type[UpBlock2D], ...]:
        def get_block(
            block: str,
        ) -> Type[CrossAttentionUpBlock2D] | Type[UpBlock2D]:
            if block == "CrossAttnUpBlock2D":
                return CrossAttentionUpBlock2D
            if block == "UpBlock2D":
                return UpBlock2D

            raise ValueError("Invalid")

        return tuple(get_block(block) for block in self.up_block_types)
