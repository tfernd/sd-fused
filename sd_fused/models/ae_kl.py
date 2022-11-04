from __future__ import annotations
from typing_extensions import Self

from pathlib import Path
import json

import torch
import torch.nn as nn
from torch import Tensor

from ..utils import normalize, denormalize
from ..layers.base import (
    Conv2d,
    HalfWeightsModel,
    SplitAttentionModel,
    FlashAttentionModel,
)
from ..layers.distribution import DiagonalGaussianDistribution
from ..layers.auto_encoder import Encoder, Decoder
from .config import VaeConfig
from .convert import diffusers2fused_vae
from .convert.states import debug_state_replacements


class AutoencoderKL(
    HalfWeightsModel,
    SplitAttentionModel,
    FlashAttentionModel,
    nn.Module,
):
    @classmethod
    def from_config(cls, path: str | Path) -> Self:
        """Creates a model from a config file."""

        path = Path(path)
        if path.is_dir():
            path /= "config.json"
        assert path.suffix == ".json"

        db = json.load(open(path, "r"))
        config = VaeConfig(**db)
        # TODO raise an exception?

        return cls(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            latent_channels=config.latent_channels,
            norm_num_groups=config.norm_num_groups,
        )

    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        latent_channels: int = 4,
        norm_num_groups: int = 32,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.latent_channels = latent_channels
        self.norm_num_groups = norm_num_groups

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )

        # TODO very bad names...
        self.quant_conv = Conv2d(2 * latent_channels)
        self.post_quant_conv = Conv2d(latent_channels)

    def encode(self, x: Tensor) -> DiagonalGaussianDistribution:
        """Encode an byte-Tensor into a posterior distribution."""

        x = normalize(x)
        x = self.encoder(x)

        moments = self.quant_conv(x)
        mean, logvar = moments.chunk(2, dim=1)

        return DiagonalGaussianDistribution(mean, logvar)

    def decode(self, z: Tensor) -> Tensor:
        """Decode the latent's space into an image."""

        z = self.post_quant_conv(z)
        out = self.decoder(z)

        out = denormalize(out)

        return out

    @classmethod
    def from_diffusers(cls, path: str | Path) -> Self:
        """Load Stable-Diffusion from diffusers checkpoint."""

        path = Path(path)
        model = cls.from_config(path)

        state_path = next(path.glob("*.bin"))
        old_state = torch.load(state_path, map_location="cpu")
        replaced_state = diffusers2fused_vae(old_state)

        # debug_state_replacements(model.state_dict(), replaced_state)

        model.load_state_dict(replaced_state)

        return model
