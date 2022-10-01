from __future__ import annotations
from typing import Optional
from typing_extensions import Self

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from ..utils import normalize, denormalize
from ..layers.base import Conv2d, HalfWeightsModel
from ..layers.distribution import DiagonalGaussianDistribution
from ..layers.auto_encoder.encoder import Encoder
from ..layers.auto_encoder.decoder import Decoder


class AutoencoderKL(HalfWeightsModel, nn.Module):
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

        self.quant_conv = Conv2d(2 * latent_channels)
        self.post_quant_conv = Conv2d(latent_channels)

    @torch.no_grad()
    def encode(self, x: Tensor) -> DiagonalGaussianDistribution:
        device = next(self.encoder.parameters()).device

        x = x.to(device)

        x = normalize(x)
        x = self.encoder(x)

        moments = self.quant_conv(x)
        mean, logvar = moments.chunk(2, dim=1)

        return DiagonalGaussianDistribution(mean, logvar)

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        device = next(self.decoder.parameters()).device

        z = z.to(device)

        z = self.post_quant_conv(z)
        out = self.decoder(z)

        out = denormalize(out)

        return out

    def forward(
        self,
        x: Tensor,
        *,
        sample: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        posterior = self.encode(x)

        z = posterior.sample(generator) if sample else posterior.mode

        return self.decode(z)

    @classmethod
    def load_sd(cls, path: str | Path) -> Self:
        """Load Stable-Diffusion."""

        path = Path(path)
        paths = list(path.glob("*.bin"))
        assert len(paths) == 1
        path = paths[0]

        state = torch.load(path, map_location="cpu")
        model = cls()
        model.load_state_dict(state)

        return model
