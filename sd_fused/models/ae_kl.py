from __future__ import annotations
from typing_extensions import Self

from pathlib import Path
import re

import torch
import torch.nn as nn
from torch import Tensor

from ..utils import normalize, denormalize
from ..layers.base import Conv2d, HalfWeightsModel
from ..layers.distribution import DiagonalGaussianDistribution
from ..layers.auto_encoder.encoder import Encoder
from ..layers.auto_encoder.decoder import Decoder


class AutoencoderKL(HalfWeightsModel, nn.Module):
    debug: bool = True

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
    def load_sd(cls, path: str | Path) -> Self:
        """Load Stable-Diffusion."""

        path = Path(path)
        paths = list(path.glob("*.bin"))
        assert len(paths) == 1
        path = paths[0]

        state = torch.load(path, map_location="cpu")
        model = cls()

        changes: list[tuple[str, str]] = [
            # up/down samplers
            (r"(up|down)samplers.0", r"\1sampler"),
            # post_process
            (
                r"(decoder|encoder).conv_norm_out.(bias|weight)",
                r"\1.post_process.0.\2",
            ),
            (
                r"(decoder|encoder).conv_out.(bias|weight)",
                r"\1.post_process.2.\2",
            ),
            # resnet-blocks pre/post-process
            (
                r"resnets.(\d).norm1.(bias|weight)",
                r"resnets.\1.pre_process.0.\2",
            ),
            (
                r"resnets.(\d).conv1.(bias|weight)",
                r"resnets.\1.pre_process.2.\2",
            ),
            (
                r"resnets.(\d).norm2.(bias|weight)",
                r"resnets.\1.post_process.0.\2",
            ),
            (
                r"resnets.(\d).conv2.(bias|weight)",
                r"resnets.\1.post_process.2.\2",
            ),
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
        if cls.debug:
            old_keys = list(state.keys())
            new_keys = list(model.state_dict().keys())

            in_old = set(old_keys) - set(new_keys)
            in_new = set(new_keys) - set(old_keys)

            with open("in-old.txt", "w") as f:
                f.write("\n".join(sorted(list(in_old))))

            with open("in-new.txt", "w") as f:
                f.write("\n".join(sorted(list(in_new))))

        model.load_state_dict(state)

        return model
