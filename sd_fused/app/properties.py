from __future__ import annotations

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding


class Properties:
    clip: ClipEmbedding

    vae: AutoencoderKL
    unet: UNet2DConditional

    @property
    def latent_size(self) -> int:
        return self.unet.in_channels
