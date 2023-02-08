from __future__ import annotations

import torch
from torch import Tensor

from ..layers.base import Base
from .properties import Properties

MAGIC = 0.18215


class Helpers(Base, Properties):
    @torch.no_grad()
    def encode(self, data: Tensor) -> Tensor:
        """Encodes (stochastically) a RGB image into a latent vector."""

        return self.vae.encode(data).sample().mul(MAGIC)

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """Decode latent vector into an RGB image."""

        return self.vae.decode(latents.div(MAGIC))
