from __future__ import annotations
from typing import Optional

from pathlib import Path
from datetime import datetime

from PIL import Image

import random
from einops import rearrange
import torch
from torch import Tensor

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from .parameters import Parameters

MAGIC = 0.18215


class Helpers:
    version: str
    model_name: str

    save_dir: Path

    clip: ClipEmbedding
    vae: AutoencoderKL
    unet: UNet2DConditional

    device: torch.device
    dtype: torch.dtype

    @property
    def latent_channels(self) -> int:
        """Latent-space channel size."""

        return self.unet.in_channels

    def save_image(
        self,
        image: Image.Image,
        parameters: Parameters,
        ID: Optional[int] = None,
    ) -> Path:
        """Save the image using the provided metadata information."""

        now = datetime.now()
        timestamp = now.strftime(r"%Y-%m-%d %H-%M-%S.%f")

        if ID is None:
            ID = random.randint(0, 2**64)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        path = self.save_dir / f"{timestamp} - {ID:x}.SD.png"
        image.save(path, bitmap_format="png", pnginfo=parameters.png_info)

        return path

    def create_images(self, data: Tensor) -> list[Image.Image]:
        """Creates a list of images according to the batch size."""

        data = rearrange(data, "B C H W -> B H W C").cpu().numpy()

        return [Image.fromarray(v) for v in data]

    @torch.no_grad()
    def encode(self, data: Tensor) -> Tensor:
        """Encodes (stochastically) a RGB image into a latent vector."""

        return self.vae.encode(data).sample().mul(MAGIC)

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """Decode latent vector into an RGB image."""

        return self.vae.decode(latents.div(MAGIC))

    @torch.no_grad()
    def get_context(
        self,
        negative_prompts: list[str],
        prompts: Optional[list[str]],
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Creates a context Tensor (negative + positive prompt) and a emphasis weights."""

        texts = negative_prompts
        if prompts is not None:
            texts.extend(prompts)

        context, weight = self.clip(texts, self.device, self.dtype)

        return context, weight
