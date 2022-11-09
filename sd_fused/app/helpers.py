from __future__ import annotations
from typing import Optional

from pathlib import Path
from datetime import datetime
from copy import deepcopy

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import random
import torch
from torch import Tensor

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from ..utils.tensors import slerp, generate_noise

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

        return self.unet.out_channels

    @property
    def is_true_inpainting(self) -> bool:
        """RunwayMl true inpainting model."""

        return self.unet.in_channels != self.latent_channels

    def save_image(
        self,
        image: Image.Image,
        png_info: Optional[PngInfo] = None,
        ID: Optional[int] = None,
    ) -> Path:
        """Save the image using the provided metadata information."""

        now = datetime.now()
        timestamp = now.strftime(r"%Y-%m-%d %H-%M-%S.%f")

        if ID is None:
            ID = random.randint(0, 2**64)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        path = self.save_dir / f"{timestamp} - {ID:x}.SD.png"
        image.save(path, bitmap_format="png", pnginfo=png_info)

        return path

    @torch.no_grad()
    def encode(
        self,
        data: Tensor,
    ) -> Tensor:
        """Encodes (stochastically) a RGB image into a latent vector."""

        return self.vae.encode(data.to(self.device), self.dtype).sample().mul(MAGIC)

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

        texts = deepcopy(negative_prompts)
        if prompts is not None:
            texts.extend(deepcopy(prompts))

        context, weight = self.clip(texts, self.device, self.dtype)

        return context, weight

    def generate_noise(
        self,
        seeds: list[int],
        sub_seeds: Optional[list[int]],
        interpolations: Optional[Tensor],
        height: int,
        width: int,
        batch_size: int,
    ) -> Tensor:
        """Generate random noise with individual seeds per batch and
        possible sub-seed interpolation."""

        shape = (batch_size, self.latent_channels, height // 8, width // 8)
        noise = generate_noise(shape, seeds, self.device, self.dtype)
        if sub_seeds is None:
            return noise

        assert interpolations is not None
        sub_noise = generate_noise(shape, sub_seeds, self.device, self.dtype)
        noise = slerp(noise, sub_noise, interpolations)

        return noise
