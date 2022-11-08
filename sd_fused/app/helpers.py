from __future__ import annotations
from typing import Optional

from pathlib import Path
from datetime import datetime

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import random
import torch
from torch import Tensor

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from ..utils.tensors import slerp, generate_noise
from ..utils.parameters import ParametersList

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
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Encodes (stochastically) a RGB image into a latent vector."""

        return self.vae.encode(data, dtype).sample().mul(MAGIC)

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """Decode latent vector into an RGB image."""

        return self.vae.decode(latents.div(MAGIC))

    @torch.no_grad()
    def get_context(
        self,
        p: ParametersList,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Creates a context Tensor (negative + positive prompt) and a emphasis weights."""

        texts = p.negative_prompts
        if p.prompts is not None:
            texts.extend(p.prompts)

        context, weight = self.clip(texts, self.device, self.dtype)

        return context, weight

    def generate_noise(self, p: ParametersList) -> Tensor:
        height, width = p.size
        shape = (len(p), self.latent_channels, height // 8, width // 8)

        noise = generate_noise(shape, p.seeds, self.device, self.dtype)
        if p.sub_seeds is not None:
            assert p.interpolations is not None
            sub_noise = generate_noise(
                shape, p.sub_seeds, self.device, self.dtype
            )
            noise = slerp(noise, sub_noise, p.interpolations)

        return noise
