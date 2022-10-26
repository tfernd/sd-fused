from __future__ import annotations
from typing import Any, Optional

from pathlib import Path
from tqdm.auto import trange
from datetime import datetime

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import random
from einops import rearrange
import torch
from torch import Tensor

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from ..scheduler import DDIMScheduler
from ..utils import (
    image2tensor,
    ResizeModes,
    image_base64,
    clear_cuda,
    generate_noise,
)
from .utils import fix_batch_size, kwargs2ignore
from .modifiers import Modifiers

MAGIC = 0.18215


class StableDiffusion(Modifiers):
    version: str = "0.4.3"

    clip: ClipEmbedding

    def __init__(
        self,
        path: str | Path,
        *,
        save_dir: str | Path = "./gallery",
        model_name: Optional[str] = None,
    ) -> None:
        self.path = path = Path(path)
        self.save_dir = Path(save_dir)
        self.model_name = model_name or path.name

        assert path.is_dir()

        self.clip = ClipEmbedding(path / "tokenizer", path / "text_encoder")

        self.vae = AutoencoderKL.from_diffusers(path / "vae")
        self.unet = UNet2DConditional.from_diffusers(path / "unet")

        # init
        self.set_low_ram(False)
        self.cpu()
        self.float()

    @property
    def latent_channels(self) -> int:
        """Latent-space channel size."""

        return self.unet.in_channels

    @torch.no_grad()
    def _generate(
        self,
        *,
        eta: float,
        steps: int,
        scale: float,
        height: int,
        width: int,
        negative_prompt: str,
        batch_size: int,
        prompt: Optional[str] = None,
        img: Optional[str] = None,
        strength: Optional[float] = None,
        seed: Optional[int | list[int]] = None,
        mode: Optional[ResizeModes] = None,
    ) -> list[tuple[Image.Image, Path]]:
        """General purpose image generation."""

        kwargs = kwargs2ignore(locals(), keys=["batch_size", "seed", "img"])

        # allowed sizes are multiple of 64
        assert height % 64 == 0
        assert width % 64 == 0

        unconditional = prompt is None
        seed, batch_size = fix_batch_size(seed, batch_size)

        context, weight = self.get_context(prompt, negative_prompt, batch_size)

        shape = (batch_size, self.latent_channels, height // 8, width // 8)
        latents, seeds = generate_noise(shape, seed, self.device, self.dtype)

        scheduler = DDIMScheduler(steps, self.device, self.dtype, seeds)

        if img is not None:
            assert strength is not None
            assert mode is not None

            img_data = image2tensor(img, height, width, mode, self.device)
            img_latents = self.encode(img_data)

            skip_step = scheduler.cutoff_index(strength)
            latents = scheduler.add_noise(img_latents, latents, skip_step)

            kwargs["img_base64"] = image_base64(img)
        else:
            skip_step = 0

        latents = self.denoise_latents(
            scheduler,
            latents,
            context,
            weight,
            scale,
            eta,
            unconditional,
            skip_step,
        )

        data = self.decode(latents)
        images = self.create_images(data)

        paths: list[Path] = []
        for seed, image in zip(seeds, images):
            metadata = self._create_metadata(seed=seed, **kwargs)
            path = self.save_image(image, metadata)
            paths.append(path)

        return list(zip(images, paths))

    def text2img(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        steps: int = 32,
        scale: float = 7.5,
        eta: float = 0,
        height: int = 512,
        width: int = 512,
        seed: Optional[int | list[int]] = None,
        batch_size: int = 1,
    ):
        """Creates an image from a prompt and (optionally) a negative prompt."""

        return self._generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            scale=scale,
            eta=eta,
            height=height,
            width=width,
            seed=seed,
            batch_size=batch_size,
        )

    def img2img(
        self,
        prompt: str,
        *,
        img: str,
        strength: float,
        negative_prompt: str = "",
        steps: int = 32,
        scale: float = 7.5,
        eta: float = 0,
        height: int = 512,
        width: int = 512,
        seed: Optional[int | list[int]] = None,
        batch_size: int = 1,
        mode: ResizeModes = "resize",
    ) -> list[tuple[Image.Image, Path]]:
        """Creates an image from a prompt and (optionally) a negative prompt
        with an image as a basis.
        """

        return self._generate(
            prompt=prompt,
            img=img,
            strength=strength,
            negative_prompt=negative_prompt,
            steps=steps,
            scale=scale,
            eta=eta,
            height=height,
            width=width,
            seed=seed,
            batch_size=batch_size,
            mode=mode,
        )

    def save_image(
        self, image: Image.Image, metadata: PngInfo, ID: Optional[int] = None
    ) -> Path:
        """Save the image using the provided metadata information."""

        now = datetime.now()
        timestamp = now.strftime(r"%Y-%m-%d %H-%M-%S.%f")

        if ID is None:
            ID = random.randint(0, 2 ** 64)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        path = self.save_dir / f"{timestamp} - {ID:x}.SD.png"
        image.save(path, bitmap_format="png", pnginfo=metadata)

        return path

    @torch.no_grad()
    def get_context(
        self, prompt: Optional[str], negative_prompt: str, batch_size: int
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Creates a context Tensor (negative + positive prompt) and a emphasis weights."""

        texts = [negative_prompt] * batch_size
        if prompt is not None:
            texts.extend([prompt] * batch_size)

        context, weight = self.clip(texts, self.device, self.dtype)

        return context, weight

    def denoise_latents(
        self,
        scheduler: DDIMScheduler,
        latents: Tensor,
        context: Tensor,
        context_weight: Optional[Tensor],
        scale: float,
        eta: float,
        unconditional: bool,
        skip_step: int = 0,
    ) -> Tensor:
        """Main loop where latents are denoised."""

        clear_cuda()
        for i in trange(skip_step, len(scheduler), desc="Denoising latents."):
            timestep = scheduler.timesteps[[i]]  # ndim=1

            pred_noise = self.pred_noise(
                latents,
                timestep,
                context,
                context_weight,
                unconditional,
                scale,
            )
            latents = scheduler.step(pred_noise, latents, i, eta)
            del pred_noise
        clear_cuda()

        return latents

    @torch.no_grad()
    def pred_noise(
        self,
        latents: Tensor,
        timestep: int | Tensor,
        context: Tensor,
        context_weights: Optional[Tensor],
        unconditional: bool,
        scale: float,
    ) -> Tensor:
        """Predict the noise from latents, context and current timestep."""

        if unconditional:
            return self.unet(latents, timestep, context, context_weights)

        if self.low_ram:
            negative_context, prompt_context = context.chunk(2, dim=0)
            if context_weights is not None:
                negative_weight, prompt_weight = context_weights.chunk(
                    2, dim=0
                )
            else:
                negative_weight = prompt_weight = None

            pred_noise_prompt = self.unet(
                latents, timestep, prompt_context, prompt_weight,
            )
            pred_noise_negative = self.unet(
                latents, timestep, negative_context, negative_weight,
            )
        else:
            latents = torch.cat([latents] * 2, dim=0)

            pred_noise_all = self.unet(
                latents, timestep, context, context_weights
            )
            pred_noise_negative, pred_noise_prompt = pred_noise_all.chunk(
                2, dim=0
            )
            del pred_noise_all

        return (
            pred_noise_negative
            + (pred_noise_prompt - pred_noise_negative) * scale
        )

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """Decode latent vector into an RGB image."""

        return self.vae.decode(latents.div(MAGIC))

    def encode(self, data: Tensor) -> Tensor:
        """Encodes (stochastically) a RGB image into a latent vector."""

        return self.vae.encode(data).sample().mul(MAGIC)

    def create_images(self, data: Tensor) -> list[Image.Image]:
        """Creates a list of images according to the batch size."""

        data = rearrange(data, "B C H W -> B H W C").cpu().numpy()

        return [Image.fromarray(v) for v in data]

    def _create_metadata(self, **kwargs: Any) -> PngInfo:
        """Creates metadata to be used for the PNG."""

        kwargs = dict(**kwargs, version=self.version, model=self.model_name,)

        metadata = PngInfo()
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                value = str(value)
            elif value is None:
                value = "null"
            elif isinstance(value, str):
                pass
            else:
                raise TypeError(
                    f"value ({value}) of type {type(value)} not supported."
                )

            metadata.add_text(f"SD {key}", value)

        return metadata

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return f'{name}(model="{self.model_name}", version={self.version})'


# TODO re-implement this better
#     def unet_scale(self, path: str | Path, scale: float = 1) -> Self:
#         """Scales the UNet to use in-between two different stable-diffusion models."""

#         path = Path(path)
#         new_unet = UNet2DConditional.load_sd(path / "unet")

#         for param, new_param in zip(
#             self.unet.parameters(), new_unet.parameters()
#         ):
#             pnew = new_param.data.to(device=self.device, dtype=self.dtype)

#             # TODO add support for other types of scaling
#             if scale == 1:
#                 param.data = pnew
#             else:
#                 param.data += pnew.sub_(param.data).mul_(scale)
#             del pnew

#         return self
