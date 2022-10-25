from __future__ import annotations
from typing import Any, Optional
from typing_extensions import Self

from pathlib import Path
from tqdm.autonotebook import trange

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import random
from einops import rearrange
import torch
from torch import Tensor

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from ..scheduler import DDIMScheduler
from ..utils import image2tensor, clear_cuda, generate_noise
from ..utils.typing import Literal
from .utils import fix_batch_size, kwargs2ignore

MAGIC = 0.18215


class StableDiffusion:
    version: str = "0.4.0"

    device: torch.device
    dtype: torch.dtype

    clip: ClipEmbedding
    vae: AutoencoderKL
    unet: UNet2DConditional

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

        self.vae = AutoencoderKL.load_sd(path / "vae")
        self.unet = UNet2DConditional.load_sd(path / "unet")

        # init
        self.set_low_ram(False)
        self.cpu()
        self.float()

    def set_low_ram(self, low_ram: bool = True) -> Self:
        """Split context into two passes to save memory."""

        self.low_ram = low_ram

        return self

    def to(self, device: Literal["cpu", "cuda"] | torch.device) -> Self:
        """Send unet and auto-encoder to device."""

        self.device = device = torch.device(device)

        self.unet.to(device=self.device, non_blocking=True)
        self.vae.to(device=self.device, non_blocking=True)

        return self

    def cuda(self) -> Self:
        """Send unet and auto-encoder to cuda."""

        clear_cuda()

        return self.to("cuda")

    def cpu(self) -> Self:
        """Send unet and auto-encoder to cpu."""

        return self.to("cpu")

    def half(self) -> Self:
        """Use half-precision for unet and auto-encoder."""

        self.unet.half()
        self.vae.half()

        self.dtype = torch.float16

        return self

    def float(self) -> Self:
        """Use full-precision for unet and auto-encoder."""

        self.unet.float()
        self.vae.float()

        self.dtype = torch.float32

        return self

    def half_weights(self, use_half_weights: bool = True) -> Self:
        """Store the weights in half-precision but
    compute forward pass in full precision.
    Useful for GPUs that gives NaN when used in half-precision.
    """

        self.unet.half_weights(use_half_weights)
        self.vae.half_weights(use_half_weights)

        return self

    def split_attention(
        self, *, cross_attention_chunks: Optional[int] = None
    ) -> Self:
        """Split cross-attention computation into chunks."""

        self.unet.split_attention(cross_attention_chunks)
        self.vae.split_attention(cross_attention_chunks)

        return self

    @torch.no_grad()
    def text2img(
        self,
        *,
        prompt: Optional[str] = None,
        negative_prompt: str = "",
        eta: float = 0,
        steps: int = 32,
        scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int | list[int]] = None,
        batch_size: int = 1,
    ) -> list[tuple[Image.Image, Path]]:
        """Creates an image from a prompt and (optionally) a negative prompt."""

        kwargs = kwargs2ignore(locals(), keys=["batch_size", "seed"])

        # allowed sizes are multiple of 64
        assert height % 64 == 0
        assert width % 64 == 0

        unconditional = prompt is None

        scheduler = DDIMScheduler(steps, self.device, self.dtype)

        batch_size = fix_batch_size(seed, batch_size)
        context, weight = self.get_context(prompt, negative_prompt, batch_size)

        shape = (batch_size, 4, height // 8, width // 8)
        latents, seeds = generate_noise(shape, seed, self.device, self.dtype)

        latents = self.denoise_latents(
            scheduler, latents, context, weight, scale, eta, unconditional
        )

        data = self.decode(latents)
        images = self.create_images(data)

        paths: list[Path] = []
        for seed, image in zip(seeds, images):
            metadata = self._create_metadata(seed=seed, **kwargs)
            path = self.save_image(image, metadata)
            paths.append(path)

        return list(zip(images, paths))

    # TODO Merge this with text2image since they share lots of code
    @torch.no_grad()
    def img2img(
        self,
        *,
        img: str,
        prompt: Optional[str] = None,
        negative_prompt: str = "",
        eta: float = 0,
        steps: int = 32,
        scale: float = 7.5,
        strength: float = 0.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int | list[int]] = None,
        batch_size: int = 1,
        mode: Literal["resize", "resize-crop", "resize-pad"] = "resize",
    ) -> list[tuple[Image.Image, Path]]:
        """Creates an image from a prompt and (optionally) a negative prompt
        with an image as a basis.
        """

        kwargs = kwargs2ignore(locals(), keys=["batch_size", "seed"])

        # allowed sizes are multiple of 64
        assert height % 64 == 0
        assert width % 64 == 0

        unconditional = prompt is None

        scheduler = DDIMScheduler(steps, self.device, self.dtype)

        batch_size = fix_batch_size(seed, batch_size)
        context, weight = self.get_context(prompt, negative_prompt, batch_size)

        shape = (batch_size, 4, height // 8, width // 8)
        noise, seeds = generate_noise(shape, seed, self.device, self.dtype)

        # TODO make into its own function
        # latents from image
        assert mode == "resize"
        data = image2tensor(img, size=(height, width), device=self.device)
        img_latents = self.encode(data)

        k = scheduler.cutoff_index(strength)
        latents = scheduler.add_noise(img_latents, noise, k)

        latents = self.denoise_latents(
            scheduler, latents, context, weight, scale, eta, unconditional, k=k
        )

        data = self.decode(latents)
        images = self.create_images(data)

        paths: list[Path] = []
        for seed, image in zip(seeds, images):
            metadata = self._create_metadata(seed=seed, **kwargs)
            # TODO add as metadata the original image as base64-encoded string?
            path = self.save_image(image, metadata)
            paths.append(path)

        return list(zip(images, paths))

    @torch.no_grad()
    def img2img_alt(
        self,
        *,
        img: str,
        prompt: str,
        eta: float = 0,
        steps: int = 32,
        scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int | list[int]] = None,
        batch_size: int = 1,
        mode: Literal["resize", "resize-crop", "resize-pad"] = "resize",
    ) -> list[tuple[Image.Image, Path]]:

        kwargs = kwargs2ignore(locals(), keys=["batch_size", "seed"])

        # allowed sizes are multiple of 64
        assert height % 64 == 0
        assert width % 64 == 0

        scheduler = DDIMScheduler(steps, self.device, self.dtype)

        batch_size = fix_batch_size(seed, batch_size)
        context, weight = self.get_context(prompt, negative_prompt, batch_size)

    def save_image(
        self, image: Image.Image, metadata: PngInfo, ID: Optional[int] = None
    ) -> Path:
        """Save the image using the provided metadata information."""

        if ID is None:
            ID = random.randint(0, 2 ** 64)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        path = self.save_dir / f"{ID:x}.SD.png"
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
        k: int = 0,
    ) -> Tensor:
        """Main loop where latents are denoised."""

        clear_cuda()
        for i in trange(k, len(scheduler), desc="Denoising latents."):
            # TODO for now unet accept only ints and not Tensor
            timestep = int(scheduler.timesteps[i].item())

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
        timestep: int,
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

        return f"{name}({self.model_name})"


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
