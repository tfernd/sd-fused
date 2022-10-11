from __future__ import annotations
from typing import Any, Optional
from typing_extensions import Self

from pathlib import Path
from tqdm.auto import trange

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
    version: str = "0.2.0"
    repo: str = "https://github.com/tfernd/sd"

    low_ram: bool = False

    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

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

        self.clip = ClipEmbedding(path / "tokenizer", path / "text_encoder")

        self.vae = AutoencoderKL.load_sd(path / "vae")
        self.unet = UNet2DConditional.load_sd(path / "unet")

    def unet_scale(self, path: str | Path, scale: float = 1) -> Self:
        """Scales the UNet to use in-between two different stable-diffusion models."""

        path = Path(path)
        new_unet = UNet2DConditional.load_sd(path / "unet")

        for param, new_param in zip(
            self.unet.parameters(), new_unet.parameters()
        ):
            pnew = new_param.data.to(device=self.device, dtype=self.dtype)

            if scale == 1:
                param.data = pnew
            else:
                param.data += pnew.sub_(param.data).mul_(scale)
            del pnew

        return self

    def set_low_ram(self, low_ram: bool = True) -> Self:
        """Split context into two passes to save memory."""

        self.low_ram = low_ram

        return self

    def cuda(self) -> Self:
        clear_cuda()

        self.device = torch.device("cuda")

        self.unet.cuda()
        self.vae.cuda()

        return self

    def half_weights(self, use: bool = True) -> Self:
        """Store the weights in half-precision but
    compute forward pass in full precision.
    Useful for GPUs that gives NaN when used in half-precision.
    """

        self.unet.half_weights(use)
        self.vae.half_weights(use)

        return self

    def half(self) -> Self:
        self.unet.half()
        self.vae.half()

        self.dtype = torch.float16

        return self

    def float(self) -> Self:
        self.unet.float()
        self.vae.float()

        self.dtype = torch.float32

        return self

    def split_attention(
        self, *, cross_attention_chunks: Optional[int] = None
    ) -> Self:
        """Split cross-attention computation into chunks."""

        self.unet.split_attention(cross_attention_chunks)

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
    ) -> list[Image.Image]:
        kwargs = kwargs2ignore(locals(), keys=["batch_size", "seed"])

        # allowed sizes are multiple of 64
        assert height % 64 == 0
        assert width % 64 == 0

        scheduler = DDIMScheduler(
            steps=steps, device=self.device, dtype=self.dtype
        )
        batch_size = fix_batch_size(seed, batch_size)
        context = self.get_context(prompt, negative_prompt, batch_size)

        shape = (batch_size, 4, height // 8, width // 8)
        latents, seeds = generate_noise(shape, seed, self.device, self.dtype)

        latents = self.denoise_latents(scheduler, latents, context, scale, eta)

        # decode latent space
        out = self.vae.decode(latents.div(MAGIC)).cpu()

        # TODO its own function!
        # create images
        out = rearrange(out, "B C H W -> B H W C").numpy()
        images = [Image.fromarray(v) for v in out]

        self.save_dir.mkdir(parents=True, exist_ok=True)
        for seed, image in zip(seeds, images):
            metadata = self._create_metadata(seed=seed, **kwargs)

            # TODO temporary file name for now
            ID = random.randint(0, 2 ** 64)
            path = self.save_dir / f"{ID:x}.png"

            image.save(path, bitmap_format="png", pnginfo=metadata)

        return images

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
    ) -> list[Image.Image]:
        kwargs = kwargs2ignore(locals(), keys=["batch_size", "seed"])

        # allowed sizes are multiple of 64
        assert height % 64 == 0
        assert width % 64 == 0

        scheduler = DDIMScheduler(
            steps=steps, device=self.device, dtype=self.dtype
        )
        batch_size = fix_batch_size(seed, batch_size)
        context = self.get_context(prompt, negative_prompt, batch_size)

        shape = (batch_size, 4, height // 8, width // 8)
        noise, seeds = generate_noise(shape, seed, self.device, self.dtype)

        # TODO make into its own function
        # latents from image
        assert mode == "resize"
        data = image2tensor(img, size=(height, width), device=self.device)
        img_latents = self.vae.encode(data).mean
        img_latents *= MAGIC

        k = scheduler.cutoff_index(strength)
        latents = scheduler.add_noise(img_latents, noise, k)

        latents = self.denoise_latents(
            scheduler, latents, context, scale, eta, k=k
        )

        # decode latent space
        out = self.vae.decode(latents.div(MAGIC)).cpu()
        clear_cuda()

        # TODO remove code duplication
        # create images
        out = rearrange(out, "B C H W -> B H W C").numpy()
        images = [Image.fromarray(v) for v in out]

        self.save_dir.mkdir(parents=True, exist_ok=True)
        for seed, image in zip(seeds, images):
            metadata = self._create_metadata(seed=seed, **kwargs)

            # TODO temporary file name for now
            ID = random.randint(0, 2 ** 64)
            path = self.save_dir / f"{ID:x}.png"

            image.save(path, bitmap_format="png", pnginfo=metadata)
            # TODO add copy of img to image

        return images

    @torch.no_grad()
    def pred_noise(
        self,
        latents: Tensor,
        timestep: int,
        context: Tensor | tuple[Tensor, Tensor],
        scale: float,
    ) -> Tensor:
        """Predict the noise from latents, context and current timestep."""

        # unconditional
        if isinstance(context, Tensor):
            return self.unet(latents, timestep=timestep, context=context)

        if self.low_ram:
            text_emb, neg_emb = context

            pred_noise_text = self.unet(latents, timestep, context=text_emb)
            pred_noise_neg = self.unet(latents, timestep, context=neg_emb)
        else:
            context = torch.cat(context, dim=0)
            latents = torch.cat([latents] * 2, dim=0)

            pred_noise_tn = self.unet(latents, timestep, context=context)
            pred_noise_text, pred_noise_neg = pred_noise_tn.chunk(2, dim=0)
            del pred_noise_tn

        return pred_noise_neg + (pred_noise_text - pred_noise_neg) * scale

    @torch.no_grad()
    def get_context(
        self, prompt: Optional[str], negative_prompt: str, batch_size: int
    ) -> Tensor | tuple[Tensor, Tensor]:
        negative_prompt = self.clip.parse_text(negative_prompt)
        if prompt is not None:
            prompt = self.clip.parse_text(prompt)

        neg_emb = self.clip(negative_prompt, self.device, self.dtype)
        neg_emb = neg_emb.expand(batch_size, -1, -1)

        if prompt is not None:
            text_emb = self.clip(prompt, self.device, self.dtype)
            text_emb = text_emb.expand(batch_size, -1, -1)

            context = (text_emb, neg_emb)
        else:
            context = neg_emb

        return context

    def denoise_latents(
        self,
        scheduler: DDIMScheduler,
        latents: Tensor,
        context: Tensor | tuple[Tensor, Tensor],
        scale: float,
        eta: float,
        k: int = 0,
    ) -> Tensor:
        """Main loop where latents are denoised."""

        # TODO if eta != 0, pass the seeds to the scheduler, so things can be reproducible.
        assert eta == 0

        clear_cuda()
        for i in trange(k + 1, len(scheduler), desc="Denoising latents."):
            timestep = int(scheduler.timesteps[i].item())
            noise_pred = self.pred_noise(latents, timestep, context, scale)
            latents = scheduler.step(noise_pred, latents, i, eta)
            del noise_pred
        clear_cuda()

        return latents

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return f"{name}()"

    def _create_metadata(self, **kwargs: Any) -> PngInfo:
        kwargs = dict(
            **kwargs,
            version=self.version,
            repo=self.repo,
            model=self.model_name,
        )

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
