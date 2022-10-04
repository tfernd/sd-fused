from __future__ import annotations
from typing import Optional
from typing_extensions import Self

from pathlib import Path
from tqdm.auto import tqdm

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

MAGIC = 0.18215

SIZES = Literal[
    256,
    320,
    384,
    448,
    512,
    576,
    640,
    704,
    768,
    832,
    896,
    960,
    1024,
    1088,
    1152,
    1216,
    1280,
]


class StableDiffusion:
    version: str = "0.1"
    repo: str = "https://github.com/tfernd/sd"
    model_name: str = "stable-diffusion"

    low_ram: bool = False

    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    def __init__(
        self,
        path: str | Path,
        save_dir: str | Path = "./gallery",
        model_name: str = "stable-diffusion-1.4",
    ) -> None:
        self.path = path = Path(path)
        self.save_dir = Path(save_dir)
        self.model_name = model_name

        self.clip = ClipEmbedding(path / "tokenizer", path / "text_encoder")

        self.vae = AutoencoderKL.load_sd(path / "vae")
        self.unet = UNet2DConditional.load_sd(path / "unet")

    def set_low_ram(self) -> Self:
        self.low_ram = True

        return self

    def cuda(self) -> Self:
        clear_cuda()

        self.device = torch.device("cuda")

        self.unet.cuda()
        self.vae.cuda()

        return self

    def half_weights(self, use: bool = True) -> Self:
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
    def set_inplace(self, inplace: bool = True) -> Self:
        self.vae.set_inplace(inplace)
        self.unet.set_inplace(inplace)

        return self

    def split_attention(
        self, *, cross_attention_chunks: Optional[int] = None
    ) -> Self:
        self.unet.split_attention(cross_attention_chunks)

        return self

    def flash_attention(self, flash: bool = True) -> Self:
        self.unet.flash_attention(flash)

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
        height: SIZES = 512,
        width: SIZES = 512,
        seed: Optional[int | list[int]] = None,
        batch_size: int = 1,
    ) -> list[Image.Image]:
        # allowed sizes are multiple of 64
        height -= height % 64
        width -= width % 64

        # scheduler
        scheduler = DDIMScheduler()
        timesteps = scheduler.set_timesteps(steps)

        # fix batch-size if seed-list is provided
        if isinstance(seed, list):
            seed = list(set(seed))  # remove duplicates
            batch_size = len(seed)

        # text embeddings
        prompt, negative_prompt, context = self.get_context_embedding(
            prompt, negative_prompt, batch_size
        )

        # seed and noise
        shape = (batch_size, 4, height // 8, width // 8)
        noise, seeds = generate_noise(shape, seed, self.device, self.dtype)

        latents = noise

        # generation
        clear_cuda()
        for i, timestep in enumerate(tqdm(timesteps, total=len(timesteps))):
            noise_pred = self.pred_noise(latents, timestep, context, scale)
            latents = scheduler.step(noise_pred, timestep, latents, eta=eta)
            del noise_pred
        clear_cuda()

        # decode latent space
        out = self.vae.decode(latents.div_(MAGIC)).cpu()
        clear_cuda()

        # create image
        out = rearrange(out, "B C H W -> B H W C").numpy()
        imgs = [Image.fromarray(v) for v in out]

        self.save_dir.mkdir(parents=True, exist_ok=True)
        for seed, img in zip(seeds, imgs):
            ID = random.randint(0, 2 ** 64)
            path = self.save_dir / f"{ID:x}.png"

            _metadata = dict(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                eta=eta,
                steps=steps,
                scale=scale,
                height=height,
                width=width,
                version=self.version,
                repo=self.repo,
                model=self.model_name,
            )
            metadata = PngInfo()
            for key, value in _metadata.items():
                if isinstance(value, (int, float)):
                    value = str(value)
                elif value is None:
                    value = "null"
                metadata.add_text(key, value)

            img.save(path, pnginfo=metadata)

        return imgs

    @torch.no_grad()
    def pred_noise(
        self,
        latents: Tensor,
        timestep: int,
        context: Tensor | tuple[Tensor, Tensor],
        scale: float,
    ) -> Tensor:
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

        diff = pred_noise_text.sub_(pred_noise_neg)

        return diff.mul_(scale).add_(pred_noise_neg)

    @torch.no_grad()
    def get_context_embedding(
        self, prompt: Optional[str], negative_prompt: str, batch_size: int
    ) -> tuple[Optional[str], str, Tensor | tuple[Tensor, Tensor]]:
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

        return prompt, negative_prompt, context

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return f"{name}()"
