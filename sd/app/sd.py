from __future__ import annotations
from typing import Optional
from typing_extensions import Self

from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import random
from einops import rearrange
import torch
from torch import Tensor

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from ..scheduler import DDIMScheduler
from ..utils import image2tensor, clear_cuda

MAGIC = 0.18215


class StableDiffusion:
    low_ram: bool = False

    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    def __init__(self, path: str | Path) -> None:
        path = Path(path)

        self.clip = ClipEmbedding(path / "tokenizer", path / "text_encoder")

        self.vae = AutoencoderKL.load_sd(path / "vae")
        self.unet = UNet2DConditional.load_sd(path / "unet")

        self.split_attention = self.unet.split_attention

    def set_low_ram(self) -> Self:
        self.low_ram = True

        return self

    def cuda(self) -> Self:
        clear_cuda()

        self.device = torch.device("cuda")

        self.unet.cuda()
        self.vae.cuda()

        return self

    def half_weights(self) -> Self:
        self.unet.half_weights()
        self.vae.half_weights()

        return self

    def half(self) -> Self:
        self.unet.half()
        self.vae.half()

        self.dtype = torch.float16

        return self

    def generate(
        self,
        *,
        prompt: Optional[str] = None,
        negative_prompt: str = "",
        eta: float = 0,
        steps: int = 32,
        scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> Image.Image:
        # scheduler
        scheduler = DDIMScheduler()
        scheduler.set_timesteps(steps)
        timesteps: list[int] = scheduler.timesteps.tolist()

        # prompt
        neg_emb = self.clip(negative_prompt).to(self.device).to(self.dtype)
        if prompt is not None:
            text_emb = self.clip(prompt).to(self.device).to(self.dtype)
            context = (text_emb, neg_emb)
        else:
            context = neg_emb

        # seed and noise
        seed = seed or random.randint(0, 2 ** 16)
        torch.manual_seed(seed)
        latents = torch.randn(1, 4, height // 8, width // 8)
        latents = latents.to(self.device).to(self.dtype)

        # generation
        clear_cuda()
        for i, timestep in enumerate(tqdm(timesteps, total=len(timesteps))):
            noise_pred = self.pred_noise(latents, timestep, context, scale)
            latents = scheduler.step(noise_pred, timestep, latents, eta=eta)
        clear_cuda()

        # decode latent space
        out = self.vae.decode(latents.div_(MAGIC)).cpu()
        clear_cuda()

        # create image
        img = Image.fromarray(rearrange(out, "1 C H W -> H W C").numpy())

        return img

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
