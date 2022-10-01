from __future__ import annotations
from typing import Optional
from typing_extensions import Self

from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
from einops import rearrange

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from ..scheduler import DDIMScheduler
from ..utils import image2tensor, clear_cuda


class StableDiffusion:
    device: torch.device = torch.device("cpu")
    low_ram: bool = False

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

        if not self.low_ram:
            self.vae.cuda()
        self.unet.cuda()

        self.device = torch.device("cuda")

        return self

    def half_weights(self) -> Self:
        self.vae.half_weights()
        self.unet.half_weights()

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
        scheduler = DDIMScheduler()
        scheduler.set_timesteps(steps)
        timesteps: list[int] = scheduler.timesteps.tolist()

        text_emb = self.clip(prompt) if prompt is not None else None
        neg_emb = self.clip(negative_prompt)
        if text_emb is not None:
            assert text_emb.shape == neg_emb.shape

        if seed is not None:
            torch.manual_seed(seed)
        latents = torch.randn(1, 4, height // 8, width // 8)

        clear_cuda()
        for i, timestep in enumerate(tqdm(timesteps, total=len(timesteps))):
            # low-vram
            pred_noise_text = self.unet(
                latents, timestep=timestep, context=text_emb
            )
            pred_noise_neg = self.unet(
                latents, timestep=timestep, context=neg_emb
            )

            diff = pred_noise_text - pred_noise_neg
            noise_pred = pred_noise_neg + diff.mul_(scale)
            del diff, pred_noise_text, pred_noise_neg

            latents = scheduler.step(
                noise_pred=noise_pred,
                timestep=timestep,
                latents=latents,
                eta=eta,
            )
        clear_cuda()

        MAGIC = 0.18215

        if self.low_ram:
            self.vae.decoder.to(self.device, non_blocking=True)
            self.vae.post_quant_conv.to(self.device, non_blocking=True)

        out = self.vae.decode(latents.div_(MAGIC)).cpu()

        if self.low_ram:
            self.vae.decoder.to("cpu", non_blocking=True)
            self.vae.post_quant_conv.to("cpu", non_blocking=True)

        clear_cuda()

        return Image.fromarray(rearrange(out, "1 C H W -> H W C").numpy())
