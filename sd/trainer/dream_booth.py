from __future__ import annotations
from typing import Optional

from pathlib import Path

import torch
from torch import Tensor
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from tqdm.auto import trange, tqdm

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from ..scheduler import DDIMScheduler
from ..utils import image2tensor, clear_cuda, generate_noise
from ..utils.typing import Literal


class DreamBoothTrainer:
    latents_instance: Tensor
    latents_class: Tensor

    context_instance: Tensor
    context_class: Tensor

    prompt_instance: str
    prompt_class: str

    def __init__(self, path: str | Path, size: int = 512) -> None:
        assert size % 64 == 0

        self.path = path = Path(path)
        self.size = size

        self.logging_dir = None

        # models
        self.clip = ClipEmbedding(path / "tokenizer", path / "text_encoder")

        self.vae = AutoencoderKL.load_sd(path / "vae")
        self.vae.cuda()

        self.unet = UNet2DConditional.load_sd(path / "unet")
        self.unet.set_inplace(False)
        self.unet.flash_attention(True)
        self.unet.half()
        self.unet.cuda()

    def _add_prompt(
        self, prompt: str, *, kind: Literal["instance", "class"]
    ) -> None:
        device = torch.device("cuda")
        comtext = self.clip(prompt, device, torch.float16)

        if kind == "instance":
            self.comtext_instance = comtext
        if kind == "class":
            self.comtext_class = comtext

    def _add_images(
        self,
        imgs: list[str | Path],
        batch_size: int,
        *,
        kind: Literal["instance", "class"]
    ) -> None:
        data_list = [image2tensor(img, resize=self.size) for img in imgs]
        data = torch.cat(data_list, dim=0).half().cuda()

        latents_list: list[Tensor] = []
        for i in trange(0, len(data), batch_size, desc="Encoding latents."):
            posterior = self.vae.encode(data[i : i + batch_size])

            latents_list.append(posterior.mean)
        latents = torch.cat(latents_list, dim=0)

        if kind == "instance":
            self.latents_instance = latents
        if kind == "class":
            self.latents_class = latents

    def add_instance_images(
        self, imgs: list[str | Path], batch_size: int
    ) -> None:
        self._add_images(imgs, batch_size, kind="instance")

    def add_class_images(
        self, imgs: list[str | Path], batch_size: int
    ) -> None:
        self._add_images(imgs, batch_size, kind="class")

    def add_instance_prompt(self, prompt: str) -> None:
        self._add_prompt(prompt, kind="instance")

    def add_class_prompt(self, prompt: str) -> None:
        self._add_prompt(prompt, kind="class")

    def train(self, *,
        lr:float= 5e-5,
        batch_size: int=1,
        use_8bit_adam:bool=True,
        weight_decay: float=0,
        betas: tuple[float,float]=(0,0),
        adam_eps:float=0
    ) -> None:

        # scale learning-rate
        lr *= batch_size
        lr *= self.accelerator.num_processes

        # optimizer 
        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            self.unet.parameters(), 
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=adam_eps,
        )

        scheduler = DDIMScheduler()