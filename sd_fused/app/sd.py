from __future__ import annotations
from typing import Optional
from typing_extensions import Self

import datetime
from pathlib import Path
import json
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import Tensor

from ..clip import ClipEmbedding
from ..models import AutoencoderKL, UNet2DConditional
from ..utils.cuda import clear_cuda
from ..utils.image import tensor2image
from .setup import Setup
from .helpers import Helpers
from .builder import Builder
from .properties import Properties
from .containers import StepInfo
from .containers import (
    cat_latents,
    stack_scales,
    cat_context,
    stack_weights,
    stack_timesteps,
    stack_indices,
    cat_noises,
)


class StableDiffusion(Setup, Helpers, Builder, Properties):
    version: str = "0.7.1-alpha"

    def __init__(
        self,
        path: str | Path,
        *,
        save_dir: str | Path = "./gallery",
        model_name: Optional[str] = None,
    ) -> None:
        """Load Stable-Diffusion from a diffusers checkpoint."""

        super().__init__()

        self.path = path = Path(path)
        self.save_dir = Path(save_dir)
        self.model_name = model_name or path.name

        assert path.is_dir()

        self.clip = ClipEmbedding(path / "tokenizer", path / "text_encoder")

        clear_cuda()
        self.vae = AutoencoderKL.from_diffusers(path / "vae")
        self.unet = UNet2DConditional.from_diffusers(path / "unet")

        self.builder()

    def get_batch(self) -> list[StepInfo]:
        batch: list[StepInfo] = []
        for _ in range(self._batch_size):
            if len(self._step_info) == 0:
                break

            batch.append(self._step_info.pop(-1))

        return batch

    @property
    def parameters(self) -> dict[str, int | dict[str, int | float | str]]:
        return dict(
            height=self._height,
            width=self._width,
            scheduler=self._scheduler.parameters,
        )

    @torch.no_grad()
    def generate(self) -> list[Image.Image]:
        self.build()

        assert len(self._step_info) != 0

        imgs: list[Image.Image] = []
        with tqdm(total=len(self._step_info) * self._steps) as pbar:
            while len(self._step_info) > 0:
                clear_cuda()

                batch = self.get_batch()

                latents = cat_latents(batch)
                scales = stack_scales(batch)
                context = cat_context(batch)
                weights = stack_weights(batch)
                timesteps = stack_timesteps(batch, self._scheduler)
                indices = stack_indices(batch)
                noises = cat_noises(batch)

                # if batch[0].current_step < 2:
                #     _, _, height, width = latents.shape
                #     pad = 1
                #     # latents = F.interpolate(latents, size=(height + pad, width + pad),
                #     # mode="bilinear", align_corners=True)
                #     shape = (1, *latents.shape[1:])
                #     noises = cat_noises(batch, shape=shape)

                pred_noise = self.predict_noise(latents, timesteps, scales, context, weights)
                latents = self._scheduler.step(indices, latents, pred_noise, noises)

                # update Info and add to stack again.
                # TODO add to it's own function
                for i, info in enumerate(batch):
                    info.clone_latents(latents[[i]])
                    info.current_step += 1

                    # finished
                    if info.current_step == self._steps:
                        print(info)

                        out = self.decode(info.latents)
                        img = tensor2image(out)
                        imgs.append(img)

                        self.save_dir.mkdir(exist_ok=True, parents=True)

                        now = datetime.datetime.now()
                        timestamp = now.strftime("%Y-%m-%d %H-%M-%S.%f")[:-3]
                        filename = f"{timestamp}_{info.seed}_{info.hash}"

                        img.save(self.save_dir / f"{filename}.png", bitmap_format="png")
                        with open(self.save_dir / f"{filename}.txt", "w", encoding="UTF-8") as f:
                            json.dump({**self.parameters, **info.parameters}, f, indent=2)

                    else:
                        self._step_info.append(info)

                pbar.update()
        clear_cuda()

        return imgs

    def predict_noise(
        self,
        latents: Tensor,
        timesteps: Tensor,
        scales: Tensor,
        context: Tensor,
        weights: Tensor,
    ):
        if self.use_low_ram:
            # prompt and negative prompt context
            pc, nc = context.chunk(2, dim=0)
            pw, nw = weights.chunk(2, dim=0)

            # predicted noise for prompt and negative prompt
            ppn = self.unet(latents, timesteps=timesteps, context=pc, weights=pw)
            npn = self.unet(latents, timesteps=timesteps, context=nc, weights=nw)
        else:
            latents = torch.cat([latents] * 2, dim=0)

            pn = self.unet(latents, timesteps=timesteps, context=context, weights=weights)
            ppn, npn = pn.chunk(2, dim=0)

        # add fake channel/spatial dimensions
        scales = scales[:, None, None, None]

        latents = npn + (ppn - npn) * scales

        return latents
