from __future__ import annotations
from typing import Iterable, Optional

from pathlib import Path
from tqdm.auto import trange, tqdm
from itertools import product
from PIL import Image

import random
import torch
from torch import Tensor

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from ..scheduler import DDIMScheduler
from ..utils import ResizeModes, clear_cuda, generate_noise, async_display
from .utils import to_list
from .parameters import Parameters, ParametersList

from .setup import Setup
from .helpers import Helpers


class StableDiffusion(Setup, Helpers):
    version: str = "0.5.0"

    def __init__(
        self,
        path: str | Path,
        *,
        save_dir: str | Path = "./gallery",
        model_name: Optional[str] = None,
    ) -> None:
        """Load Stable-Diffusion diffusers checkpoint."""

        self.path = path = Path(path)
        self.save_dir = Path(save_dir)
        self.model_name = model_name or path.name

        assert path.is_dir()

        self.clip = ClipEmbedding(path / "tokenizer", path / "text_encoder")

        self.vae = AutoencoderKL.from_diffusers(path / "vae")
        self.unet = UNet2DConditional.from_diffusers(path / "unet")

        # init
        self.set_low_ram(False)
        self.split_attention(None)
        self.cpu()
        self.float()

    # TODO this needs to be divided into separated functions
    def generate(
        self,
        *,
        eta: float | Iterable[float] = 0,
        steps: int | Iterable[int] = 32,
        scale: float | Iterable[float] = 7.5,
        height: int | Iterable[int] = 512,
        width: int | Iterable[int] = 512,
        negative_prompt: str | Iterable[str] = "",
        # optional
        prompt: Optional[str | Iterable[str]] = None,
        img: Optional[str | Iterable[str]] = None,
        mask: Optional[str | Iterable[str]] = None,
        strength: Optional[float | Iterable[float]] = None,
        seed: Optional[int | Iterable[int]] = None,
        mode: Optional[ResizeModes] = None,
        batch_size: int = 1,
        repeat: int = 1,
        show: bool = True,
    ) -> list[tuple[Image.Image, Path, Parameters]]:
        """Create a list of parameters and group them
        into batches to be processed.
        """

        if seed is not None:
            repeat = 1
            seed = to_list(seed)

        # listify args
        list_eta = to_list(eta)
        list_steps = to_list(steps)
        list_scale = to_list(scale)
        list_height = to_list(height)
        list_width = to_list(width)
        list_negative_prompt = to_list(negative_prompt)
        list_prompt = to_list(prompt)
        list_img = to_list(img)
        list_mask = to_list(mask)
        list_strength = to_list(strength)

        # all possible combinations of all parameters
        # fmt: off
        keys = [
            'eta', 'steps', 'scale',
            'height', 'width',
            'negative_prompt', 'prompt',
            'img', 'mask','strength'
        ]
        perms = list(product(
            list_eta, list_steps, list_scale, 
            list_height, list_width,
            list_negative_prompt, list_prompt, 
            list_img, list_mask, list_strength,
        ))
        list_kwargs = [
            dict(zip(keys, args)) 
            for args in perms 
            for _ in range(repeat)
        ]
        # fmt: on

        # generate seeds
        size = len(list_kwargs)
        if seed is None:
            seeds = [random.randint(0, 2**32 - 1) for _ in range(size)]
        else:
            num_seeds = len(seed)

            list_kwargs = [
                kwargs for kwargs in list_kwargs for _ in range(num_seeds)
            ]
            seeds = [s for _ in range(size) for s in seed]

        # create parameters list and group them
        parameters = [
            Parameters(**kwargs, mode=mode, seed=seed, device=self.device)
            for (seed, kwargs) in zip(seeds, list_kwargs)
        ]

        groups: list[list[Parameters]] = []
        for parameter in parameters:
            if len(groups) == 0:
                groups.append([parameter])
                continue

            can_share = False
            for group in groups:
                can_share = True
                for other_parameter in group:
                    can_share &= parameter.can_share_batch(other_parameter)

                if can_share:
                    group.append(parameter)
                    break

            if not can_share:
                groups.append([parameter])

        # optimal_batch_size = sum(len(g) for g in groups) // len(groups)

        # create batches from groups
        batched_parameters: list[list[Parameters]] = []
        for group in groups:
            for i in range(0, len(group), batch_size):
                s = slice(i, min(i + batch_size, len(group)))

                batched_parameters.append(group[s])

        out: list[tuple[Image.Image, Path, Parameters]] = []
        for params in batched_parameters:
            ipp = self.generate_from_parameters(ParametersList(params))
            out.extend(ipp)

            if show:
                for image, path, parameters in ipp:
                    async_display(image, parameters)

        return out

    @torch.no_grad()
    def generate_from_parameters(
        self,
        p: ParametersList,
    ) -> list[tuple[Image.Image, Path, Parameters]]:
        scheduler = DDIMScheduler(p.steps, self.device, self.dtype, p.seeds)
        context, weight = self.get_context(p.negative_prompts, p.prompts)

        height, width = p.size
        shape = (len(p), self.latent_channels, height // 8, width // 8)
        latents = generate_noise(shape, p.seeds, self.device, self.dtype)

        # TODO make into its own function!
        if p.images_data is not None:
            assert p.strength is not None

            images_latents = self.encode(p.images_data.to(self.device))

            skip_step = scheduler.skip_step(p.strength)
            latents = scheduler.add_noise(images_latents, latents, skip_step)
        else:
            skip_step = 0

        # TODO add mask handling

        latents = self.denoise_latents(
            scheduler,
            latents,
            context,
            weight,
            p.scales,
            p.etas,
            p.unconditional,
            skip_step,
        )

        data = self.decode(latents)
        images = self.create_images(data)

        paths: list[Path] = []
        for parameter, image in zip(p, images):
            path = self.save_image(image, parameter)
            paths.append(path)

        return list(zip(images, paths, list(p)))

    def denoise_latents(
        self,
        scheduler: DDIMScheduler,
        latents: Tensor,
        context: Tensor,
        context_weight: Optional[Tensor],
        scales: Tensor,
        etas: Tensor,
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
                scales,
            )
            latents = scheduler.step(pred_noise, latents, i, etas)
            del pred_noise
        clear_cuda()

        return latents

    @torch.no_grad()
    def pred_noise(
        self,
        latents: Tensor,
        timestep: Tensor,
        context: Tensor,
        context_weights: Optional[Tensor],
        unconditional: bool,
        scales: Tensor,
    ) -> Tensor:
        """Predict the noise from latents, context and current timestep."""
        # fmt: off

        if unconditional:
            return self.unet(latents, timestep, context, context_weights)

        if self.low_ram:
            negative_context, prompt_context = context.chunk(2, dim=0)
            if context_weights is not None:
                negative_weight, prompt_weight = context_weights.chunk(2, dim=0)
            else:
                negative_weight = prompt_weight = None

            pred_noise_prompt = self.unet(latents, timestep, prompt_context, prompt_weight)
            pred_noise_negative = self.unet(latents, timestep, negative_context, negative_weight)
        else:
            latents = torch.cat([latents] * 2, dim=0)

            pred_noise_all = self.unet(latents, timestep, context, context_weights)
            pred_noise_negative, pred_noise_prompt = pred_noise_all.chunk(2, dim=0)
            del pred_noise_all
        
        scales = scales[:, None, None, None] # add fake channel/spatial dimensions

        return pred_noise_negative + (pred_noise_prompt - pred_noise_negative) * scales
        # fmt: on
