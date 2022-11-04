from __future__ import annotations
from typing import Iterable, Optional

from pathlib import Path
from tqdm.auto import trange, tqdm
from PIL import Image
from IPython.display import display

import random
import torch
import torch.nn.functional as F
from torch import Tensor

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from ..scheduler import DDIMScheduler
from ..utils import ResizeModes, clear_cuda, generate_noise
from .utils import to_list, product_args
from .parameters import (
    Parameters,
    ParametersList,
    group_parameters,
    batch_parameters,
)
from .setup import Setup
from .helpers import Helpers


class StableDiffusion(Setup, Helpers):
    version: str = "0.5.1"

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
        self.flash_attention(False)
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
        # TODO add SHARE-seed, so parameters combinations share the same seed.

        list_kwargs = product_args(
            eta=eta,
            steps=steps,
            scale=scale,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            prompt=prompt,
            img=img,
            mask=mask,
            strength=strength,
            repeat=repeat,
        )

        # generate seeds # TODO separate
        size = len(list_kwargs)
        if seed is None:
            seeds = [random.randint(0, 2**32 - 1) for _ in range(size)]
        else:
            num_seeds = len(seed)

            list_kwargs = [
                kwargs for kwargs in list_kwargs for _ in range(num_seeds)
            ]
            seeds = [s for _ in range(size) for s in seed]

        # create parameters list and group/batch them
        parameters = [
            Parameters(
                **kwargs,
                mode=mode,
                seed=seed,
                device=self.device,
                dtype=self.dtype,
            )
            for (seed, kwargs) in zip(seeds, list_kwargs)
        ]
        groups = group_parameters(parameters)
        batched_parameters = batch_parameters(groups, batch_size)
        # optimal_batch_size = sum(len(g) for g in groups) // len(groups)

        out: list[tuple[Image.Image, Path, Parameters]] = []
        for params in tqdm(batched_parameters):
            ipp = self.generate_from_parameters(ParametersList(params))
            out.extend(ipp)

            if show:
                for image, path, parameters in ipp:
                    display(image, parameters)

        return out

    @torch.no_grad()
    def generate_from_parameters(
        self,
        pL: ParametersList,
    ) -> list[tuple[Image.Image, Path, Parameters]]:

        context, weight = self.get_context(pL.negative_prompts, pL.prompts)

        scheduler = DDIMScheduler(pL.steps, self.device, self.dtype, pL.seeds)
        skip_step = scheduler.skip_step(pL.strength)

        height, width = pL.size
        shape = (len(pL), self.latent_channels, height // 8, width // 8)
        noise = generate_noise(shape, pL.seeds, self.device, self.dtype)

        # TODO make into its own function!
        masks: Optional[Tensor] = None
        masked_images_latents: Optional[Tensor] = None
        if pL.images_data is not None:

            if pL.masks_data is not None:
                # TODO for now only the real deal
                assert self.is_true_inpainting
                assert pL.masked_images_data is not None
                raise NotImplemented("Needs some rework")  # TODO

                masks = F.interpolate(
                    pL.masks_data.float(),
                    size=(height // 8, width // 8),
                )

                masked_images_latents = self.encode(pL.masked_images_data)
                latents = scheduler.add_noise(
                    masked_images_latents, noise, skip_step
                )
                # latents = noise
            else:
                images_latents = self.encode(pL.images_data, self.dtype)
                latents = scheduler.add_noise(images_latents, noise, skip_step)
        else:
            latents = noise

        latents = self.denoise_latents(
            scheduler,
            latents,
            context,
            weight,
            pL.scales,
            pL.etas,
            pL.unconditional,
            skip_step,
            masks,
            masked_images_latents,
        )

        data = self.decode(latents)
        images = self.create_images(data)

        paths: list[Path] = []
        for parameter, image in zip(pL, images):
            path = self.save_image(image, parameter.png_info)
            paths.append(path)

        return list(zip(images, paths, list(pL)))

    def denoise_latents(
        self,
        scheduler: DDIMScheduler,
        latents: Tensor,
        context: Tensor,
        context_weight: Optional[Tensor],
        scales: Tensor,
        etas: Tensor,
        unconditional: bool,
        skip_step: int,
        masks: Optional[Tensor],
        masked_images_latents: Optional[Tensor],
    ) -> Tensor:
        """Main loop where latents are denoised."""

        clear_cuda()
        for i in trange(skip_step, len(scheduler), desc="Denoising latents."):
            timestep = scheduler.timesteps[[i]]  # ndim=1

            if masks is not None and masked_images_latents is not None:
                input_latents = torch.cat(
                    [latents, masks, masked_images_latents],
                    dim=1,
                )
            else:
                input_latents = latents

            pred_noise = self.pred_noise(
                input_latents,
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

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return f"{name}(model={self.model_name}, version={self.version})"
