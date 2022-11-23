from __future__ import annotations
from typing import Optional

from pathlib import Path
from tqdm.auto import trange, tqdm
from PIL import Image
from IPython.display import display

from copy import deepcopy
import torch
from torch import Tensor

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from ..utils.cuda import clear_cuda
from ..scheduler import Scheduler, DDIMScheduler
from ..utils.image import tensor2images
from ..utils.image import ImageType, ResizeModes
from ..utils.typing import MaybeIterable
from ..utils.diverse import to_list, product_args
from ..clip.parser import prompts_choices
from ..utils.tensors import random_seeds
from ..utils.parameters import Parameters, ParametersList, group_parameters, batch_parameters
from .setup import Setup
from .helpers import Helpers


class StableDiffusion(Setup, Helpers):
    version: str = "0.6.0"

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

        # initialize
        self.low_ram(False)
        self.split_attention(None)
        self.flash_attention(False)
        self.tome(None)

    def generate(
        self,
        *,
        eta: MaybeIterable[float] = 0,
        steps: MaybeIterable[int] = 32,
        height: MaybeIterable[int] = 512,
        width: MaybeIterable[int] = 512,
        negative_prompt: MaybeIterable[str] = "",
        # optionals
        scale: Optional[MaybeIterable[float]] = 7.5,
        prompt: Optional[MaybeIterable[str]] = None,
        img: Optional[MaybeIterable[ImageType]] = None,
        mask: Optional[MaybeIterable[ImageType]] = None,
        strength: Optional[MaybeIterable[float]] = None,
        mode: Optional[ResizeModes] = None,
        seed: Optional[MaybeIterable[int]] = None,
        # sub_seed: Optional[int] = None,  # TODO Iterable?
        # seed_interpolation: Optional[MaybeIterable[float]] = None,
        # latents: Optional[Tensor] = None, # TODO
        batch_size: int = 1,
        repeat: int = 1,
        show: bool = True,
        share_seed: bool = True,
    ) -> list[tuple[Image.Image, Path, Parameters]]:
        """Create a list of parameters and group them
        into batches to be processed.
        """

        if seed is not None:
            repeat = 1

        if prompt is not None:
            prompt = prompts_choices(prompt)
        negative_prompt = prompts_choices(negative_prompt)

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
            # seed_interpolation=seed_interpolation,
        )
        size = len(list_kwargs)
        list_kwargs = deepcopy(list_kwargs * repeat)

        # if seeds are given or share-seed set
        # each repeated-iteration has the same seed
        if seed is not None or share_seed:
            if seed is None:
                seed = random_seeds(repeat)
            seeds = [s for s in to_list(seed) for _ in range(size)]

        # otherwise each iteration has it's own unique seed
        else:
            seeds = random_seeds(size * repeat)

        # create parameters list and group/batch them
        parameters = [
            Parameters(**kwargs, mode=mode, seed=seed, device=self.device, dtype=self.dtype)  #  sub_seed=sub_seed
            for (seed, kwargs) in zip(seeds, list_kwargs)
        ]
        groups = group_parameters(parameters)
        batched_parameters = batch_parameters(groups, batch_size)

        out: list[tuple[Image.Image, Path, Parameters]] = []
        for params in tqdm(batched_parameters, desc="Generating batches"):
            ipp = self.generate_from_parameters(ParametersList(params))
            out.extend(ipp)

            if show:
                for image, path, parameters in ipp:
                    print(parameters)
                    display(image)

        return out

    @torch.no_grad()
    def generate_from_parameters(
        self,
        pL: ParametersList,
    ) -> list[tuple[Image.Image, Path, Parameters]]:

        context, weight = self.get_context(pL.negative_prompts, pL.prompts)

        # TODO make general
        scheduler = DDIMScheduler(
            pL.steps, pL.shape(self.latent_channels), pL.seeds, pL.strength, self.device, self.dtype
        )

        enc = lambda x: self.encode(x) if x is not None else None
        image_latents = enc(pL.images_data)
        mask_latents = enc(pL.masks_data)
        masked_image_latents = enc(pL.masked_images_data)

        latents = scheduler.prepare_latents(image_latents, mask_latents, masked_image_latents)
        # TODO add to scheduler
        latents = self.denoise_latents(scheduler, latents, context, weight, pL.unconditional, pL.scales, pL.etas)

        data = self.decode(latents)
        images = tensor2images(data)

        paths: list[Path] = []
        for parameter, image in zip(pL, images):
            path = self.save_image(image, parameter.png_info)
            paths.append(path)

        return list(zip(images, paths, list(pL)))

    def denoise_latents(
        self,
        scheduler: Scheduler,
        latents: Tensor,
        context: Tensor,
        weight: Optional[Tensor],
        unconditional: bool,
        scales: Optional[Tensor],
        etas: Optional[Tensor],
    ) -> Tensor:
        """Main loop where latents are denoised."""

        clear_cuda()
        for index in trange(scheduler.skip_timestep, scheduler.steps, desc="Denoising latents"):
            timestep = int(scheduler.timesteps[index].item())

            pred_noise = scheduler.pred_noise(
                self.unet, latents, timestep, context, weight, scales, unconditional, self.use_low_ram
            )
            latents = scheduler.step(pred_noise, latents, index, etas=etas)

            del pred_noise
        clear_cuda()

        return latents

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return f'{name}(model="{self.model_name}", version="{self.version}")'
