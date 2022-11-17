from __future__ import annotations
from typing import Optional

from pathlib import Path
from tqdm.auto import trange, tqdm
from PIL import Image
from IPython.display import display

from copy import deepcopy
import math
import random
import torch
import torch.nn.functional as F
from torch import Tensor

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from ..utils.cuda import clear_cuda
from ..scheduler import DDIMScheduler
from ..utils.image import tensor2images, image2tensor, image_size
from ..utils.image import ImageType, ResizeModes
from ..utils.typing import MaybeIterable
from ..utils.diverse import to_list, product_args, separate
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
        sub_seed: Optional[int] = None,  # TODO Iterable?
        seed_interpolation: Optional[MaybeIterable[float]] = None,
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
            seed_interpolation=seed_interpolation,
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
            Parameters(**kwargs, mode=mode, seed=seed, sub_seed=sub_seed, device=self.device, dtype=self.dtype)
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
        noise = self.generate_noise(pL.seeds, pL.sub_seeds, pL.seeds_interpolation, pL.height, pL.width, len(pL))

        # TODO move out from here
        scheduler = DDIMScheduler(pL.steps, self.device, self.dtype, pL.seeds)
        scheduler.set_skip_step(pL.strength)

        latents, masked_latents = self.prepare_latents(
            scheduler, noise, pL.images_data, pL.masks_data, pL.masked_images_data
        )
        latents = self.denoise_latents(
            scheduler, latents, masked_latents, context, weight, pL.unconditional, pL.scales, pL.etas
        )

        data = self.decode(latents)
        images = tensor2images(data)

        paths: list[Path] = []
        for parameter, image in zip(pL, images):
            path = self.save_image(image, parameter.png_info)
            paths.append(path)

        return list(zip(images, paths, list(pL)))

    def prepare_latents(
        self,
        scheduler: DDIMScheduler,
        noise: Tensor,
        images: Optional[Tensor],
        masks: Optional[Tensor],
        masked_images: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Prepare initial latents for generation."""

        # text2img
        if images is None:
            assert masks is None
            assert masked_images is None

            return noise, None

        # img2img
        if masks is None:
            assert masked_images is None

            images_latents = self.encode(images)
            latents = scheduler.add_noise(images_latents, noise, scheduler.skip_step)

            return latents, None

        # inpainting
        raise NotImplementedError  # TODO
        # https://github.com/huggingface/diffusers/blob/32b0736d8ad7ec124affca3a00a266f5addcbd91/src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion_inpaint.py#L371
        assert self.is_true_inpainting

        B, C, H, W = images.shape
        mask_latents = F.interpolate(masks.float(), size=(H // 8, W // 8))

        assert masked_images is not None
        masked_image_latents = self.encode(masked_images)

        mask_and_masked_image_latents = torch.cat([mask_latents, masked_image_latents], dim=1)

        return noise, mask_and_masked_image_latents

    def denoise_latents(
        self,
        scheduler: DDIMScheduler,
        latents: Tensor,
        masked_latents: Optional[Tensor],
        context: Tensor,
        weight: Optional[Tensor],
        unconditional: bool,
        scales: Optional[Tensor],
        etas: Optional[Tensor],
    ) -> Tensor:
        """Main loop where latents are denoised."""

        clear_cuda()
        for i in trange(scheduler.skip_step, len(scheduler), desc="Denoising latents"):
            timestep = scheduler.timesteps[[i]]  # ndim=1

            input_latents = latents
            if masked_latents is not None:
                assert scheduler.skip_step == 0
                input_latents = torch.cat([latents, masked_latents], dim=1)

            pred_noise = self.pred_noise(input_latents, timestep, context, weight, unconditional, scales)
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
        scales: Optional[Tensor],
    ) -> Tensor:
        """Predict the noise from latents, context and current timestep."""

        if unconditional:
            assert scales is None
            return self.unet(latents, timestep, context, context_weights)

        assert scales is not None

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

        scales = scales[:, None, None, None]  # add fake channel/spatial dimensions

        latents = pred_noise_negative + (pred_noise_prompt - pred_noise_negative) * scales

        return latents

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return f'{name}(model="{self.model_name}", version="{self.version}")'
