from __future__ import annotations
from typing import Optional

from pathlib import Path
from tqdm.auto import trange, tqdm
from PIL import Image
from IPython.display import display

import math
import random
import torch
import torch.nn.functional as F
from torch import Tensor

from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from ..clip.parser import prompts_choices
from ..scheduler import DDIMScheduler
from ..utils.cuda import clear_cuda
from ..utils.diverse import to_list, product_args, separate
from ..utils.image import ResizeModes, tensor2images, image2tensor, ImageType, image_size
from ..utils.tensors import random_seeds
from ..utils.typing import MaybeIterable
from ..utils.parameters import (
    Parameters,
    ParametersList,
    group_parameters,
    batch_parameters,
)
from .setup import Setup
from .helpers import Helpers


class StableDiffusion(Setup, Helpers):
    version: str = "0.5.5.3"

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
        self.scheduler_renorm(False)
        self.set_low_ram(False)
        self.split_attention(None)
        self.flash_attention(False)
        self.tome(None)
        self.cpu()
        self.float()

    def generate(
        self,
        *,
        eta: MaybeIterable[float] = 0,
        steps: MaybeIterable[int] = 32,
        scale: MaybeIterable[float] = 7.5,
        # scale: Optional[MaybeIterable[float]] = 7.5,  # ?Optional
        height: MaybeIterable[int] = 512,
        width: MaybeIterable[int] = 512,
        negative_prompt: MaybeIterable[str] = "",
        # optionals
        prompt: Optional[MaybeIterable[str]] = None,
        img: Optional[MaybeIterable[ImageType]] = None,
        mask: Optional[MaybeIterable[ImageType]] = None,
        strength: Optional[MaybeIterable[float]] = None,
        mode: Optional[ResizeModes] = None,
        seed: Optional[MaybeIterable[int]] = None,
        sub_seed: Optional[int] = None,  # TODO Iterable?
        # TODO rename to seed_interpolation?
        interpolation: Optional[MaybeIterable[float]] = None,
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
            seed = to_list(seed)

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
            interpolation=interpolation,
            repeat=repeat,
        )

        # generate seeds
        size = len(list_kwargs)
        if seed is None:
            seeds = random_seeds(repeat) if share_seed else random_seeds(size)
        else:
            # ! This part and the bottom one seems to be doing the same thing...
            num_seeds = len(seed)

            # duplicate parameters for each seed
            list_kwargs = [kwargs for kwargs in list_kwargs for _ in range(num_seeds)]
            # duplicat seeds for each parameter
            seeds = [s for _ in range(size) for s in seed]

        # create parameters list and group/batch them
        parameters = [
            Parameters(
                **kwargs,  # type: ignore
                mode=mode,
                seed=seed,
                sub_seed=sub_seed,
                device=self.device,
                dtype=self.dtype,
            )
            for (seed, kwargs) in zip(seeds, list_kwargs)
        ]
        groups = group_parameters(parameters)
        batched_parameters = batch_parameters(groups, batch_size)

        out: list[tuple[Image.Image, Path, Parameters]] = []
        for params in tqdm(batched_parameters):
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

        context, context_weight = self.get_context(pL.negative_prompts, pL.prompts)
        noise = self.generate_noise(pL.seeds, pL.sub_seeds, pL.interpolations, pL.height, pL.width, len(pL))

        # TODO move out from here
        scheduler = DDIMScheduler(pL.steps, self.device, self.dtype, pL.seeds, self.renorm)
        scheduler.set_skip_step(pL.strength)

        latents, masked_latents = self.prepare_latents(scheduler, noise, pL.images_data, pL.masks_data, pL.masked_images_data)
        latents = self.denoise_latents(
            scheduler, latents, masked_latents, context, context_weight, pL.unconditional, pL.scales, pL.etas
        )

        data = self.decode(latents)
        images = tensor2images(data)

        paths: list[Path] = []
        for parameter, image in zip(pL, images):
            path = self.save_image(image, parameter.png_info)
            paths.append(path)

        return list(zip(images, paths, list(pL)))

    def enhance(
        self,
        *,
        img: ImageType,
        factor: int | float,
        strength: float,
        overlap: int = 64,
        prompt: Optional[str] = None,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        eta: float = 0,
        steps: int = 32,
        scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        batch_size: int = 1,
        show: bool = True,
    ):
        """Upscale an image."""

        # upscaled final size
        H, W = image_size(img)
        H, W = round(H * factor / 8) * 8, round(W * factor / 8) * 8

        # image data and overlap weights
        data = image2tensor(img, H, W, mode="resize")
        weight = torch.zeros(1, self.latent_channels, H, W).to(self.device, self.dtype)

        # number of crops along the two dimensions
        nH = math.ceil(H / (height - overlap / 2))
        nW = math.ceil(W / (width - overlap / 2))

        # start indices for the crops
        pi = torch.linspace(0, H - height - 1, nH).round().long()
        pj = torch.linspace(0, W - width - 1, nW).round().long()

        # create latents of crops
        latents_list: list[Tensor] = []
        for n in trange(nH * nW, desc="Encoding latents"):
            i, j = pi[n // nW], pj[n % nW]

            d = data[..., i : i + height, j : j + width]
            weight[..., i : i + height, j : j + width] += 1

            latents_list.append(self.encode(d))
        image_latents = torch.cat(latents_list, dim=0)
        weight_latents = F.interpolate(weight, (H // 8, W // 8))

        # Standard
        unconditional = prompt is None
        prompts = [prompt] if prompt is not None else None
        context, context_weight = self.get_context([negative_prompt], prompts)

        seeds = random_seeds(1) * nH * nW
        scheduler = DDIMScheduler(steps, self.device, self.dtype, seeds, self.renorm)
        scheduler.set_skip_step(strength)

        sub_seeds = interpolations = None
        noise = self.generate_noise(seeds, sub_seeds, interpolations, height, width, nH * nW)
        latents = scheduler.add_noise(image_latents, noise, scheduler.skip_step)

        scales = torch.tensor([scale], device=self.device)

        clear_cuda()
        for i in trange(scheduler.skip_step, len(scheduler), desc="Denoising latents."):
            timestep = scheduler.timesteps[[i]]  # ndim=1

            for k in range(0, nH * nW, batch_size):
                s = slice(k, k + batch_size)
                pred_noise = self.pred_noise(latents[s], timestep, context, context_weight, unconditional, scales)
                latents = scheduler.step(pred_noise, latents, i, eta)

                del pred_noise

            # stitch latents
            big_latents = torch.zeros_like(weight_latents)
            for n in range(nH * nW):
                i, j = pi[n // nW] // 8, pj[n % nW] // 8

                big_latents[..., i : i + height // 8, j : j + width // 8] += latents[[n]]
            big_latents /= weight_latents

            # crop latents
            for n in range(nH * nW):
                i, j = pi[n // nW] // 8, pj[n % nW] // 8

                latents[[n]] = big_latents[..., i : i + height // 8, j : j + width // 8]
        clear_cuda()

        data_list: list[Tensor] = []
        for n in trange(nH * nW, desc="Decoding latents"):
            data_list.append(self.decode(latents[[n]].cuda()))

        data = torch.zeros_like(data).float().cuda()
        for n in range(nH * nW):
            i, j = pi[n // nW], pj[n % nW]

            data[..., i : i + height, j : j + width] += data_list[n]
        data /= weight[:, :3]
        data = data.clamp(0, 255).byte()

        image = tensor2images(data)[0]
        path = self.save_image(image)

        return image

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
        context_weight: Optional[Tensor],
        unconditional: bool,
        scales: Tensor,
        etas: Tensor,
    ) -> Tensor:
        """Main loop where latents are denoised."""

        clear_cuda()
        for i in trange(scheduler.skip_step, len(scheduler), desc="Denoising latents."):
            timestep = scheduler.timesteps[[i]]  # ndim=1

            input_latents = latents
            if masked_latents is not None:
                assert scheduler.skip_step == 0
                input_latents = torch.cat([latents, masked_latents], dim=1)

            pred_noise = self.pred_noise(input_latents, timestep, context, context_weight, unconditional, scales)
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

        scales = scales[:, None, None, None]  # add fake channel/spatial dimensions

        latents = pred_noise_negative + (pred_noise_prompt - pred_noise_negative) * scales

        return latents

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return f'{name}(model="{self.model_name}", version="{self.version}")'
