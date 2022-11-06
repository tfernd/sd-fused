from __future__ import annotations
from typing import Optional

from pathlib import Path
from tqdm.auto import trange, tqdm
from PIL import Image
from IPython.display import display

import math
import random
import torch
from torch import Tensor


from ..models import AutoencoderKL, UNet2DConditional
from ..clip import ClipEmbedding
from ..clip.parser import prompts_choices
from ..scheduler import DDIMScheduler
from ..utils.cuda import clear_cuda
from ..utils.diverse import to_list, product_args
from ..utils.image import ResizeModes, tensor2images, image2tensor
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
    version: str = "0.5.3"

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
        self.scheduler_renorm(False)
        self.set_low_ram(False)
        self.split_attention(None)
        self.flash_attention(False)
        self.tome(None)
        self.cpu()
        self.float()

    # TODO this needs to be divided into separated functions
    def generate(
        self,
        *,
        eta: MaybeIterable[float] = 0,
        steps: MaybeIterable[int] = 32,
        scale: MaybeIterable[float] = 7.5,  # ?Optional
        height: MaybeIterable[int] = 512,
        width: MaybeIterable[int] = 512,
        negative_prompt: MaybeIterable[str] = "",
        # optional
        prompt: Optional[MaybeIterable[str]] = None,
        # TODO add support for PIL.Image
        img: Optional[MaybeIterable[str | Image.Image]] = None,
        mask: Optional[MaybeIterable[str | Image.Image]] = None,
        strength: Optional[MaybeIterable[float]] = None,
        mode: Optional[ResizeModes] = None,
        seed: Optional[MaybeIterable[int]] = None,
        sub_seed: Optional[int] = None,  # TODO Iterable?
        # TODO seed_interpolation?
        interpolation: Optional[MaybeIterable[float]] = None,
        # latents: Optional[Tensor] = None, # TODO
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
        # optimal_batch_size = sum(len(g) for g in groups) // len(groups)

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

        context, weight = self.get_context(pL)
        noise = self.generate_noise(pL)
        scheduler = DDIMScheduler(
            pL.steps, self.device, self.dtype, pL.seeds, renorm=self.renorm
        )
        scheduler.set_skip_step(pL.strength)

        # TODO make into its own function!
        masks: Optional[Tensor] = None
        masked_images_latents: Optional[Tensor] = None
        if pL.images_data is not None:

            if pL.masks_data is not None:
                # TODO for now only the real deal
                # assert self.is_true_inpainting
                # assert pL.masked_images_data is not None
                raise NotImplemented("Needs some rework")  # TODO

                # masks = F.interpolate(
                #     pL.masks_data.float(),
                #     size=(height // 8, width // 8),
                # )

                # masked_images_latents = self.encode(pL.masked_images_data)
                # latents = scheduler.add_noise(
                #     masked_images_latents, noise, skip_step
                # )
                # # latents = noise
            else:
                images_latents = self.encode(pL.images_data, self.dtype)
                latents = scheduler.add_noise(
                    images_latents, noise, scheduler.skip_step
                )
        else:
            latents = noise

        latents = self.denoise_latents(
            scheduler,
            latents,
            context,
            weight,
            pL,
            masks,
            masked_images_latents,
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
        img: str,
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
        assert factor > 1
        assert 0 < overlap < min(height, width)

        data = image2tensor(img)
        B, C, H, W = data.shape

        H = round(factor * H)
        W = round(factor * W)
        data = image2tensor(img, H, W, mode="resize")
        B, C, H, W = data.shape

        nH = math.ceil(H / (height - overlap / 2))
        nW = math.ceil(W / (width - overlap / 2))

        pi = torch.linspace(0, H - height - 1, nH).round().long()
        pj = torch.linspace(0, W - width - 1, nW).round().long()

        # TODO avoid for-loop
        datas: list[Tensor] = []
        for i in pi:
            for j in pj:
                d = data[..., i : i + height, j : j + width]
                datas.append(d)
        data = torch.cat(datas, dim=0)

        imgs = tensor2images(data)

        out = self.generate(
            eta=eta,
            steps=steps,
            scale=scale,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            prompt=prompt,
            img=imgs,
            strength=strength,
            mode="resize",
            seed=seed,
            # sub_seed/interpolation
            batch_size=batch_size,
            show=show,
        )

        # TODO stich images

    def denoise_latents(
        self,
        scheduler: DDIMScheduler,
        latents: Tensor,
        context: Tensor,
        context_weight: Optional[Tensor],
        p: ParametersList,
        masks: Optional[Tensor],
        masked_images_latents: Optional[Tensor],
    ) -> Tensor:
        """Main loop where latents are denoised."""

        clear_cuda()
        for i in trange(
            scheduler.skip_step, len(scheduler), desc="Denoising latents."
        ):
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
                p.unconditional,
                p.scales,
            )
            latents = scheduler.step(pred_noise, latents, i, p.etas)

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
