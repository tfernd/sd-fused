from __future__ import annotations
from typing import Optional

from abc import ABC, abstractmethod

import math

import torch
from torch import Tensor

from ..layers.base.types import Device
from ..models import UNet2DConditional
from ..utils.tensors import generate_noise


class Scheduler(ABC):
    """Base-class for all schedulers."""

    timesteps: Tensor

    def __init__(
        self,
        steps: int,
        shape: tuple[int, ...],
        seeds: list[int],
        strength: Optional[float] = None,  # img2img/inpainting
        device: Optional[Device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if strength is not None:
            assert 0 < strength <= 1

        self.steps = steps
        self.shape = shape
        self.seeds = seeds
        self.strength = strength
        self.device = device
        self.dtype = dtype

        # TODO add sub-seeds
        self.noise = generate_noise(shape, seeds, device, dtype, steps)

    @property
    def skip_timestep(self) -> int:
        """Text-to-Image generation starting timestep."""

        if self.strength is None:
            return 0

        return math.ceil(self.steps * (1 - self.strength))

    @abstractmethod
    def add_noise(self, latents: Tensor, noise: Tensor, index: int) -> Tensor:
        """Add noise for a timestep."""

    def prepare_latents(
        self,
        image_latents: Optional[Tensor] = None,
        mask_latents: Optional[Tensor] = None,  # ! old-stype inpainting
        masked_image_latents: Optional[Tensor] = None,  # ! new-style inpainting
    ) -> Tensor:
        """Prepare initial latents for generation."""

        noise = self.noise[0]

        if image_latents is None:
            return noise

        if mask_latents is None and masked_image_latents is None:
            return self.add_noise(image_latents, noise, self.skip_timestep)

        # TODO inpainting
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        pred_noise: Tensor,
        latents: Tensor,
        index: int,
        **kwargs,  # ! type
    ) -> Tensor:
        """Get the previous timestep for the latents."""

    @torch.no_grad()
    def pred_noise(
        self,
        unet: UNet2DConditional,
        latents: Tensor,
        timestep: int,
        context: Tensor,
        weights: Optional[Tensor],
        scale: Optional[Tensor],
        unconditional: bool,
        low_ram: bool,
    ) -> Tensor:
        """Noise prediction for a given timestep."""

        if unconditional:
            assert scale is None

            return unet(latents, timestep, context, weights)

        assert scale is not None

        if low_ram:
            negative_context, prompt_context = context.chunk(2, dim=0)
            if weights is not None:
                negative_weight, prompt_weight = weights.chunk(2, dim=0)
            else:
                negative_weight = prompt_weight = None

            pred_noise_prompt = unet(latents, timestep, prompt_context, prompt_weight)
            pred_noise_negative = unet(latents, timestep, negative_context, negative_weight)
        else:
            latents = torch.cat([latents] * 2, dim=0)

            pred_noise_all = unet(latents, timestep, context, weights)
            pred_noise_negative, pred_noise_prompt = pred_noise_all.chunk(2, dim=0)

        scale = scale[:, None, None, None]  # add fake channel/spatial dimensions

        latents = pred_noise_negative + (pred_noise_prompt - pred_noise_negative) * scale

        return latents
