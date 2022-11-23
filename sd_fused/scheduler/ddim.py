from __future__ import annotations
from typing import Optional

import math

import torch
from torch import Tensor

from ..layers.base.types import Device
from ..utils.tensors import to_tensor
from .scheduler import Scheduler

TRAINED_STEPS = 1_000
BETA_BEGIN = 0.00085
BETA_END = 0.012
POWER = 2


class DDIMScheduler(Scheduler):
    """Denoising Diffusion Implicit Models scheduler."""

    # https://arxiv.org/abs/2010.02502

    ᾱ: Tensor
    ϖ: Tensor
    σ: Tensor

    @property  # TODO Type
    def info(self) -> dict[str, str | int]:
        return dict(
            name=self.__class__.__qualname__,
            steps=self.steps,
            skip_timestep=self.skip_timestep,
        )

    def __init__(
        self,
        steps: int,
        shape: tuple[int, ...],
        seeds: list[int],
        strength: Optional[float] = None,  # img2img/inpainting
        device: Optional[Device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(steps, shape, seeds, strength, device, dtype)

        assert steps <= TRAINED_STEPS

        # scheduler betas and alphas
        β_begin = math.pow(BETA_BEGIN, 1 / POWER)
        β_end = math.pow(BETA_END, 1 / POWER)
        β = torch.linspace(β_begin, β_end, TRAINED_STEPS).pow(POWER)
        β = β.to(torch.float64)  # extra-precision

        # increase steps by 1 to account last timestep
        steps += 1

        # trimmed timesteps for selection
        timesteps = torch.linspace(TRAINED_STEPS - 1, 0, steps).ceil().long()

        # cummulative ᾱ trimmed
        α = 1 - β
        ᾱ = α.cumprod(dim=0)
        ᾱ /= ᾱ.max()  # makes last-value=1
        ᾱ = ᾱ[timesteps]
        ϖ = 1 - ᾱ
        del α, β  # reminder that is not used anymore

        # standard deviation, eq (16)
        σ = torch.sqrt(ϖ[1:] / ϖ[:-1] * (1 - ᾱ[:-1] / ᾱ[1:]))

        # use device/dtype
        self.timesteps = timesteps.to(device=device)
        self.ᾱ = ᾱ.to(device=device, dtype=dtype)
        self.ϖ = ϖ.to(device=device, dtype=dtype)
        self.σ = σ.to(device=device, dtype=dtype)

    def add_noise(
        self,
        latents: Tensor,
        noise: Tensor,
        index: int,
    ) -> Tensor:
        # eq 4
        return latents * self.ᾱ[index].sqrt() + noise * self.ϖ[index].sqrt()

    def step(
        self,
        pred_noise: Tensor,
        latents: Tensor,
        index: int,
        etas: Optional[float | Tensor] = None,
    ) -> Tensor:
        etas = to_tensor(etas, self.device, self.dtype, add_spatial=True)

        # eq (12) part 1
        pred_latent = latents - self.ϖ[index].sqrt() * pred_noise
        pred_latent /= self.ᾱ[index].sqrt()

        # eq (12) part 2
        temp = 1 - self.ᾱ[index + 1] - self.σ[index].mul(etas).square()
        pred_dir = temp.abs().sqrt() * pred_noise

        # eq (12) part 3
        noise = self.noise[index] * self.σ[index] * etas

        # full eq (12)
        latents = pred_latent * self.ᾱ[index + 1].sqrt() + pred_dir + noise

        return latents

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return f"{name}(steps={self.steps})"
