from __future__ import annotations
from typing import Optional

import math

import torch
from torch import Tensor


class DDIMScheduler:
    # https://arxiv.org/abs/2010.02502
    def __init__(
        self,
        *,
        steps: int,
        offset: int = 0,  # ? Not needed?
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # seeds: list[int], # TODO pre-generate noises based on seeds
        # batch_size: int = 1, # TODO Above!
        # ? DO NOT CHANGE!? Make it GLOBAL constant?
        trained_steps: int = 1_000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        power: float = 2,
    ) -> None:
        assert steps <= trained_steps
        self.steps = steps
        self.device = device
        self.dtype = dtype

        # scaled-linear scheduler
        beta_start = math.pow(beta_start, 1 / power)
        beta_end = math.pow(beta_end, 1 / power)
        β = torch.linspace(beta_start, beta_end, trained_steps).pow(power)

        α = 1 - β
        ᾱ = α.cumprod(dim=0)

        # trimmed timesteps for selection
        chunk = trained_steps / steps
        timesteps = torch.arange(offset, offset + chunk * steps, chunk)
        timesteps = timesteps.flip(0).long()

        # mask variables
        β, α, ᾱ = β[timesteps], α[timesteps], ᾱ[timesteps]

        # add final timestep
        ᾱ = torch.cat([ᾱ, torch.ones(1)])
        ϖ = 1 - ᾱ

        # standard deviation eq (16)
        σ = torch.sqrt((1 - ᾱ[1:]) / (1 - ᾱ[:-1]) * (1 - ᾱ[:-1] / ᾱ[1:]))

        # use device
        self.ᾱ = ᾱ.to(device=device, dtype=dtype)
        self.ϖ = ϖ.to(device=device, dtype=dtype)
        self.σ = σ.to(device=device, dtype=dtype)
        self.timesteps = timesteps.to(device=device)

    def step(
        self, pred_noise: Tensor, latent: Tensor, i: int, eta: float = 0,
    ) -> Tensor:
        """Get the previous latents according to the DDIM paper."""

        assert 0 <= i < self.steps
        # TODO add support for i as Tensor

        # eq (12) part 1
        pred_latent = latent - self.ϖ[i].sqrt() * pred_noise
        pred_latent /= self.ᾱ[i].sqrt()

        # eq (12) part 2
        temp = 1 - self.ᾱ[i + 1] - self.σ[i].mul(eta).square()
        pred_direction = torch.sqrt(temp) * pred_noise

        # eq (12) part 3
        # TODO add seeds
        noise = torch.randn_like(latent) * self.σ[i] * eta

        # full eq (12)
        prev_latent = pred_latent * self.ᾱ[i + 1].sqrt() + pred_direction
        prev_latent += noise

        return prev_latent

    def add_noise(self, latents: Tensor, eps: Tensor, i: int) -> Tensor:
        """Add noise to latents according to the index i."""

        assert 0 <= i < self.steps
        # TODO add support for i as Tensor

        # eq 4
        return latents * self.ᾱ[i].sqrt() + eps * self.ϖ[i].sqrt()

    def cutoff_index(self, strength: float) -> int:
        """For a given strength [0, 1) what is the cutoff index?"""

        assert 0 < strength <= 1

        return round(len(self) * (1 - strength))

    def __len__(self) -> int:
        return self.steps
