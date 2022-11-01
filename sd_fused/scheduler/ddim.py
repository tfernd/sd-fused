from __future__ import annotations
from typing import Optional

import math

import torch
from torch import Tensor

from einops import rearrange

from ..utils import generate_noise

TRAINED_STEPS = 1_000
BETA_BEGIN = 0.00085
BETA_END = 0.012
POWER = 2


class DDIMScheduler:
    """Denoising Diffusion Implicit Models scheduler."""

    # https://arxiv.org/abs/2010.02502

    timesteps: Tensor
    ᾱ: Tensor
    ϖ: Tensor
    σ: Tensor

    noises: Optional[Tensor] = None

    def __init__(
        self,
        steps: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[list[int]] = None,
    ) -> None:
        assert steps <= TRAINED_STEPS

        self.steps = steps
        self.device = device
        self.dtype = dtype
        self.seed = seed

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

    def step(
        self,
        pred_noise: Tensor,
        latents: Tensor,
        i: int | Tensor,
        eta: float | Tensor = 0,
    ) -> Tensor:
        """Get the previous latents according to the DDIM paper."""
        i = to_tensor(i, steps=self.steps, device=self.device)
        eta = to_tensor(eta, device=self.device, dtype=self.dtype)

        # eq (12) part 1
        pred_latent = latents - self.ϖ[i].sqrt() * pred_noise
        pred_latent /= self.ᾱ[i].sqrt()

        # eq (12) part 2
        temp = 1 - self.ᾱ[i + 1] - self.σ[i].mul(eta).square()
        pred_dir = torch.sqrt(temp) * pred_noise

        # eq (12) part 3
        if self.noises is None:
            # pre-generate noises for all steps # ! ugly... needs some work
            shape = (*latents.shape, self.steps)
            self.noises, _ = generate_noise(
                shape, self.seed, self.device, self.dtype
            )
            self.noises = rearrange(self.noises, "B C H W S -> S B C H W")
        noise = self.noises[i.flatten()].flatten(0, 1)
        noise *= self.σ[i] * eta

        # full eq (12)
        return pred_latent * self.ᾱ[i + 1].sqrt() + pred_dir + noise

    def add_noise(
        self, latents: Tensor, eps: Tensor, i: int | Tensor
    ) -> Tensor:
        """Add noise to latents according to the index i."""

        i = to_tensor(i, steps=self.steps, device=self.device)

        # eq 4
        return latents * self.ᾱ[i].sqrt() + eps * self.ϖ[i].sqrt()

    def skip_step(self, strength: float) -> int:
        """The index generation needs to start."""

        assert 0 < strength <= 1

        return math.ceil(len(self) * (1 - strength))

    def __len__(self) -> int:
        return self.steps

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return f"{name}(steps={self.steps})"


def to_tensor(
    x: int | float | Tensor,
    *,
    steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Convert a number to a Tensor with fake channel/spatial dimensions."""

    if isinstance(x, (int, float)):
        x = torch.tensor([x], device=device, dtype=dtype)
    else:
        assert x.ndim == 1
        x = x.to(device=device, dtype=dtype)

    if steps is not None:
        assert torch.all(0 <= x) and torch.all(x < steps)

    x = x.view(-1, 1, 1, 1)

    return x
