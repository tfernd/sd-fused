from __future__ import annotations
from typing import Optional
from typing_extensions import Self

from dataclasses import dataclass, field

import hashlib

import torch
from torch import Tensor

from ..layers.base import Base


@dataclass
class Prompt:
    prompt: str
    emb: Tensor
    weight: Tensor

    def __repr__(self) -> str:
        return f'{self.__class__.__qualname__}("{self.prompt}")'


@dataclass
class StepInfo(Base):
    prompt: Prompt
    negative_prompt: Prompt
    steps: int
    current_step: int
    scale: float
    seed: int

    latents: Tensor = field(init=False, repr=False)

    def init_latents(self, latent_size: int, height: int, width: int) -> Self:
        self.latents = self.gen_noise(shape=(latent_size, height // 8, width // 8))

        return self

    def gen_noise(
        self,
        shape: Optional[tuple[int, int, int]] = None,
    ) -> Tensor:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.current_step)

        noise = torch.randn(shape or self.latents.shape[1:], generator=generator)

        return noise[None]

    @property
    def hash(self) -> str:
        text = "| ".join(
            [
                self.prompt.prompt,
                self.negative_prompt.prompt,
                str(self.current_step),
                str(self.scale),
            ]
        )

        return hashlib.md5(text.encode()).hexdigest()

    @property
    def parameters(self) -> dict[str, str | int | float]:
        return dict(
            prompt=self.prompt.prompt,
            negative_prompt=self.negative_prompt.prompt,
            scale=self.scale,
            seed=self.seed,
            steps=self.steps,
        )

    def clone_latents(self, latents: Tensor) -> Self:
        self.latents = latents.clone()

        return self

def send(x: Tensor, info: StepInfo) -> Tensor:
    device = info.device
    dtype = info.dtype

    return x.to(device=device, dtype=dtype)


def cat_latents(info_batch: list[StepInfo]) -> Tensor:
    latents = torch.cat([info.latents for info in info_batch])

    return send(latents, info_batch[0])


def cat_context(info_batch: list[StepInfo]) -> Tensor:
    context = torch.cat(
        [info.prompt.emb for info in info_batch] + [info.negative_prompt.emb for info in info_batch]
    )

    return send(context, info_batch[0])


def stack_weights(info_batch: list[StepInfo]) -> Tensor:
    weights = torch.stack(
        [info.prompt.weight for info in info_batch] + [info.negative_prompt.weight for info in info_batch]
    )

    return send(weights, info_batch[0])


def stack_scales(info_batch: list[StepInfo]) -> Tensor:
    scales = torch.tensor([info.scale for info in info_batch])

    return send(scales, info_batch[0])


from ..scheduler import Scheduler


def stack_timesteps(info_batch: list[StepInfo], scheduler: Scheduler) -> Tensor:
    timesteps = torch.stack([scheduler.timesteps[info.current_step] for info in info_batch])

    return send(timesteps, info_batch[0])


def stack_indices(info_batch: list[StepInfo]) -> Tensor:
    indices = torch.tensor([info.current_step for info in info_batch])

    return send(indices, info_batch[0]).long()  # ! hacky


def cat_noises(
    info_batch: list[StepInfo],
    shape: Optional[tuple[int, int, int]] = None,
) -> Tensor:
    noises = torch.cat([info.gen_noise(shape) for info in info_batch])

    return send(noises, info_batch[0])
