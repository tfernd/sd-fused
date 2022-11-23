from __future__ import annotations
from typing import Optional

import random

import torch
from torch import Tensor

from ...layers.base.types import Device


def generate_noise(
    shape: tuple[int, ...],
    seeds: list[int],
    device: Optional[Device] = None,
    dtype: Optional[torch.dtype] = None,
    repeat: int = 1,  # TODO: support repeat
) -> Tensor:
    """Generate random noise with individual seeds per batch."""

    batch_size, *rest = shape
    assert len(seeds) == batch_size

    extended_shape = (repeat, batch_size, *rest)

    noise = torch.empty(extended_shape)
    for n in range(repeat):
        for i, s in enumerate(seeds):
            generator = torch.Generator()
            generator.manual_seed(s + n)

            noise[n, i] = torch.randn(*rest, generator=generator)

    noise = noise.to(device=device, dtype=dtype)

    if repeat == 1:
        return noise[0]

    return noise


def random_seeds(size: int) -> list[int]:
    """Generate random seeds."""

    return [random.randint(0, 2**32 - 1) for _ in range(size)]
