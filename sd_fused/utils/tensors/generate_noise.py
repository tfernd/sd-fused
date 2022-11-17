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
) -> Tensor:
    """Generate random noise with individual seeds per batch."""

    noise = torch.empty(shape)
    for i, s in enumerate(seeds):
        generator = torch.Generator()
        generator.manual_seed(s)

        noise[i] = torch.randn(*shape[1:], generator=generator)

    return noise.to(device=device, dtype=dtype)


def random_seeds(size: int) -> list[int]:
    """Generate random seeds."""

    return [random.randint(0, 2**32 - 1) for _ in range(size)]
