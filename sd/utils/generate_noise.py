from __future__ import annotations
from typing import NamedTuple, Optional

import torch
from torch import Tensor

# NOISE INCREASED SIZE
SIZE = 2_048


class NoiseOutput(NamedTuple):
    """Noise tensor and their respective seeds."""

    noise: Tensor
    seed: list[int]


def generate_noise(
    shape: tuple[int, int, int, int],
    seed: Optional[int | list[int]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> NoiseOutput:
    """Generate random noise with individual seeds per batch."""

    batch_size, channels, height, width = shape
    dh, dw = height - SIZE, width - SIZE
    sh = slice(dh // 2, dh // 2 + height)
    sw = slice(dw // 2, dw // 2 + width)

    if seed is None:
        seeds = torch.randint(0, 2 ** 32, (batch_size,)).tolist()
    elif isinstance(seed, int):
        seeds = [seed]
    else:
        seeds = seed

    temp_shape = (batch_size, channels, SIZE, SIZE)
    noise = torch.empty(temp_shape)
    for i, s in enumerate(seeds):
        generator = torch.Generator()
        generator.manual_seed(s)

        noise[i] = torch.randn(*temp_shape[1:], generator=generator)
    noise = noise[..., sh, sw]  # crop

    noise = noise.to(device=device, dtype=dtype)

    return NoiseOutput(noise, seeds)
