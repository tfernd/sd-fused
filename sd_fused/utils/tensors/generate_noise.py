from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor


def generate_noise(
    shape: tuple[int, ...],
    seeds: list[int],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Generate random noise with individual seeds per batch."""

    noise = torch.empty(shape)
    for i, s in enumerate(seeds):
        generator = torch.Generator()
        generator.manual_seed(s)

        noise[i] = torch.randn(*shape[1:], generator=generator)

    return noise.to(device=device, dtype=dtype)
