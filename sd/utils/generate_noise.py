from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor


def generate_noise(
    shape: tuple[int, ...],
    seed: Optional[int | list[int]],
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
) -> tuple[Tensor, list[int]]:
    batch_size = shape[0]

    if seed is None:
        seeds = torch.randint(0, 2 ** 32, (batch_size,)).tolist()
    elif isinstance(seed, int):
        seeds = [seed]
    else:
        seeds = seed

    out: list[Tensor] = []
    for s in seeds:
        generator = torch.Generator()
        generator.manual_seed(s)
        out.append(torch.randn(*shape[1:], generator=generator))

    noise = torch.stack(out, dim=0).to(device=device, dtype=dtype)

    return noise, seeds
