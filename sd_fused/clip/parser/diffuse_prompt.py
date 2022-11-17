from __future__ import annotations

import re

import math
from typing import Optional
import torch


def diffuse_prompt(
    prompt: str,
    vmax: float = 1,
    size: int = 1,
    seed: Optional[int] = None,
) -> list[str]:
    """Diffuse attention-weights to a prompt."""

    assert ":" not in prompt

    pattern = re.compile(r"(\w+)")
    words = pattern.split(prompt)
    n = len(words)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    weigths = torch.randn(size, n, generator=generator)
    weigths = weigths.cumsum(0) / math.sqrt(size) * vmax
    weigths += 1 - weigths[[0]]  # start at the same weight
    weigths = weigths.tolist()

    prompts: list[str] = []
    for weight in weigths:
        prompt = "".join([f"{w}:{a:.3f}" if pattern.search(w) else w for w, a in zip(words, weight)])
        prompts.append(prompt)

    return prompts
