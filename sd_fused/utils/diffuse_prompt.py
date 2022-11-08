from __future__ import annotations

import re

import math
import torch


def diffuse_prompt(
    prompt: str,
    vmax: float = 1,
    size: int = 1,
) -> list[str]:
    """Diffuse attention-weights to a prompt."""

    assert ":" not in prompt

    pattern = re.compile(r"(\w+)")
    words = pattern.split(prompt)
    n = len(words)

    weigths = torch.randn(size, n)
    weigths = weigths.cumsum(0) / math.sqrt(size) / vmax
    weigths += 1 - weigths[[0]]  # start at the same weight
    weigths = weigths.tolist()

    prompts: list[str] = []
    for weight in weigths:
        prompt = "".join(
            [
                f"{w}:{a:.3f}" if pattern.search(w) else w
                for w, a in zip(words, weight)
            ]
        )
        prompts.append(prompt)

    return prompts
