from __future__ import annotations
from typing import Optional

from torch import Tensor

from einops import rearrange


def weighted_values(
    v: Tensor,  # (B, heads, T, C)
    weights: Optional[Tensor] = None,  # (B, T)
) -> Tensor:
    "Modify the values to give emphasis to some tokens."

    assert v.ndim == 4

    if weights is None:
        return v

    assert weights.ndim == 2
    weights = rearrange(weights, "B T -> B 1 T 1")

    return v * weights
