from __future__ import annotations
from typing import Optional

from torch import Tensor

from einops import repeat


def weighted_values(x: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    """Modify the `values` by a given token-`weight`."""

    if weights is None:
        return x

    assert weights.ndim == 2
    assert weights.shape[1] == x.shape[1]

    # qkv has heads combined into the batch, but weights don't
    num_heads = x.shape[0] // weights.shape[0]
    weights = repeat(weights, "b t -> (b h) t 1", h=num_heads)

    return x * weights
