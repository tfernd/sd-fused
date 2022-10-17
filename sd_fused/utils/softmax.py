from __future__ import annotations

# import torch
from torch import Tensor


def softmax(x: Tensor, *, dim: int) -> Tensor:
    """In-place softmax implementation."""

    dtype = x.dtype

    x = x.float()
    x = x.softmax(dim)
    # x = x.softmax(dim, dtype=torch.float32) # ? is this better?
    x = x.to(dtype)

    return x
