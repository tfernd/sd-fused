from __future__ import annotations

from torch import Tensor


def softmax_(x: Tensor, dim: int, eps: float = 1e-6) -> Tensor:
    """In-place softmax implementation."""

    dtype = x.dtype

    x = x.float()

    x -= x.max(dim, keepdim=True).values
    x = x.exp_()
    # TODO needed eps?
    x /= x.sum(dim, keepdim=True).add_(eps)

    x = x.to(dtype)

    return x
