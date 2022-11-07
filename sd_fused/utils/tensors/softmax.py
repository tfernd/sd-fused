from __future__ import annotations

from torch import Tensor


def softmax(x: Tensor, *, dim: int) -> Tensor:
    """Softmax implementation."""

    x = x.softmax(dim, dtype=x.dtype)

    return x


def softmax_(x: Tensor, *, dim: int) -> Tensor:
    """In-place softmax implementation."""

    x -= x.amax(dim, keepdim=True)
    x = x.exp_()
    x /= x.sum(dim, keepdim=True)

    return x
