from __future__ import annotations

from torch import Tensor


def softmax(x: Tensor, *, dim: int, inplace: bool = True) -> Tensor:
    """In-place softmax implementation."""

    dtype = x.dtype

    x = x.float()

    if inplace:
        x -= x.max(dim, keepdim=True).values
        x = x.exp_()
        x /= x.sum(dim, keepdim=True)
    else:
        x = x.softmax(dim)

    x = x.to(dtype)

    return x
