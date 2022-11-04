from __future__ import annotations

from torch import Tensor


def softmax(x: Tensor, *, dim: int) -> Tensor:
    """In-place softmax implementation."""

    dtype = x.dtype

    # TODO test only half!
    # TODO use in-place?
    # x = x.float()
    x = x.softmax(dim)
    # x = x.softmax(dim, dtype=torch.float32) # ? is this better?
    # x = x.to(dtype)

    return x
