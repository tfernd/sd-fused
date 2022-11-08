from __future__ import annotations

from torch import Tensor


def softmax(x: Tensor, *, dim: int) -> Tensor:
    """Softmax implementation."""

    return x.softmax(dim, dtype=x.dtype)
