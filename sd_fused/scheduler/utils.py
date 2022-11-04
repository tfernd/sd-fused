from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor


def to_tensor(
    x: int | float | Tensor,
    *,
    steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Convert a number to a Tensor with fake channel/spatial dimensions."""

    if isinstance(x, (int, float)):
        x = torch.tensor([x], device=device, dtype=dtype)
    else:
        assert x.ndim == 1
        x = x.to(device=device, dtype=dtype)

    if steps is not None:
        assert torch.all(0 <= x) and torch.all(x < steps)

    x = x.view(-1, 1, 1, 1)

    return x
