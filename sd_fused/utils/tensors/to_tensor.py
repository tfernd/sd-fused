from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from ...layers.base.types import Device


def to_tensor(
    x: Optional[int | float | Tensor],
    device: Optional[Device] = None,
    dtype: Optional[torch.dtype] = None,
    *,
    add_spatial: bool = False,
) -> Tensor:
    """Convert a number to a Tensor with fake channel/spatial dimensions."""

    if x is None:
        x = 0

    if isinstance(x, (int, float)):
        x = torch.tensor([x], device=device, dtype=dtype)
    else:
        assert x.ndim == 1
        x = x.to(device=device, dtype=dtype)

    if add_spatial:
        x = x.view(-1, 1, 1, 1)

    return x
