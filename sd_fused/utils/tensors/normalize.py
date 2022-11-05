from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor


def normalize(data: Tensor, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Normalize a byte-Tensor to the [-1, 1] range."""

    assert data.dtype == torch.uint8

    return data.div(255 / 2).sub(1).to(dtype)


def denormalize(data: Tensor) -> Tensor:
    """Denormalize a tensor of the range [-1, 1] to a byte-Tensor."""

    assert data.requires_grad == False

    # ? what if we want gradients?
    return data.add(1).mul(255 / 2).clamp(0, 255).byte()
