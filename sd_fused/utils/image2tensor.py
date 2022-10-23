from __future__ import annotations
from typing import Optional, NamedTuple

from pathlib import Path
from PIL import Image

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange


class ImageSize(NamedTuple):
    width: int
    height: int


def image2tensor(
    path: str | Path,
    *,
    size: Optional[int | tuple[int, int]] = None,
    device: Optional[torch.device] = None
) -> Tensor:
    """Open an image as pytorch batched-Tensor (B C H W)."""

    img = Image.open(path).convert("RGB")
    data = torch.from_numpy(np.asarray(img))

    if size is not None:
        if isinstance(size, int):
            size = (size, size)

        data = data.float()
        data = rearrange(data, "H W C -> 1 C H W")
        data = F.interpolate(
            data, size, align_corners=True, antialias=True, mode="bicubic"
        )
        data = data.clamp(0, 255).byte()
        data = rearrange(data, "1 C H W -> H W C")

    data = data.to(device=device)
    data = rearrange(data, "H W C -> 1 C H W")

    return data
