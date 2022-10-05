from __future__ import annotations
from typing import Optional, NamedTuple

from pathlib import Path
from PIL import Image

import numpy as np

import torch
from torch import Tensor

from einops import rearrange


class ImageSize(NamedTuple):
    width: int
    height: int


def image2tensor(
    path: str | Path, *, size: Optional[int | ImageSize] = None, device: Optional[torch.device]=None
) -> Tensor:
    """Open an image as pytorch batched-Tensor (B C H W)."""

    img = Image.open(path).convert("RGB")
    if size is not None:
        if isinstance(size, int):
            size = ImageSize(size, size)
        img = img.resize(size)

    data = torch.from_numpy(np.asarray(img))
    data = data.to(device=device)
    data = rearrange(data, "H W C -> 1 C H W")

    return data
