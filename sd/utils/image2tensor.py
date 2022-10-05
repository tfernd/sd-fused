from __future__ import annotations
from typing import Optional

from pathlib import Path
from PIL import Image

import numpy as np

import torch
from torch import Tensor

from einops import rearrange


def image2tensor(
    path: str | Path, *, resize: Optional[int | tuple[int, int]] = None,
) -> Tensor:
    """Open an image as pytorch batched-Tensor (B C H W)."""

    img = Image.open(path).convert("RGB")
    if resize is not None:
        if isinstance(resize, int):
            resize = (resize, resize)
        img = img.resize(resize)

    data = torch.from_numpy(np.asarray(img))
    data = rearrange(data, "H W C -> 1 C H W")

    return data
