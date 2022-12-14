from __future__ import annotations

from PIL import Image

import torch
from torch import Tensor
from einops import rearrange


def tensor2images(data: Tensor) -> list[Image.Image]:
    """Creates a list of images according to the batch size."""

    assert data.dtype == torch.uint8

    data = rearrange(data, "B C H W -> B H W C").cpu().numpy()

    return [Image.fromarray(v) for v in data]
