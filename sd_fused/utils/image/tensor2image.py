from __future__ import annotations

from PIL import Image

import torch
from torch import Tensor
from einops import rearrange


@torch.no_grad()
def tensor2image(data: Tensor) -> Image.Image:
    assert data.dtype == torch.uint8
    assert data.size(0) == 1

    data = rearrange(data, "1 C H W -> H W C").cpu()

    return Image.fromarray(data.numpy())
