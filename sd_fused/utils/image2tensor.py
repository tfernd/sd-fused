from __future__ import annotations
from typing import Optional

from functools import partial
from pathlib import Path
from PIL import Image

import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange

from .typing import Literal


ResizeModes = Literal["resize", "resize-crop", "resize-pad"]


def image2tensor(
    path: str | Path,
    height: int,
    width: int,
    mode: ResizeModes,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Open an image as pytorch batched-Tensor (B=1 C H W)."""

    img = Image.open(path).convert("RGB")
    resize = partial(img.resize, resample=Image.LANCZOS)

    if mode == "resize":
        img = resize((width, height))
    else:
        ar = width / height
        src_ar = img.width / img.height

        diff = ar > src_ar
        if mode == "resize-pad":
            diff = not diff

        w = width if diff else math.ceil(height * src_ar)
        h = height if not diff else math.ceil(width / src_ar)

        img = resize((w, h))

    data = torch.from_numpy(np.asarray(img)).to(device)
    data = rearrange(data, "H W C -> 1 C H W")

    # crop/padding size
    h, w = data.shape[-2:]
    dh, dw = height - h, width - w

    pad = (
        dw // 2,
        dw - dw // 2,
        dh // 2,
        dh - dh // 2,
    )
    data = F.pad(data, pad, value=0)

    return data
