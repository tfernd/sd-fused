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

from ..typing import Literal
from .open_image import open_image


ResizeModes = Literal["resize", "resize-crop", "resize-pad"]


def image2tensor(
    path: str | Path | Image.Image,
    height: Optional[int] = None,
    width: Optional[int] = None,
    mode: ResizeModes = "resize",
    device: Optional[torch.device] = None,
    rescale: Optional[float] = None,
) -> Tensor:
    """Open an image/url as pytorch batched-Tensor (B=1 C H W)."""

    img = open_image(path)
    resize = partial(img.resize, resample=Image.LANCZOS)

    if height is None or width is None:
        assert height is None and width is None

        width, height = img.size
        if rescale is not None:
            width = math.ceil(width * rescale)
            height = math.ceil(height * rescale)

    if mode == "resize" or rescale is not None:
        img = resize((width, height))
    else:
        ar = width / height
        src_ar = img.width / img.height

        diff = ar > src_ar
        if mode == "resize-pad":
            diff = not diff

        w = math.ceil(width if diff else height * src_ar)
        h = math.ceil(height if not diff else width / src_ar)

        img = resize((w, h))

    data = torch.from_numpy(np.asarray(img).copy()).to(device)
    data = rearrange(data, "H W C -> 1 C H W")

    # crop/padding size
    h, w = data.shape[-2:]
    dh, dw = height - h, width - w

    pad = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
    data = F.pad(data, pad, value=0)

    return data
