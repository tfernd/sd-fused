from __future__ import annotations
from typing import Optional

from functools import partial
from pathlib import Path
from PIL import Image
import validators
import requests
from io import BytesIO

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
    """Open an image/url as pytorch batched-Tensor (B=1 C H W)."""

    # TODO de-duplicate
    if isinstance(path, str) and validators.url(path):  # type: ignore
        # as url
        response = requests.get(path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(path)
    img = img.convert("RGB")

    resize = partial(img.resize, resample=Image.LANCZOS)

    if mode == "resize":
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

    pad = (
        dw // 2,
        dw - dw // 2,
        dh // 2,
        dh - dh // 2,
    )
    data = F.pad(data, pad, value=0)

    return data
