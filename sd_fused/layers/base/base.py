from __future__ import annotations

import torch

from .types import Device


class Base:
    dtype: torch.dtype = torch.float16
    device: Device = "cuda"
