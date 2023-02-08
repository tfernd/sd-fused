from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .base import Base


class Parameter(Base, nn.Parameter):
    ...


class Buffer(Parameter):
    def __new__(cls, data: Tensor) -> Tensor:
        data = data.detach()
        data.to(dtype=cls.dtype, device=cls.device)

        return data
