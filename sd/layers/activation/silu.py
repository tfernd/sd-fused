from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid().mul_(x)
