from __future__ import annotations
from typing import Optional

import torch.nn as nn

from .half_weights import HalfWeights
from .ops import Ops


class Linear(Ops, HalfWeights, nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        *,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features

        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias,
        )
