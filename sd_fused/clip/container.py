from __future__ import annotations
from typing import NamedTuple, Optional

from torch import Tensor


class TensorAndWeight(NamedTuple):
    tensor: Tensor
    weight: Tensor


class TensorAndMaybeWeight(NamedTuple):
    tensor: Tensor
    weight: Optional[Tensor] = None
