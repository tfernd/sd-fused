from __future__ import annotations
from typing import NamedTuple

from torch import Tensor

class OutputStates(NamedTuple):
    x: Tensor
    states: list[Tensor]