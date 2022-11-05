from __future__ import annotations

import math

from torch import Tensor


def scale_qk(*xs: Tensor) -> tuple[Tensor, Tensor]:
    """Scale the qk tensor by the channel-dimension."""

    assert len(xs) == 2

    return tuple(x * math.pow(x.size(-1), -1 / 4) for x in xs)
