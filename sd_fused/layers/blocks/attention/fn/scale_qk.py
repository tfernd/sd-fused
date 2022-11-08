from __future__ import annotations

import math

from torch import Tensor


def scale_qk(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """Scale the qk tensor by the channel-dimension."""

    C = q.size(2)
    scale = math.pow(C, -1 / 4)

    return q * scale, k * scale
