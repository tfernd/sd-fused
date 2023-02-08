from __future__ import annotations

import math

from torch import Tensor


def scale_qk(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    "Scale the query and key."

    C = q.size(-1)
    scale = math.pow(C, -1 / 4)

    return q * scale, k * scale
