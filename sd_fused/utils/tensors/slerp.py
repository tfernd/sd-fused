from __future__ import annotations

import torch
from torch import Tensor


def slerp(a: Tensor, b: Tensor, t: Tensor) -> Tensor:
    "Spherical linear interpolation."

    # https://en.wikipedia.org/wiki/Slerp

    # 0 <= t <= 1
    assert t.ge(0).all() and t.le(1).all()

    assert a.shape == b.shape
    assert t.shape[0] == a.shape[0]
    assert a.ndim == 4
    assert t.ndim == 1

    t = t[:, None, None, None]

    an = a / a.norm(dim=1, keepdim=True)
    bn = b / b.norm(dim=1, keepdim=True)

    Ω = an.mul(bn).sum(1).clamp(-1, 1).acos()

    den = torch.sin(Ω)

    A = torch.sin((1 - t) * Ω)
    B = torch.sin(t * Ω)

    return (A * a + B * b) / den
