from __future__ import annotations

from torch import Tensor


def join_spatial_dim(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    "Join height-width spatial dimensions of qkv."

    is_self_attention = q.ndim == k.ndim

    q = q.flatten(2, 3)
    if is_self_attention:
        k = k.flatten(2, 3)
        v = v.flatten(2, 3)

    return q, k, v
