from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
import math

from einops import rearrange

from .....utils.typing import Protocol, Literal


Modes = Literal["sum", "mean"]


class Merge(Protocol):
    def __call__(self, x: Tensor, mode: Modes = "mean") -> Tensor:
        ...


def tome(metric: Tensor, r: int | float) -> Merge:
    # https://arxiv.org/abs/2210.09461
    # https://github.com/facebookresearch/ToMe/blob/main/tome/merge.py

    B, T, C = metric.shape

    if not isinstance(r, int):
        r = math.floor(r * T)

    assert 0 < r <= T / 2  # max 50% reduction

    with torch.no_grad():
        metric = metric / metric.norm(dim=2, keepdim=True)
        a, b = metric[:, ::2], metric[:, 1::2]
        score = a @ b.transpose(1, 2)
        del a, b

        node_max, node_idx = score.max(dim=2, keepdim=True)
        del score
        edge_idx = node_max.argsort(dim=1, descending=True)

        # Unmerged/Merged Tokens
        size = (math.ceil(T / 2) - r, r)
        unmerged_idx, src_idx = edge_idx.split(size, dim=1)

        dst_idx = node_idx.gather(dim=1, index=src_idx)

    def merge(x: Tensor, mode: Modes = "mean") -> Tensor:
        src, dst = x[:, ::2], x[:, 1::2]
        B, T, C = src.shape

        unmerged = src.gather(dim=1, index=unmerged_idx.expand(-1, -1, C))
        src = src.gather(dim=1, index=src_idx.expand(-1, -1, C))
        dst = dst.scatter_reduce(1, dst_idx.expand(-1, -1, C), src, reduce=mode)

        return torch.cat([unmerged, dst], dim=1)

    return merge


def merge_weighted_average(
    merge: Merge,
    x: Tensor,
    size: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    B, T, C = x.shape

    if size is None:
        size = torch.ones(B, T, 1, dtype=x.dtype, device=x.device)

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size

    return x, size


def token_average(
    k: Tensor,
    v: Tensor,
    r: int | float,
) -> tuple[Tensor, Tensor, Tensor]:
    merge = tome(k, r)

    k, size = merge_weighted_average(merge, k)
    v, size = merge_weighted_average(merge, v)
    bias = size.log()

    return k, v, bias
