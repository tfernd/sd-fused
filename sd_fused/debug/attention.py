#%%
from __future__ import annotations
from typing import Callable, Optional

import math

import seaborn as sns

import torch
from torch import Tensor
from einops import rearrange
from time import perf_counter
from tqdm.auto import tqdm, trange

from benchmark import benchmark


def scale_qk(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """Scale the qk tensor by the channel-dimension."""

    C = q.size(2)
    scale = math.pow(C, -1 / 4)

    return q * scale, k * scale


def softmax(x: Tensor, *, dim: int) -> Tensor:
    """Softmax implementation."""

    return x.softmax(dim, dtype=x.dtype)


def standard_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
) -> Tensor:
    q, k = scale_qk(q, k)

    attn = softmax(q @ k.transpose(1, 2), dim=2)
    del q, k

    return attn @ v


def chunked_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C)
    chunks: int = 2,
) -> Tensor:
    B, T, C = q.shape

    q, k = scale_qk(q, k)
    kT = k.transpose(1, 2)

    out = torch.empty_like(q)
    for i in range(0, B, chunks):
        s = slice(i, min(i + chunks, B))

        attn = softmax(q[s] @ kT[s], dim=2)
        out[s] = attn @ v[s]
        del attn

    return out


def make_args():
    # 8 4096 4096/77 40
    # 8 1024 1024/77 80
    # 8 256 256/77 160
    # 8 64 64/77 160

    q = torch.randn(8, 32**2, 80, device="cuda")
    k = torch.randn(8, 32**2, 80, device="cuda")
    v = torch.randn(8, 32**2, 80, device="cuda")

    return q, k, v


args = q, k, v = make_args()

# dt = benchmark(standard_attention, make_args, repeats=500, warmup=20, plot=True)
# dt = benchmark(chunked_attention0, make_args, repeats=500, warmup=20, plot=True)
