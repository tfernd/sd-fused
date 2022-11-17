from __future__ import annotations
from typing import Callable

import seaborn as sns

from matplotlib import pyplot as plt
import math
import torch
from torch import Tensor
from time import perf_counter
from tqdm.auto import tqdm, trange
import gc
import torch


def clear_cuda() -> None:
    """Clear CUDA memory and garbage collection."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def benchmark(fn: Callable, make_args: Callable, repeats: int, warmup: int, plot: bool = True) -> Tensor:
    clear_cuda()

    for _ in trange(warmup):
        torch.cuda.synchronize()
        args = make_args()
        out = fn(*args)
        torch.cuda.synchronize()

        del args, out

    dts: list[float] = []
    for n in trange(repeats):
        args = make_args()

        torch.cuda.synchronize()
        t0 = perf_counter()
        out = fn(*args)
        torch.cuda.synchronize()
        dt = perf_counter() - t0

        dts.append(dt)

        del args, out
    clear_cuda()

    out = torch.tensor(dts)
    mu = out.mean().item()

    exp = math.floor(math.log10(mu) / 3) * 3
    scale = 10**exp

    out /= scale

    mu, std = out.mean().item(), out.std().item()
    print(f"{mu:.3f} ± {std:.3f} (1e{exp}s)")

    if plot:
        sns.histplot(out.numpy(), bins=repeats)  # type: ignore
        plt.show()

    return out
