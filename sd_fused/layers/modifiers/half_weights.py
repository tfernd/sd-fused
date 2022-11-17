from __future__ import annotations
from typing_extensions import Self

from functools import wraps

import torch
from torch import Tensor

from ..base.module import Module


class HalfWeightsModule(Module):
    use_half_weights: bool = False

    def half_weights(self, use: bool = True) -> Self:
        self.use_half_weights = use

        return self.half() if use or self.dtype == torch.float16 else self.float()


def half_weights(fun):
    @wraps(fun)
    def wrapper(self: HalfWeightsModule, *args, **kwargs):
        if self.use_half_weights:
            self.float()

            args = tuple(a.float() if isinstance(a, Tensor) else a for a in args)
            kwargs = {k: v.float() if isinstance(v, Tensor) else v for k, v in kwargs.items()}

            out = fun(self, *args, **kwargs)
            self.half()
        else:
            out = fun(self, *args, **kwargs)

        return out

    return wrapper
