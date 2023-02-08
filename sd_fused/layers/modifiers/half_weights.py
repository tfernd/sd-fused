from __future__ import annotations
from typing_extensions import Self

from functools import wraps

import torch
from torch import Tensor

from ..base import Module


class HalfWeightsModule(Module):
    use_half_weights: bool = False

    def half_weights(self, use: bool = True) -> Self:
        self.use_half_weights = use

        if use or self.dtype == torch.float16:
            return self.half()
        return self.float()


# TODO add dtypes
def half_weights(fun):
    @wraps(fun)
    def wrapper(self: HalfWeightsModule, *args, **kwargs):
        if self.use_half_weights:
            self.float()

            # TODO use to_float
            args = tuple(a.float() if isinstance(a, Tensor) else a for a in args)
            kwargs = {k: v.float() if isinstance(v, Tensor) else v for k, v in kwargs.items()}

            out = fun(self, *args, **kwargs)
            self.half()
        else:
            out = fun(self, *args, **kwargs)

        return out

    return wrapper


# Generalize
# def to_float(self,
# x : Tensor | tuple[Tensor,...] | list[Tensor] | dict[str, Tensor| Any]

# x: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor], dict]) -> Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor], dict]:
#     if isinstance(x, torch.Tensor):
#         return x.float()
#     elif isinstance(x, (tuple, list)):
#         return type(x)(self.to_float(i) for i in x)
#     elif isinstance(x, dict):
#         return {k: self.to_float(v) for k, v in x.items()}
#     else:
#         return x
